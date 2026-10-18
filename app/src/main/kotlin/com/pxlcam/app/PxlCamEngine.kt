package com.pxlcam.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.*
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.view.Surface
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Hardware-accelerated Camera2 + MediaCodec H.264 studio streaming engine.
 *
 * Includes the critical MediaTek / Android Camera2 hardware ISP fixes:
 *  - Uses CameraDevice.TEMPLATE_RECORD without overriding TONEMAP_MODE to HIGH_QUALITY
 *    (which resets the 64-point hardware sRGB + ADRC gamma LUT to a dark 2-point linear ramp).
 *  - Preserves native TONEMAP_MODE_FAST and NOISE_REDUCTION_MODE_FAST for 30fps zero-lag
 *    scrcpy-grade brightness and clarity.
 *  - Defaults to CONTROL_AWB_MODE_AUTO (never auto-switching to SHADE in warm indoor light).
 *  - Streams 26-byte 'PXLC' framed H.264 NAL packets over TCP :9876 and accepts live JSON
 *    hardware control commands from macOS Pxl Cam.app and the OBS Custom Dock.
 */
class PxlCamEngine(private val context: Context) {

    interface Listener {
        fun onEngineStateChanged(
            streaming: Boolean,
            connectedClients: Int,
            fps: Float,
            cameraIndex: Int,
            flashOn: Boolean,
            torchStrengthPercent: Int,
            evOffset: Int,
            zoomRatio: Float,
            wbMode: Int,
            afMode: Int
        )
    }

    companion object {
        private const val TAG = "PxlCamEngine"
        private const val MAGIC_PXLC = 0x50584C43
        private const val CODEC_H264: Byte = 2
        const val STREAM_PORT = 9876
    }

    @Volatile var currentCameraIndex: Int = 0
        private set
    @Volatile var streamWidth: Int = 1920
        private set
    @Volatile var streamHeight: Int = 1080
        private set
    @Volatile var isFlashOn: Boolean = false
        private set
    @Volatile var torchStrengthPercent: Int = 100
        private set
    @Volatile var maxHwTorchStrengthLevel: Int = 1
        private set
    @Volatile var exposureOffset: Int = 0
        private set
    @Volatile var zoomRatio: Float = 1.0f
        private set
    @Volatile var wbModeIndex: Int = 0
        private set
    @Volatile var afModeIndex: Int = 0
        private set
    @Volatile var currentFps: Float = 0.0f
        private set

    private var listener: Listener? = null
    private var previewSurface: Surface? = null

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var encoderThread: HandlerThread? = null
    private var encoderHandler: Handler? = null

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var activeCharacteristics: CameraCharacteristics? = null
    private var mediaCodec: MediaCodec? = null
    private var encoderInputSurface: Surface? = null

    @Volatile private var cachedCodecConfig: ByteArray? = null
    @Volatile private var sensorRotationQuadrants: Int = 0

    private val running = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null
    private val clients = CopyOnWriteArrayList<ClientSession>()
    private val frameCounter = AtomicInteger(0)
    private var lastFpsTimestampMs: Long = System.currentTimeMillis()

    private class ClientSession(
        val socket: Socket,
        val output: DataOutputStream,
        val needsConfig: AtomicBoolean = AtomicBoolean(true)
    )

    fun setListener(l: Listener?) {
        listener = l
        notifyListener()
    }

    fun setPreviewSurface(surface: Surface?) {
        previewSurface = surface
        if (running.get()) {
            cameraHandler?.post { reopenCameraAndEncoder() }
        }
    }

    fun start(width: Int = 1920, height: Int = 1080) {
        streamWidth = width
        streamHeight = height
        if (running.getAndSet(true)) return

        cameraThread = HandlerThread("PxlCamCameraThread").apply { start() }
        cameraHandler = Handler(cameraThread!!.looper)
        encoderThread = HandlerThread("PxlCamEncoderThread").apply { start() }
        encoderHandler = Handler(encoderThread!!.looper)

        startTcpServerThread()
        cameraHandler?.post { reopenCameraAndEncoder() }
        notifyListener()
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        try {
            serverSocket?.close()
        } catch (_: Throwable) {}
        serverSocket = null

        for (c in clients) {
            try { c.socket.close() } catch (_: Throwable) {}
        }
        clients.clear()

        cameraHandler?.post {
            closeCameraAndEncoder()
            cameraThread?.quitSafely()
            encoderThread?.quitSafely()
        }
        notifyListener()
    }

    fun setLens(index: Int) {
        val target = if (index == 1) 1 else 0
        if (target != currentCameraIndex) {
            currentCameraIndex = target
            if (currentCameraIndex == 1) isFlashOn = false
            cameraHandler?.post { reopenCameraAndEncoder() }
            notifyListener()
        }
    }

    fun setFlash(enabled: Boolean) {
        if (currentCameraIndex == 1 && enabled) return
        isFlashOn = enabled
        if (enabled && torchStrengthPercent == 0) {
            torchStrengthPercent = 100
        }
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    fun setTorchStrength(percent: Int) {
        val clamped = percent.coerceIn(0, 100)
        torchStrengthPercent = clamped
        if (clamped == 0) {
            isFlashOn = false
        } else if (currentCameraIndex == 0) {
            isFlashOn = true
        }
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    fun setExposureCompensation(ev: Int) {
        exposureOffset = ev.coerceIn(-12, 12)
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    fun setZoom(zoom: Float) {
        zoomRatio = zoom.coerceIn(1.0f, 4.0f)
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    fun setWhiteBalance(mode: Int) {
        wbModeIndex = mode.coerceIn(0, 5)
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    fun setFocusMode(mode: Int) {
        afModeIndex = mode.coerceIn(0, 3)
        cameraHandler?.post { updateRepeatingRequest() }
        notifyListener()
    }

    private fun startTcpServerThread() {
        Thread({
            try {
                val srv = ServerSocket()
                srv.reuseAddress = true
                srv.bind(InetSocketAddress(STREAM_PORT))
                serverSocket = srv
                Log.i(TAG, "PXLC TCP Server listening on :$STREAM_PORT")

                while (running.get()) {
                    val sock = srv.accept()
                    sock.tcpNoDelay = true
                    sock.sendBufferSize = 512 * 1024
                    val out = DataOutputStream(BufferedOutputStream(sock.getOutputStream(), 64 * 1024))
                    val session = ClientSession(sock, out)
                    clients.add(session)
                    notifyListener()

                    cameraHandler?.post { requestImmediateKeyframe() }

                    Thread({
                        handleClientControlLoop(session)
                    }, "PxlCamClientCtrl").start()
                }
            } catch (e: Throwable) {
                if (running.get()) {
                    Log.e(TAG, "TCP Server error", e)
                }
            }
        }, "PxlCamTcpAccept").start()
    }

    private fun handleClientControlLoop(session: ClientSession) {
        try {
            val reader = BufferedReader(InputStreamReader(session.socket.getInputStream(), Charsets.UTF_8))
            while (running.get() && !session.socket.isClosed) {
                val line = reader.readLine() ?: break
                val trimmed = line.trim()
                if (trimmed.isEmpty()) continue
                if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                    applyJsonControl(trimmed)
                } else if (trimmed.contains(":")) {
                    applyColonCommand(trimmed)
                }
            }
        } catch (_: Throwable) {
        } finally {
            clients.remove(session)
            try { session.socket.close() } catch (_: Throwable) {}
            notifyListener()
        }
    }

    private fun applyColonCommand(cmd: String) {
        try {
            val parts = cmd.split(":", limit = 2)
            if (parts.size < 2) return
            val key = parts[0].trim().uppercase()
            val value = parts[1].trim()
            when (key) {
                "LENS" -> {
                    val idx = value.toIntOrNull() ?: return
                    setLens(idx)
                }
                "FLASH" -> {
                    val mode = value.toIntOrNull() ?: 0
                    setFlash(mode != 0)
                }
                "TORCH_STRENGTH", "FLASH_LEVEL" -> {
                    val pct = value.toIntOrNull() ?: return
                    setTorchStrength(pct)
                }
                "EV" -> {
                    val ev = value.toIntOrNull() ?: return
                    setExposureCompensation(ev)
                }
                "ZOOM" -> {
                    val z = value.toFloatOrNull() ?: return
                    setZoom(z)
                }
                "WB" -> {
                    val wb = value.toIntOrNull() ?: return
                    setWhiteBalance(wb)
                }
                "AFMODE" -> {
                    val af = value.toIntOrNull() ?: return
                    setFocusMode(af)
                }
                "RES" -> {
                    val wh = value.lowercase().split("x")
                    if (wh.size == 2) {
                        val w = wh[0].toIntOrNull()
                        val h = wh[1].toIntOrNull()
                        if (w != null && h != null && (w != streamWidth || h != streamHeight)) {
                            streamWidth = w.coerceIn(640, 3840)
                            streamHeight = h.coerceIn(480, 2160)
                            cameraHandler?.post { reopenCameraAndEncoder() }
                            notifyListener()
                        }
                    }
                }
                "AF" -> {
                    if (value.equals("TRIGGER", ignoreCase = true)) {
                        cameraHandler?.post { updateRepeatingRequest() }
                    }
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Failed to parse colon command: $cmd", e)
        }
    }

    private fun applyJsonControl(jsonStr: String) {
        try {
            val obj = JSONObject(jsonStr)
            var reopenNeeded = false

            if (obj.has("camera")) {
                val requestedLens = if (obj.optInt("camera", 0) == 1) 1 else 0
                if (requestedLens != currentCameraIndex) {
                    currentCameraIndex = requestedLens
                    if (currentCameraIndex == 1) isFlashOn = false
                    reopenNeeded = true
                }
            }
            if (obj.has("flash")) {
                val f = obj.optBoolean("flash", false) || obj.optInt("flash", 0) == 1
                isFlashOn = if (currentCameraIndex == 0) f else false
                if (isFlashOn && torchStrengthPercent == 0) {
                    torchStrengthPercent = 100
                }
            }
            if (obj.has("flashLevel") || obj.has("torchStrength")) {
                val pct = if (obj.has("flashLevel")) obj.optInt("flashLevel", torchStrengthPercent)
                else obj.optInt("torchStrength", torchStrengthPercent)
                torchStrengthPercent = pct.coerceIn(0, 100)
                if (torchStrengthPercent == 0) {
                    isFlashOn = false
                } else if (currentCameraIndex == 0) {
                    isFlashOn = true
                }
            }
            if (obj.has("ev")) {
                exposureOffset = obj.optInt("ev", exposureOffset).coerceIn(-12, 12)
            }
            if (obj.has("zoom")) {
                zoomRatio = obj.optDouble("zoom", zoomRatio.toDouble()).toFloat().coerceIn(1.0f, 4.0f)
            }
            if (obj.has("wb")) {
                wbModeIndex = obj.optInt("wb", wbModeIndex).coerceIn(0, 5)
            }
            if (obj.has("af")) {
                afModeIndex = obj.optInt("af", afModeIndex).coerceIn(0, 3)
            }

            if (reopenNeeded) {
                cameraHandler?.post { reopenCameraAndEncoder() }
            } else {
                cameraHandler?.post { updateRepeatingRequest() }
            }
            notifyListener()
        } catch (e: Throwable) {
            Log.w(TAG, "Failed to parse JSON control: $jsonStr", e)
        }
    }

    @SuppressLint("MissingPermission")
    private fun reopenCameraAndEncoder() {
        closeCameraAndEncoder()
        if (!running.get()) return

        try {
            val mgr = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = mgr.cameraIdList
            if (cameraIds.isEmpty()) return

            val targetFacing = if (currentCameraIndex == 1) {
                CameraCharacteristics.LENS_FACING_FRONT
            } else {
                CameraCharacteristics.LENS_FACING_BACK
            }

            var selectedId = cameraIds[0]
            for (id in cameraIds) {
                val chars = mgr.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing == targetFacing) {
                    selectedId = id
                    break
                }
            }

            val chars = mgr.getCameraCharacteristics(selectedId)
            activeCharacteristics = chars
            maxHwTorchStrengthLevel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                chars.get(CameraCharacteristics.FLASH_INFO_STRENGTH_MAXIMUM_LEVEL) ?: 1
            } else {
                1
            }
            val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            sensorRotationQuadrants = ((sensorOrientation / 90) % 4 + 4) % 4

            setupMediaCodec()

            mgr.openCamera(selectedId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession(camera)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "CameraDevice error: $error")
                    camera.close()
                    cameraDevice = null
                }
            }, cameraHandler)
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to open camera", e)
        }
    }

    private fun setupMediaCodec() {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, streamWidth, streamHeight).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, 12_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
            setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT709)
            setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
            setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileHigh)
                setInteger(MediaFormat.KEY_LEVEL, MediaCodecInfo.CodecProfileLevel.AVCLevel41)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            }
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.setCallback(object : MediaCodec.Callback() {
            override fun onInputBufferAvailable(mc: MediaCodec, index: Int) {}

            override fun onOutputBufferAvailable(mc: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
                try {
                    if (info.size > 0) {
                        val buf = mc.getOutputBuffer(index)
                        if (buf != null) {
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            val bytes = ByteArray(info.size)
                            buf.get(bytes)

                            if ((info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                                cachedCodecConfig = bytes
                                broadcastEncodedPacket(bytes)
                            } else {
                                val isKey = (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0
                                val cfg = cachedCodecConfig
                                val payload = if (isKey && cfg != null) {
                                    ByteArray(cfg.size + bytes.size).also {
                                        System.arraycopy(cfg, 0, it, 0, cfg.size)
                                        System.arraycopy(bytes, 0, it, cfg.size, bytes.size)
                                    }
                                } else {
                                    bytes
                                }
                                broadcastEncodedPacket(payload)
                                updateFpsTelemetry()
                            }
                        }
                    }
                } catch (_: Throwable) {
                } finally {
                    try { mc.releaseOutputBuffer(index, false) } catch (_: Throwable) {}
                }
            }

            override fun onError(mc: MediaCodec, e: MediaCodec.CodecException) {
                Log.e(TAG, "MediaCodec error", e)
            }

            override fun onOutputFormatChanged(mc: MediaCodec, format: MediaFormat) {
                val sps = format.getByteBuffer("csd-0")
                val pps = format.getByteBuffer("csd-1")
                if (sps != null && pps != null) {
                    val spsBytes = ByteArray(sps.remaining())
                    sps.duplicate().get(spsBytes)
                    val ppsBytes = ByteArray(pps.remaining())
                    pps.duplicate().get(ppsBytes)
                    val combined = ByteArray(spsBytes.size + ppsBytes.size)
                    System.arraycopy(spsBytes, 0, combined, 0, spsBytes.size)
                    System.arraycopy(ppsBytes, 0, combined, spsBytes.size, ppsBytes.size)
                    cachedCodecConfig = combined
                }
            }
        }, encoderHandler)

        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoderInputSurface = codec.createInputSurface()
        codec.start()
        mediaCodec = codec
    }

    private fun createCaptureSession(camera: CameraDevice) {
        val encSurface = encoderInputSurface ?: return
        val targets = mutableListOf(encSurface)
        previewSurface?.takeIf { it.isValid }?.let { targets.add(it) }

        camera.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                updateRepeatingRequest()
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "CameraCaptureSession configure failed")
            }
        }, cameraHandler)
    }

    private fun updateRepeatingRequest() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        val encSurface = encoderInputSurface ?: return

        try {
            // CRITICAL FIX: Use TEMPLATE_RECORD and preserve MediaTek's native 64-point hardware
            // sRGB + ADRC tone curve (TONEMAP_MODE_FAST) and hardware 3DNR (NOISE_REDUCTION_MODE_FAST).
            // Never override TONEMAP_MODE to HIGH_QUALITY without an explicit custom curve.
            val req = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
            req.addTarget(encSurface)
            previewSurface?.takeIf { it.isValid }?.let { req.addTarget(it) }

            req.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
            req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)

            activeCharacteristics?.let { chars ->
                val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                var chosenRange: Range<Int>? = null
                if (fpsRanges != null) {
                    // Prefer [15, 20] or [10, 20] on MediaTek TEMPLATE_RECORD so hardware shutter
                    // integrates for 50ms-66.7ms instead of pinning at 33.3ms (1/30s) at max ISO 3200
                    for (r in fpsRanges) {
                        if (r.upper == 20 && r.lower < r.upper) {
                            if (chosenRange == null || r.lower > chosenRange.lower) {
                                chosenRange = r
                            }
                        }
                    }
                    if (chosenRange == null) {
                        for (r in fpsRanges) {
                            if (r.upper == 30 && r.lower <= 15) {
                                if (chosenRange == null || r.lower > chosenRange.lower) {
                                    chosenRange = r
                                }
                            }
                        }
                    }
                    if (chosenRange == null && fpsRanges.isNotEmpty()) {
                        chosenRange = fpsRanges.maxByOrNull { it.upper }
                    }
                }
                if (chosenRange != null) {
                    req.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, chosenRange)
                }

                val evRange = chars.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
                if (evRange != null) {
                    var effectiveEv = exposureOffset
                    if (isFlashOn && currentCameraIndex == 0 && torchStrengthPercent in 1..99 && maxHwTorchStrengthLevel <= 1) {
                        val flashDimSteps = kotlin.math.round(((torchStrengthPercent - 100) / 100.0f) * 8.0f).toInt()
                        effectiveEv += flashDimSteps
                    }
                    val clampedEv = effectiveEv.coerceIn(evRange.lower, evRange.upper)
                    req.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, clampedEv)
                }

                // Hardware Zoom via SCALER_CROP_REGION / CONTROL_ZOOM_RATIO
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    req.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoomRatio)
                } else {
                    val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                    if (activeArray != null && zoomRatio > 1.0f) {
                        val cropW = (activeArray.width() / zoomRatio).toInt()
                        val cropH = (activeArray.height() / zoomRatio).toInt()
                        val left = (activeArray.width() - cropW) / 2
                        val top = (activeArray.height() - cropH) / 2
                        req.set(CaptureRequest.SCALER_CROP_REGION, Rect(left, top, left + cropW, top + cropH))
                    }
                }
            }

            // Flash Torch + Hardware/Optical Torch Dimmer
            val torchActive = isFlashOn && currentCameraIndex == 0 && torchStrengthPercent > 0
            req.set(
                CaptureRequest.FLASH_MODE,
                if (torchActive) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF
            )
            if (torchActive && Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM && maxHwTorchStrengthLevel > 1) {
                val hwLevel = (1 + kotlin.math.round((torchStrengthPercent / 100.0f) * (maxHwTorchStrengthLevel - 1)).toInt())
                    .coerceIn(1, maxHwTorchStrengthLevel)
                try {
                    req.set(CaptureRequest.FLASH_STRENGTH_LEVEL, hwLevel)
                } catch (_: Throwable) {}
            }

            // White Balance (Default: AUTO so indoor lighting is never tinted rosy by SHADE)
            val awbMode = when (wbModeIndex) {
                1 -> CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
                2 -> CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
                3 -> CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
                4 -> CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
                5 -> CaptureRequest.CONTROL_AWB_MODE_SHADE
                else -> CaptureRequest.CONTROL_AWB_MODE_AUTO
            }
            req.set(CaptureRequest.CONTROL_AWB_MODE, awbMode)

            // Focus Mode
            when (afModeIndex) {
                1 -> req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                2 -> {
                    req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                    req.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_IDLE)
                }
                3 -> {
                    req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                    req.set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.0f)
                }
                else -> req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            }

            session.setRepeatingRequest(req.build(), null, cameraHandler)
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to update repeating request", e)
        }
    }

    private fun requestImmediateKeyframe() {
        try {
            val b = Bundle().apply {
                putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
            }
            mediaCodec?.setParameters(b)
        } catch (_: Throwable) {}
    }

    private fun broadcastEncodedPacket(payload: ByteArray) {
        if (clients.isEmpty()) return
        val packedCamAndRot = ((currentCameraIndex and 0x03) or ((sensorRotationQuadrants and 0x03) shl 2)).toByte()
        val nowNs = System.nanoTime()

        val header = ByteBuffer.allocate(26).apply {
            putInt(MAGIC_PXLC)
            put(CODEC_H264)
            put(packedCamAndRot)
            putShort(streamWidth.toShort())
            putShort(streamHeight.toShort())
            putInt(12_000_000)
            putLong(nowNs)
            putInt(payload.size)
        }.array()

        for (client in clients) {
            try {
                synchronized(client.output) {
                    if (client.needsConfig.getAndSet(false)) {
                        cachedCodecConfig?.let { cfg ->
                            val cfgHeader = ByteBuffer.allocate(26).apply {
                                putInt(MAGIC_PXLC)
                                put(CODEC_H264)
                                put(packedCamAndRot)
                                putShort(streamWidth.toShort())
                                putShort(streamHeight.toShort())
                                putInt(12_000_000)
                                putLong(nowNs)
                                putInt(cfg.size)
                            }.array()
                            client.output.write(cfgHeader)
                            client.output.write(cfg)
                        }
                    }
                    client.output.write(header)
                    client.output.write(payload)
                    client.output.flush()
                }
            } catch (_: Throwable) {
                clients.remove(client)
                try { client.socket.close() } catch (_: Throwable) {}
                notifyListener()
            }
        }
    }

    private fun updateFpsTelemetry() {
        val count = frameCounter.incrementAndGet()
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTimestampMs
        if (elapsed >= 1000L) {
            currentFps = (count * 1000.0f) / elapsed.toFloat()
            frameCounter.set(0)
            lastFpsTimestampMs = now
            notifyListener()
        }
    }

    private fun closeCameraAndEncoder() {
        try { captureSession?.close() } catch (_: Throwable) {}
        captureSession = null
        try { cameraDevice?.close() } catch (_: Throwable) {}
        cameraDevice = null
        try { mediaCodec?.stop() } catch (_: Throwable) {}
        try { mediaCodec?.release() } catch (_: Throwable) {}
        mediaCodec = null
        try { encoderInputSurface?.release() } catch (_: Throwable) {}
        encoderInputSurface = null
    }

    private fun notifyListener() {
        listener?.onEngineStateChanged(
            streaming = running.get(),
            connectedClients = clients.size,
            fps = currentFps,
            cameraIndex = currentCameraIndex,
            flashOn = isFlashOn,
            torchStrengthPercent = torchStrengthPercent,
            evOffset = exposureOffset,
            zoomRatio = zoomRatio,
            wbMode = wbModeIndex,
            afMode = afModeIndex
        )
    }
}
