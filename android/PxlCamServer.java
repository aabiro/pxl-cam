package com.pxlcam.server;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.Process;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;
import android.view.Surface;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class PxlCamServer {
    private static final int PORT = 9876;
    private static final int MAGIC = 0x50584C43; // "PXLC"

    private static CameraManager cameraManager;
    private static HandlerThread cameraThread;
    private static Handler cameraHandler;

    private static CameraDevice currentDevice;
    private static CameraCaptureSession currentSession;
    private static MediaCodec currentEncoder;
    private static Surface encoderInputSurface;
    private static ImageReader currentReader;
    private static CaptureRequest.Builder currentRequestBuilder;
    private static Rect activeArraySize;
    private static Range<Integer>[] availableFpsRanges;
    private static Range<Integer> aeCompRange;

    // Live configurable camera state (Defaults: Back Camera 0 at 1920x1080 Hardware Video ISP 3DNR mode — identical to scrcpy)
    private static volatile String cameraId = "0";
    private static volatile int requestedWidth = 1920;
    private static volatile int requestedHeight = 1080;
    private static volatile int activeWidth = 1920;
    private static volatile int activeHeight = 1080;
    private static volatile float zoomFactor = 1.0f;
    private static volatile int exposureComp = 0; // Keep sensor digital ISO clean; lift exposure in 32-bit Metal ISP
    private static volatile int flashMode = 0;    // 0 = Off, 1 = On (Torch), 2 = Auto Flash
    private static volatile int torchStrengthPercent = 100; // 0..100% Flash/Torch dimmer level
    private static volatile int maxHardwareTorchStrength = 1;
    private static volatile CaptureRequest.Key<Byte> keyFlashFiringPower = null;
    private static volatile CaptureRequest.Key<Byte> keyMtkTorchDuty = null;
    private static volatile CaptureRequest.Key<Integer> keyAndroid15FlashStrength = null;
    private static volatile int afMode = 0;       // 0 = Continuous Video, 1 = Macro, 2 = Locked
    private static volatile int wbMode = 0;       // 0 = Auto, 1 = Daylight, 2 = Cloudy, 3 = Incandescent, 4 = Fluorescent, 5 = Shade
    private static volatile int fpsMode = 0;      // 0 = Low-Light Auto (10-30 / 15-30), 1 = Locked 30fps, 2 = Max Night (5-30)
    private static volatile boolean aeLocked = false;
    private static volatile boolean awbLocked = false;
    private static volatile byte jpegQuality = 90;
    private static volatile boolean screenOff = true;

    // Live hardware accelerometer orientation tracking (0=Landscape 0°, 1=Portrait 90° CW, 2=Landscape 180°, 3=Portrait 270° CW)
    private static volatile int sensorOrientation = 90;
    private static volatile int lensFacing = CameraCharacteristics.LENS_FACING_BACK;
    private static volatile int deviceOrientationDegrees = 270; // Default Landscape Left (270° -> 0° camera rotation)
    private static volatile int autoRotationQuadrants = 0;

    private static final AtomicLong lastFrameTimeMs = new AtomicLong(SystemClock.elapsedRealtime());
    private static final AtomicInteger frameSequence = new AtomicInteger(0);
    private static final AtomicBoolean encoderRunning = new AtomicBoolean(false);
    private static volatile byte[] codecConfigBytes = null;
    private static final LinkedBlockingQueue<FrameData> frameQueue = new LinkedBlockingQueue<>(24);
    private static final AtomicBoolean reconfiguring = new AtomicBoolean(false);

    private static final class FrameData {
        final byte[] payloadBytes;
        final int width;
        final int height;
        final int camera;
        final int seq;
        final long timestampNs;
        final int codec; // 2 = Hardware Video ISP H.264 (scrcpy pipeline), 1 = JPEG

        FrameData(byte[] payloadBytes, int width, int height, int camera, int seq, long timestampNs, int codec) {
            this.payloadBytes = payloadBytes;
            this.width = width;
            this.height = height;
            this.camera = camera;
            this.seq = seq;
            this.timestampNs = timestampNs;
            this.codec = codec;
        }
    }

    public static void main(String[] args) {
        try {
            if (args.length >= 1) {
                cameraId = args[0];
            }
            if (args.length >= 3) {
                requestedWidth = Integer.parseInt(args[1]);
                requestedHeight = Integer.parseInt(args[2]);
            }

            System.out.println("[PxlCam] Initializing Android system context & workarounds...");
            Looper.prepareMainLooper();

            Class<?> workaroundsClass = Class.forName("com.genymobile.scrcpy.Workarounds");
            Method applyMethod = workaroundsClass.getDeclaredMethod("apply");
            applyMethod.setAccessible(true);
            applyMethod.invoke(null);

            Class<?> serviceMgrClass = Class.forName("com.genymobile.scrcpy.wrappers.ServiceManager");
            Method getCamMgrMethod = serviceMgrClass.getDeclaredMethod("getCameraManager");
            getCamMgrMethod.setAccessible(true);
            cameraManager = (CameraManager) getCamMgrMethod.invoke(null);

            cameraThread = new HandlerThread("PxlCamHardwareThread", Process.THREAD_PRIORITY_URGENT_DISPLAY);
            cameraThread.start();
            cameraHandler = new Handler(cameraThread.getLooper());

            setPhysicalScreenPower(false);
            startOrientationSensorListener(workaroundsClass);
            startCameraPipeline();
            startWatchdogThread();

            try (ServerSocket serverSocket = new ServerSocket(PORT)) {
                serverSocket.setReuseAddress(true);
                System.out.println("[PxlCam] Listening on 127.0.0.1:" + PORT + " (Camera=" + cameraId + " " + activeWidth + "x" + activeHeight + ")");

                while (true) {
                    Socket client = serverSocket.accept();
                    client.setTcpNoDelay(true);
                    client.setSendBufferSize(2 * 1024 * 1024);
                    System.out.println("[PxlCam] Mac receiver connected!");
                    handleClient(client);
                }
            }
        } catch (Throwable t) {
            t.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void handleClient(Socket client) {
        AtomicBoolean connected = new AtomicBoolean(true);

        // Flush stale frames and request an immediate IDR keyframe so Mac locks on within 1 frame
        frameQueue.clear();
        requestSyncFrameNow();

        // Control reader thread (Mac -> Phone live ISP commands)
        Thread controlThread = new Thread(() -> {
            try {
                BufferedReader reader = new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8));
                String line;
                while (connected.get() && (line = reader.readLine()) != null) {
                    processCommand(line.trim());
                }
            } catch (Throwable ignored) {
            } finally {
                connected.set(false);
            }
        }, "PxlCamControlThread");
        controlThread.setDaemon(true);
        controlThread.start();

        // High-priority sender loop (Phone -> Mac)
        try {
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(client.getOutputStream(), 512 * 1024));
            while (connected.get() && !client.isClosed()) {
                FrameData frame = frameQueue.poll(100, TimeUnit.MILLISECONDS);
                if (frame != null) {
                    int packedCamAndRot = (frame.camera & 0x03) | ((autoRotationQuadrants & 0x03) << 2);
                    out.writeInt(MAGIC);
                    out.writeByte(frame.codec);      // Byte 4: 2 = Hardware Video ISP H.264, 1 = JPEG
                    out.writeByte(packedCamAndRot);  // Byte 5: bits 0..1 = camera, bits 2..3 = autoRotationQuadrants
                    out.writeShort(frame.width);
                    out.writeShort(frame.height);
                    out.writeInt(frame.seq);
                    out.writeLong(frame.timestampNs);
                    out.writeInt(frame.payloadBytes.length);
                    out.write(frame.payloadBytes);
                    out.flush();
                }
            }
        } catch (Throwable ignored) {
        } finally {
            connected.set(false);
            try {
                client.close();
            } catch (Throwable ignored) {
            }
            System.out.println("[PxlCam] Mac receiver disconnected.");
        }
    }

    private static void processCommand(String cmd) {
        if (cmd.isEmpty()) return;
        System.out.println("[PxlCam] Command: " + cmd);
        try {
            if (cmd.startsWith("LENS:")) {
                String nextLens = cmd.substring(5).trim();
                if (!nextLens.equals(cameraId)) {
                    cameraId = nextLens;
                    if (!"0".equals(cameraId)) {
                        flashMode = 0;
                    }
                    restartCameraPipelineAsync();
                }
            } else if (cmd.startsWith("RES:")) {
                String[] parts = cmd.substring(4).trim().split("x");
                if (parts.length == 2) {
                    int w = Integer.parseInt(parts[0]);
                    int h = Integer.parseInt(parts[1]);
                    if (w != requestedWidth || h != requestedHeight) {
                        requestedWidth = w;
                        requestedHeight = h;
                        restartCameraPipelineAsync();
                    }
                }
            } else if (cmd.startsWith("ZOOM:")) {
                zoomFactor = Math.max(1.0f, Math.min(8.0f, Float.parseFloat(cmd.substring(5).trim())));
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("EV:")) {
                exposureComp = Math.max(-12, Math.min(12, Integer.parseInt(cmd.substring(3).trim())));
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("FLASH:")) {
                flashMode = Integer.parseInt(cmd.substring(6).trim()); // 0=Off, 1=On(Torch), 2=Auto
                if (flashMode != 0 && torchStrengthPercent == 0) {
                    torchStrengthPercent = 100;
                }
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("TORCH:")) {
                flashMode = "1".equals(cmd.substring(6).trim()) ? 1 : 0;
                if (flashMode != 0 && torchStrengthPercent == 0) {
                    torchStrengthPercent = 100;
                }
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("TORCH_STRENGTH:") || cmd.startsWith("FLASH_LEVEL:")) {
                String rawVal = cmd.contains(":") ? cmd.substring(cmd.indexOf(':') + 1).trim() : "100";
                int pct = Math.max(0, Math.min(100, Integer.parseInt(rawVal)));
                torchStrengthPercent = pct;
                if (pct == 0) {
                    flashMode = 0;
                } else if ("0".equals(cameraId)) {
                    flashMode = 1;
                }
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("AFMODE:")) {
                afMode = Integer.parseInt(cmd.substring(7).trim()); // 0=Continuous, 1=Macro, 2=Locked
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("AF:TRIGGER")) {
                cameraHandler.post(PxlCamServer::triggerAutofocusNow);
            } else if (cmd.startsWith("AF:")) {
                afMode = "1".equals(cmd.substring(3).trim()) ? 2 : 0;
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("WB:")) {
                wbMode = Integer.parseInt(cmd.substring(3).trim());
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("FPSMODE:")) {
                fpsMode = Integer.parseInt(cmd.substring(8).trim());
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("AELOCK:")) {
                aeLocked = "1".equals(cmd.substring(7).trim());
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("AWBLOCK:")) {
                awbLocked = "1".equals(cmd.substring(8).trim());
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("QUALITY:")) {
                int q = Math.max(50, Math.min(98, Integer.parseInt(cmd.substring(8).trim())));
                jpegQuality = (byte) q;
                cameraHandler.post(PxlCamServer::applyLiveRepeatingRequest);
            } else if (cmd.startsWith("SCREEN:")) {
                screenOff = "0".equals(cmd.substring(7).trim());
                setPhysicalScreenPower(!screenOff);
            }
        } catch (Throwable t) {
            t.printStackTrace(System.err);
        }
    }

    private static void setPhysicalScreenPower(boolean powerOn) {
        try {
            Class<?> scClass = Class.forName("com.genymobile.scrcpy.wrappers.SurfaceControl");
            int mode = powerOn ? 2 : 0; // POWER_MODE_NORMAL = 2, POWER_MODE_OFF = 0
            Method hasIdsMethod = scClass.getDeclaredMethod("hasGetPhysicalDisplayIdsMethod");
            boolean hasIds = (Boolean) hasIdsMethod.invoke(null);
            Method setPowerMethod = scClass.getDeclaredMethod("setDisplayPowerMode", IBinder.class, int.class);
            if (hasIds) {
                Method getIdsMethod = scClass.getDeclaredMethod("getPhysicalDisplayIds");
                Method getTokenMethod = scClass.getDeclaredMethod("getPhysicalDisplayToken", long.class);
                long[] ids = (long[]) getIdsMethod.invoke(null);
                if (ids != null) {
                    for (long id : ids) {
                        IBinder token = (IBinder) getTokenMethod.invoke(null, id);
                        if (token != null) {
                            setPowerMethod.invoke(null, token, mode);
                        }
                    }
                }
            } else {
                Method getBuiltInMethod = scClass.getDeclaredMethod("getBuiltInDisplay");
                IBinder token = (IBinder) getBuiltInMethod.invoke(null);
                if (token != null) {
                    setPowerMethod.invoke(null, token, mode);
                }
            }
            System.out.println("[PxlCam] Physical display power set to: " + (powerOn ? "ON" : "OFF"));
        } catch (Throwable t) {
            System.err.println("[PxlCam] Could not toggle display power: " + t.getMessage());
        }
    }

    private static void recomputeAutoRotationQuadrants() {
        int rotDeg;
        if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            rotDeg = (sensorOrientation - deviceOrientationDegrees + 360) % 360;
        } else {
            rotDeg = (sensorOrientation + deviceOrientationDegrees) % 360;
        }
        int nextQuad = (rotDeg / 90) & 0x03;
        if (nextQuad != autoRotationQuadrants) {
            autoRotationQuadrants = nextQuad;
            System.out.println("[PxlCam] Orientation changed -> deviceDeg=" + deviceOrientationDegrees
                    + " rotQuadrants=" + nextQuad + (nextQuad % 2 == 1 ? " (PORTRAIT 9:16)" : " (LANDSCAPE 16:9)"));
        }
    }

    private static void startOrientationSensorListener(Class<?> workaroundsClass) {
        try {
            Method getSysCtx = workaroundsClass.getDeclaredMethod("getSystemContext");
            getSysCtx.setAccessible(true);
            Context sysCtx = (Context) getSysCtx.invoke(null);
            if (sysCtx == null) return;
            SensorManager sm = (SensorManager) sysCtx.getSystemService(Context.SENSOR_SERVICE);
            if (sm == null) return;
            Sensor accel = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            if (accel == null) return;

            SensorEventListener listener = new SensorEventListener() {
                @Override
                public void onSensorChanged(SensorEvent event) {
                    if (event == null || event.values == null || event.values.length < 2) return;
                    float ax = event.values[0];
                    float ay = event.values[1];
                    // Ignore when phone is flat on a desk (gravity mostly on Z axis)
                    if (ax * ax + ay * ay < 3.0f) return;

                    int nextDeg = deviceOrientationDegrees;
                    // 1.35x hysteresis prevents oscillation near 45-degree diagonals
                    if (Math.abs(ay) > Math.abs(ax) * 1.35f) {
                        nextDeg = (ay > 0) ? 0 : 180;
                    } else if (Math.abs(ax) > Math.abs(ay) * 1.35f) {
                        nextDeg = (ax > 0) ? 270 : 90;
                    }
                    if (nextDeg != deviceOrientationDegrees) {
                        deviceOrientationDegrees = nextDeg;
                        recomputeAutoRotationQuadrants();
                    }
                }

                @Override
                public void onAccuracyChanged(Sensor sensor, int accuracy) {}
            };

            boolean ok = sm.registerListener(listener, accel, SensorManager.SENSOR_DELAY_UI, cameraHandler);
            System.out.println("[PxlCam] Hardware Accelerometer Auto-Rotation listener active: " + ok);
        } catch (Throwable t) {
            System.err.println("[PxlCam] Accelerometer auto-rotate unavailable: " + t.getMessage());
        }
    }

    private static void startWatchdogThread() {
        Thread watchdog = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(500);
                    long elapsed = SystemClock.elapsedRealtime() - lastFrameTimeMs.get();
                    if (elapsed > 2500 && !reconfiguring.get()) {
                        System.err.println("[PxlCam] Watchdog detected ISP stall (" + elapsed + "ms) — auto-healing Camera2 session...");
                        restartCameraPipelineAsync();
                    }
                } catch (Throwable ignored) {
                }
            }
        }, "PxlCamWatchdog");
        watchdog.setDaemon(true);
        watchdog.start();
    }

    private static void restartCameraPipelineAsync() {
        if (!reconfiguring.compareAndSet(false, true)) return;
        Thread reconfigThread = new Thread(() -> {
            try {
                closeCameraResources();
                openAndConfigureCamera();
            } catch (Throwable t) {
                t.printStackTrace(System.err);
            } finally {
                lastFrameTimeMs.set(SystemClock.elapsedRealtime());
                reconfiguring.set(false);
            }
        }, "PxlCamReconfigThread");
        reconfigThread.setDaemon(true);
        reconfigThread.start();
    }

    private static void startCameraPipeline() throws Exception {
        openAndConfigureCamera();
    }

    private static synchronized void closeCameraResources() {
        try {
            if (currentSession != null) {
                currentSession.close();
                currentSession = null;
            }
        } catch (Throwable ignored) {}
        try {
            if (currentDevice != null) {
                currentDevice.close();
                currentDevice = null;
            }
        } catch (Throwable ignored) {}
        try {
            if (currentEncoder != null) {
                try {
                    currentEncoder.stop();
                } catch (Throwable ignored) {}
                currentEncoder.release();
                currentEncoder = null;
            }
        } catch (Throwable ignored) {}
        try {
            if (encoderInputSurface != null) {
                encoderInputSurface.release();
                encoderInputSurface = null;
            }
        } catch (Throwable ignored) {}
        try {
            if (currentReader != null) {
                currentReader.close();
                currentReader = null;
            }
        } catch (Throwable ignored) {}
        codecConfigBytes = null;
        frameQueue.clear();
    }

    private static void requestSyncFrameNow() {
        MediaCodec enc = currentEncoder;
        if (enc != null) {
            try {
                Bundle params = new Bundle();
                params.putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0);
                enc.setParameters(params);
            } catch (Throwable ignored) {}
        }
    }

    private static Size chooseOptimalVideoSize(Size[] videoSizes, int reqW, int reqH) {
        // Cap at 1920x1080 so the 13MP CMOS sensor stays in 2x2 Hardware Analog Pixel Binning mode
        // and routes through MediaTek's dedicated Hardware Video ISP (MDP + 3DNR) just like scrcpy.
        int targetW = Math.min(reqW, 1920);
        int targetH = Math.min(reqH, 1080);
        if (videoSizes == null || videoSizes.length == 0) {
            return new Size(targetW, targetH);
        }

        Size bestMatch = null;
        long bestScore = Long.MAX_VALUE;
        for (Size s : videoSizes) {
            if (s.getWidth() > 1920 || s.getHeight() > 1088) continue;
            double aspect = (double) s.getWidth() / s.getHeight();
            boolean is16by9 = Math.abs(aspect - (16.0 / 9.0)) < 0.06;
            if (!is16by9) continue;

            if (s.getWidth() == targetW && s.getHeight() == targetH) {
                return s;
            }
            long diff = Math.abs((long) s.getWidth() - targetW) + Math.abs((long) s.getHeight() - targetH);
            if (diff < bestScore) {
                bestScore = diff;
                bestMatch = s;
            }
        }
        return bestMatch != null ? bestMatch : new Size(1920, 1080);
    }

    private static void startEncoderDrainThread(final MediaCodec encoder, final int encW, final int encH, final String camIdStr) {
        encoderRunning.set(true);
        Thread drainThread = new Thread(() -> {
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            while (encoderRunning.get() && currentEncoder == encoder) {
                try {
                    int outIndex = encoder.dequeueOutputBuffer(info, 25000);
                    if (outIndex >= 0) {
                        ByteBuffer outBuf = encoder.getOutputBuffer(outIndex);
                        if (outBuf != null && info.size > 0) {
                            outBuf.position(info.offset);
                            outBuf.limit(info.offset + info.size);
                            byte[] rawBytes = new byte[info.size];
                            outBuf.get(rawBytes);

                            if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                                codecConfigBytes = rawBytes;
                            } else {
                                byte[] cfg = codecConfigBytes;
                                byte[] packetBytes;
                                if (cfg != null && cfg.length > 0) {
                                    packetBytes = new byte[cfg.length + rawBytes.length];
                                    System.arraycopy(cfg, 0, packetBytes, 0, cfg.length);
                                    System.arraycopy(rawBytes, 0, packetBytes, cfg.length, rawBytes.length);
                                } else {
                                    packetBytes = rawBytes;
                                }

                                int camInt = "1".equals(camIdStr) ? 1 : 0;
                                int seq = frameSequence.incrementAndGet();
                                long tsNs = info.presentationTimeUs * 1000L;
                                lastFrameTimeMs.set(SystemClock.elapsedRealtime());

                                FrameData data = new FrameData(packetBytes, encW, encH, camInt, seq, tsNs, 2);
                                if (frameQueue.size() >= 15) {
                                    frameQueue.clear();
                                    requestSyncFrameNow();
                                }
                                frameQueue.offer(data);
                            }
                        }
                        encoder.releaseOutputBuffer(outIndex, false);
                    }
                } catch (IllegalStateException stoppedEx) {
                    break;
                } catch (Throwable t) {
                    try {
                        Thread.sleep(5);
                    } catch (InterruptedException ignored) {}
                }
            }
        }, "PxlCamH264Drain");
        drainThread.setDaemon(true);
        drainThread.start();
    }

    private static synchronized void openAndConfigureCamera() throws Exception {
        CameraCharacteristics chars = cameraManager.getCameraCharacteristics(cameraId);
        activeArraySize = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        Integer so = chars.get(CameraCharacteristics.SENSOR_ORIENTATION);
        if (so != null) {
            sensorOrientation = so;
        }
        Integer lf = chars.get(CameraCharacteristics.LENS_FACING);
        if (lf != null) {
            lensFacing = lf;
        }
        recomputeAutoRotationQuadrants();

        StreamConfigurationMap map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] videoSizes = map != null ? map.getOutputSizes(MediaCodec.class) : null;
        Size chosen = chooseOptimalVideoSize(videoSizes, requestedWidth, requestedHeight);

        activeWidth = chosen.getWidth();
        activeHeight = chosen.getHeight();
        System.out.println("[PxlCam] Opening Camera " + cameraId + " via Hardware Video ISP (MediaCodec Surface) at " + activeWidth + "x" + activeHeight);

        // Configure zero-copy Hardware H.264 Encoder with COLOR_FormatSurface (exact same path as scrcpy)
        MediaFormat format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, activeWidth, activeHeight);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        format.setInteger(MediaFormat.KEY_BIT_RATE, 18_000_000); // 18 Mbps studio quality
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1); // Keyframe every 1s for instant lock
        try {
            format.setInteger("max-bframes", 0); // Zero B-frame latency
        } catch (Throwable ignored) {}

        currentEncoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
        currentEncoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        encoderInputSurface = currentEncoder.createInputSurface();
        currentEncoder.start();
        startEncoderDrainThread(currentEncoder, activeWidth, activeHeight, cameraId);

        CountDownLatch openLatch = new CountDownLatch(1);
        cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
            @Override
            public void onOpened(CameraDevice camera) {
                currentDevice = camera;
                openLatch.countDown();
            }

            @Override
            public void onDisconnected(CameraDevice camera) {
                openLatch.countDown();
                restartCameraPipelineAsync();
            }

            @Override
            public void onError(CameraDevice camera, int error) {
                System.err.println("[PxlCam] CameraDevice error: " + error);
                openLatch.countDown();
                restartCameraPipelineAsync();
            }
        }, cameraHandler);

        openLatch.await(4, TimeUnit.SECONDS);
        if (currentDevice == null) {
            throw new RuntimeException("Failed to open Camera " + cameraId);
        }

        Surface targetSurface = encoderInputSurface;
        try {
            // TEMPLATE_RECORD + MediaCodec InputSurface engages MediaTek's dedicated Hardware Video ISP (MDP + 3DNR)
            currentRequestBuilder = currentDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
        } catch (Throwable t) {
            currentRequestBuilder = currentDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        }
        currentRequestBuilder.addTarget(targetSurface);

        currentRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        currentRequestBuilder.set(CaptureRequest.CONTROL_AE_ANTIBANDING_MODE, CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_AUTO);

        // IMPORTANT: Keep TEMPLATE_RECORD's native hardware ISP defaults (FAST tonemap, FAST 3DNR, FAST edge)
        // identical to scrcpy! Overriding TONEMAP_MODE to HIGH_QUALITY on MediaTek HAL resets the hardware
        // sRGB/ADRC 64-point gamma LUT to a 2-point linear identity line [0,0 -> 1,1], making video 4x darker
        // and disabling hardware temporal 3DNR.
        availableFpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        aeCompRange = chars.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE);
        initTorchStrengthCapabilities(chars);

        CountDownLatch sessionLatch = new CountDownLatch(1);
        currentDevice.createCaptureSession(Collections.singletonList(targetSurface), new CameraCaptureSession.StateCallback() {
            @Override
            public void onConfigured(CameraCaptureSession session) {
                currentSession = session;
                applyLiveRepeatingRequest();
                sessionLatch.countDown();
            }

            @Override
            public void onConfigureFailed(CameraCaptureSession session) {
                System.err.println("[PxlCam] CaptureSession configuration failed");
                sessionLatch.countDown();
            }
        }, cameraHandler);
        sessionLatch.await(3, TimeUnit.SECONDS);
    }

    @SuppressWarnings("unchecked")
    private static void initTorchStrengthCapabilities(CameraCharacteristics chars) {
        maxHardwareTorchStrength = 1;
        try {
            java.lang.reflect.Field f = CameraCharacteristics.class.getDeclaredField("FLASH_INFO_STRENGTH_MAXIMUM_LEVEL");
            Object keyObj = f.get(null);
            if (keyObj instanceof CameraCharacteristics.Key) {
                Integer maxLvl = (Integer) chars.get((CameraCharacteristics.Key<Integer>) keyObj);
                if (maxLvl != null && maxLvl > 1) {
                    maxHardwareTorchStrength = maxLvl;
                }
            }
        } catch (Throwable ignored) {}

        try {
            java.lang.reflect.Constructor<CaptureRequest.Key> ctor =
                CaptureRequest.Key.class.getDeclaredConstructor(String.class, Class.class);
            ctor.setAccessible(true);
            if (keyFlashFiringPower == null) {
                keyFlashFiringPower = (CaptureRequest.Key<Byte>) ctor.newInstance("android.flash.firingPower", Byte.class);
            }
            if (keyMtkTorchDuty == null) {
                keyMtkTorchDuty = (CaptureRequest.Key<Byte>) ctor.newInstance("com.mediatek.flashfeature.customizedTorchDuty", Byte.class);
            }
            if (keyAndroid15FlashStrength == null) {
                keyAndroid15FlashStrength = (CaptureRequest.Key<Integer>) ctor.newInstance("android.flash.strengthLevel", Integer.class);
            }
        } catch (Throwable ignored) {}
    }

    private static boolean containsMode(int[] modes, int target) {
        if (modes == null) return false;
        for (int m : modes) {
            if (m == target) return true;
        }
        return false;
    }

    private static Range<Integer> selectTargetFpsRange() {
        if (availableFpsRanges == null || availableFpsRanges.length == 0) return null;
        Range<Integer> fallback = availableFpsRanges[availableFpsRanges.length - 1];
        if (fpsMode == 1) {
            // Locked 30 fps (Studio lighting)
            for (Range<Integer> r : availableFpsRanges) {
                if (r.getLower() == 30 && r.getUpper() == 30) return r;
            }
            return fallback;
        } else if (fpsMode == 2) {
            // Max Night Boost: On MediaTek HALs with TEMPLATE_RECORD, any range with upper==30
            // pins shutter integration at 33.3ms (1/30s) and forces noisy ISO 3200.
            // Prefer [15, 20] or [10, 15] so the sensor integrates for 50ms-66.7ms (2x more photons, half the ISO grain).
            Range<Integer> bestLowUpper = null;
            for (Range<Integer> r : availableFpsRanges) {
                if (r.getUpper() >= 15 && r.getUpper() <= 20 && r.getLower() < r.getUpper()) {
                    if (bestLowUpper == null || r.getUpper() > bestLowUpper.getUpper()) {
                        bestLowUpper = r; // Prefer [15, 20], else [10, 15]
                    }
                }
            }
            if (bestLowUpper != null) return bestLowUpper;
            Range<Integer> bestNight = fallback;
            for (Range<Integer> r : availableFpsRanges) {
                if (r.getUpper() >= 24 && r.getLower() < bestNight.getLower()) {
                    bestNight = r;
                }
            }
            return bestNight;
        } else {
            // Default (fpsMode == 0): Low-Light Auto
            // On MediaTek HALs with TEMPLATE_RECORD, any range with upper == 30 (including [5, 30])
            // pins hardware shutter integration at 33.3ms (1/30s) and forces max ISO 3200 in indoor/dim rooms.
            // Prefer [15, 20] or [10, 20] by default so hardware AE has 50ms-66.7ms integration out of the box.
            Range<Integer> bestLowUpper = null;
            for (Range<Integer> r : availableFpsRanges) {
                if (r.getUpper() == 20 && r.getLower() < r.getUpper()) {
                    if (bestLowUpper == null || r.getLower() > bestLowUpper.getLower()) {
                        bestLowUpper = r; // Prefer [15, 20], else [10, 20]
                    }
                }
            }
            if (bestLowUpper != null) return bestLowUpper;
            Range<Integer> bestAuto = null;
            for (Range<Integer> r : availableFpsRanges) {
                if (r.getUpper() == 30 && r.getLower() <= 15) {
                    if (bestAuto == null || r.getLower() > bestAuto.getLower()) {
                        bestAuto = r;
                    }
                }
            }
            return bestAuto != null ? bestAuto : fallback;
        }
    }

    private static void triggerAutofocusNow() {
        if (currentSession == null || currentRequestBuilder == null) return;
        try {
            currentRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO);
            currentRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START);
            currentSession.capture(currentRequestBuilder.build(), null, cameraHandler);
            currentRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE);
            applyLiveRepeatingRequest();
        } catch (Throwable t) {
            System.err.println("[PxlCam] AF trigger error: " + t.getMessage());
        }
    }

    private static void applyLiveRepeatingRequest() {
        if (currentSession == null || currentRequestBuilder == null) return;
        try {
            currentRequestBuilder.set(CaptureRequest.JPEG_QUALITY, jpegQuality);

            boolean torchActive = "0".equals(cameraId) && flashMode == 1 && torchStrengthPercent > 0;
            // On fixed-GPIO LED drivers (maxHardwareTorchStrength <= 1, e.g., Moto G Pure OCP8132),
            // apply proportional sensor AE attenuation when Torch is dimmed below 100% so the
            // flash-illuminated scene smoothly dims without blowing out highlights.
            int flashDimEvOffset = 0;
            if (torchActive && maxHardwareTorchStrength <= 1 && torchStrengthPercent < 100) {
                flashDimEvOffset = Math.round(((torchStrengthPercent - 100) / 100.0f) * 6.0f);
            }

            int clampedEv = exposureComp + flashDimEvOffset;
            if (aeCompRange != null) {
                clampedEv = Math.max(aeCompRange.getLower(), Math.min(aeCompRange.getUpper(), clampedEv));
            }
            currentRequestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, clampedEv);
            currentRequestBuilder.set(CaptureRequest.CONTROL_AE_LOCK, aeLocked);
            currentRequestBuilder.set(CaptureRequest.CONTROL_AWB_LOCK, awbLocked);

            Range<Integer> targetRange = selectTargetFpsRange();
            if (targetRange != null) {
                currentRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, targetRange);
            }

            // Flash mode: 0 = Off, 1 = On (Torch), 2 = Auto (never use ON_AUTO_FLASH in continuous video
            // because MediaTek HAL gets stuck in aeState=FLASH_REQUIRED without turning on the LED)
            if (torchActive) {
                currentRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON);
                currentRequestBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);
                byte power10 = (byte) Math.max(1, Math.min(10, Math.round(torchStrengthPercent / 10.0f)));
                if (keyFlashFiringPower != null) {
                    try { currentRequestBuilder.set(keyFlashFiringPower, power10); } catch (Throwable ignored) {}
                }
                if (keyMtkTorchDuty != null) {
                    try { currentRequestBuilder.set(keyMtkTorchDuty, power10); } catch (Throwable ignored) {}
                }
                if (maxHardwareTorchStrength > 1) {
                    int hwLevel = Math.max(1, Math.min(maxHardwareTorchStrength,
                        Math.round((torchStrengthPercent / 100.0f) * maxHardwareTorchStrength)));
                    if (keyAndroid15FlashStrength != null) {
                        try { currentRequestBuilder.set(keyAndroid15FlashStrength, hwLevel); } catch (Throwable ignored) {}
                    }
                    try {
                        Method m = CameraManager.class.getMethod("turnOnTorchWithStrengthLevel", String.class, int.class);
                        m.invoke(cameraManager, cameraId, hwLevel);
                    } catch (Throwable ignored) {}
                }
            } else {
                currentRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON);
                currentRequestBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF);
            }

            // White Balance mode
            int awbConst;
            switch (wbMode) {
                case 1: awbConst = CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT; break;
                case 2: awbConst = CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT; break;
                case 3: awbConst = CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT; break;
                case 4: awbConst = CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT; break;
                case 5: awbConst = CameraMetadata.CONTROL_AWB_MODE_SHADE; break;
                default: awbConst = CameraMetadata.CONTROL_AWB_MODE_AUTO; break;
            }
            currentRequestBuilder.set(CaptureRequest.CONTROL_AWB_MODE, awbConst);

            // Autofocus mode: 0 = Continuous Video, 1 = Macro, 2 = Locked
            int afConst;
            switch (afMode) {
                case 1: afConst = CameraMetadata.CONTROL_AF_MODE_MACRO; break;
                case 2: afConst = CameraMetadata.CONTROL_AF_MODE_OFF; break;
                default: afConst = CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO; break;
            }
            currentRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, afConst);

            if (activeArraySize != null) {
                if (zoomFactor > 1.01f) {
                    int centerX = activeArraySize.width() / 2;
                    int centerY = activeArraySize.height() / 2;
                    int deltaX = (int) (0.5f * activeArraySize.width() / zoomFactor);
                    int deltaY = (int) (0.5f * activeArraySize.height() / zoomFactor);
                    Rect crop = new Rect(centerX - deltaX, centerY - deltaY, centerX + deltaX, centerY + deltaY);
                    currentRequestBuilder.set(CaptureRequest.SCALER_CROP_REGION, crop);
                } else {
                    currentRequestBuilder.set(CaptureRequest.SCALER_CROP_REGION, activeArraySize);
                }
            }

            currentSession.setRepeatingRequest(currentRequestBuilder.build(), new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureFailed(CameraCaptureSession session, CaptureRequest request, CaptureFailure failure) {
                    System.err.println("[PxlCam] Capture failure reason=" + failure.getReason() + " — continuing ring buffer");
                }
            }, cameraHandler);
        } catch (CameraAccessException e) {
            System.err.println("[PxlCam] Failed to apply repeating request: " + e.getMessage());
        }
    }
}
