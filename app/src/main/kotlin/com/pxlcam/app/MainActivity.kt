package com.pxlcam.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Pxl Cam Studio Activity (com.pxlcam.app)
 *
 * Play Store–compliant studio camera & hardware ISP broadcaster featuring:
 *  - Cyber Kitsune (#FF8400) dark studio theme
 *  - Live 1080p Camera2 hardware ISP viewfinder (with 64-pt sRGB + ADRC tone curve)
 *  - Real-time bidirectional sync with macOS Pxl Cam.app & OBS Custom Dock (:9876)
 *  - Hardware Lens, Flash, Exposure Compensation (EV), Zoom, White Balance, and Focus controls
 *  - OLED Power Saver dimmer overlay for zero-heat multi-hour OBS sessions
 *  - In-app Privacy & Data Safety + Open Source Licenses disclosure
 */
class MainActivity : AppCompatActivity(), PxlCamEngine.Listener {

    private var streamingService: PxlCamStreamingService? = null
    private var serviceBound = false

    private lateinit var surfaceView: SurfaceView
    private lateinit var topLeftOverlayBrand: ImageView
    private lateinit var statusBadge: TextView
    private lateinit var telemetryText: TextView
    private lateinit var btnBackLens: Button
    private lateinit var btnFrontLens: Button
    private lateinit var btnFlash: Button
    private lateinit var flashSeekBar: SeekBar
    private lateinit var flashValueText: TextView
    private lateinit var evSeekBar: SeekBar
    private lateinit var evValueText: TextView
    private lateinit var zoomSeekBar: SeekBar
    private lateinit var zoomValueText: TextView
    private val wbButtons = mutableListOf<Button>()
    private val afButtons = mutableListOf<Button>()
    private lateinit var oledOverlay: FrameLayout

    private val controlsHideHandler = Handler(Looper.getMainLooper())
    private val controlsHideRunnable = Runnable {
        if (::telemetryText.isInitialized) {
            telemetryText.animate().alpha(0f).setDuration(300).start()
        }
        if (::topLeftOverlayBrand.isInitialized) {
            topLeftOverlayBrand.animate().alpha(0f).setDuration(300).start()
        }
    }

    private fun showControlsAndResetTimer() {
        if (::telemetryText.isInitialized && telemetryText.alpha < 0.99f) {
            telemetryText.animate().alpha(1f).setDuration(300).start()
        }
        if (::topLeftOverlayBrand.isInitialized && topLeftOverlayBrand.alpha < 0.99f) {
            topLeftOverlayBrand.animate().alpha(1f).setDuration(300).start()
        }
        controlsHideHandler.removeCallbacks(controlsHideRunnable)
        controlsHideHandler.postDelayed(controlsHideRunnable, 3000L)
    }

    override fun dispatchTouchEvent(ev: MotionEvent?): Boolean {
        showControlsAndResetTimer()
        return super.dispatchTouchEvent(ev)
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        val camGranted = result[Manifest.permission.CAMERA] == true ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (camGranted) {
            startAndBindStreamingService()
        } else {
            showPermissionRationaleDialog()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? PxlCamStreamingService.LocalBinder ?: return
            streamingService = localBinder.getService()
            serviceBound = true
            streamingService?.engine?.setListener(this@MainActivity)
            if (surfaceView.holder.surface.isValid) {
                streamingService?.engine?.setPreviewSurface(surfaceView.holder.surface)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            streamingService?.engine?.setListener(null)
            streamingService = null
            serviceBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(buildStudioLayout())
        showControlsAndResetTimer()
        ensurePermissionsAndStart()
    }

    override fun onDestroy() {
        controlsHideHandler.removeCallbacks(controlsHideRunnable)
        if (serviceBound) {
            streamingService?.engine?.setListener(null)
            unbindService(serviceConnection)
            serviceBound = false
        }
        super.onDestroy()
    }

    private fun ensurePermissionsAndStart() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.CAMERA)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (needed.isEmpty()) {
            startAndBindStreamingService()
        } else {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun startAndBindStreamingService() {
        val intent = Intent(this, PxlCamStreamingService::class.java).apply {
            action = PxlCamStreamingService.ACTION_START
        }
        ContextCompat.startForegroundService(this, intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun buildStudioLayout(): View {
        val rootFrame = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#08090C"))
        }

        val scroll = ScrollView(this).apply {
            isFillViewport = true
        }

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(20))
        }

        // 1. Top Studio Header
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = roundedBox("#12141C", "#44FF8400", dp(12))
            setPadding(dp(12), dp(10), dp(12), dp(10))
        }

        val logoView = ImageView(this).apply {
            setImageResource(R.drawable.pxl_cam_logo)
            layoutParams = LinearLayout.LayoutParams(dp(40), dp(40)).apply {
                rightMargin = dp(6)
            }
        }
        header.addView(logoView)

        val wordmarkHeaderView = ImageView(this).apply {
            setImageResource(R.drawable.pxl_cam_wordmark_bttf)
            scaleType = ImageView.ScaleType.FIT_START
            adjustViewBounds = true
            layoutParams = LinearLayout.LayoutParams(dp(56), dp(44)).apply {
                rightMargin = dp(8)
            }
        }
        header.addView(wordmarkHeaderView)

        val titleCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        statusBadge = TextView(this).apply {
            text = "READY • TCP :9876 • 64-PT sRGB ISP"
            setTextColor(Color.parseColor("#19E673"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10.5f)
            typeface = Typeface.MONOSPACE
        }
        titleCol.addView(statusBadge)
        header.addView(titleCol)

        val btnOled = makePillButton("🌙 OLED", false) {
            setOledSaverMode(true)
        }.apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(34)).apply {
                rightMargin = dp(6)
            }
        }
        val btnPrivacy = makePillButton("ⓘ Info", false) {
            showPrivacyAndLicensesDialog()
        }.apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(34))
        }
        header.addView(btnOled)
        header.addView(btnPrivacy)
        container.addView(header)

        // 2. Live 16:9 Viewfinder Card
        val previewFrame = FrameLayout(this).apply {
            background = roundedBox("#000000", "#66FF8400", dp(12))
            setPadding(dp(2), dp(2), dp(2), dp(2))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(220)
            ).apply {
                topMargin = dp(12)
                bottomMargin = dp(12)
            }
        }

        surfaceView = SurfaceView(this)
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                streamingService?.engine?.setPreviewSurface(holder.surface)
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                streamingService?.engine?.setPreviewSurface(null)
            }
        })
        previewFrame.addView(surfaceView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        // Top-Left Coral Fox + 80's Glam BTTF PXL/CAM Camera Wordmark Overlay
        topLeftOverlayBrand = ImageView(this).apply {
            setImageResource(R.drawable.pxl_cam_top_left_overlay)
            scaleType = ImageView.ScaleType.FIT_XY
            background = roundedBox("#8805070B", "#44FF8400", dp(8))
            setPadding(0, 0, 0, 0)
            layoutParams = FrameLayout.LayoutParams(
                dp(95),
                dp(62),
                Gravity.TOP or Gravity.START
            ).apply {
                leftMargin = dp(10)
                topMargin = dp(10)
            }
        }
        previewFrame.addView(topLeftOverlayBrand)

        telemetryText = TextView(this).apply {
            text = "1920x1080 • 30.0 FPS • H.264 HW 12 Mbps"
            setTextColor(Color.parseColor("#FF8400"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10.5f)
            typeface = Typeface.MONOSPACE
            background = roundedBox("#CC08090C", "#55FF8400", dp(6))
            setPadding(dp(8), dp(4), dp(8), dp(4))
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.START
            ).apply {
                leftMargin = dp(10)
                bottomMargin = dp(10)
            }
        }
        previewFrame.addView(telemetryText)
        container.addView(previewFrame)

        // 3. Hardware Lens & Flash Card
        val lensCard = makeSectionCard("HARDWARE LENS & TORCH")
        val lensRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        btnBackLens = makePillButton("Back Lens", true) {
            streamingService?.engine?.setLens(0)
        }.apply {
            layoutParams = LinearLayout.LayoutParams(0, dp(38), 1f).apply { rightMargin = dp(6) }
        }
        btnFrontLens = makePillButton("Front Lens", false) {
            streamingService?.engine?.setLens(1)
        }.apply {
            layoutParams = LinearLayout.LayoutParams(0, dp(38), 1f).apply { rightMargin = dp(6) }
        }
        btnFlash = makePillButton("⚡ Flash OFF", false) {
            val eng = streamingService?.engine ?: return@makePillButton
            eng.setFlash(!eng.isFlashOn)
        }.apply {
            layoutParams = LinearLayout.LayoutParams(0, dp(38), 1f)
        }
        lensRow.addView(btnBackLens)
        lensRow.addView(btnFrontLens)
        lensRow.addView(btnFlash)
        lensCard.addView(lensRow)

        val flashDimRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(8), 0, 0)
        }
        flashDimRow.addView(makeLabel("⚡ Dimmer", 68))
        flashSeekBar = SeekBar(this).apply {
            max = 100
            progress = 0
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    flashValueText.text = if (p == 0) "OFF" else "$p%"
                    if (fromUser) streamingService?.engine?.setTorchStrength(p)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        flashValueText = makeValueLabel("OFF")
        flashDimRow.addView(flashSeekBar)
        flashDimRow.addView(flashValueText)
        lensCard.addView(flashDimRow)
        container.addView(lensCard)

        // 4. Exposure & Zoom Sliders Card
        val sensorCard = makeSectionCard("HARDWARE EXPOSURE & OPTICS")
        val evRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        evRow.addView(makeLabel("HW EV", 68))
        evSeekBar = SeekBar(this).apply {
            max = 16
            progress = 8
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    val ev = p - 8
                    evValueText.text = if (ev > 0) "+$ev" else "$ev"
                    if (fromUser) streamingService?.engine?.setExposureCompensation(ev)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        evValueText = makeValueLabel("0")
        evRow.addView(evSeekBar)
        evRow.addView(evValueText)
        sensorCard.addView(evRow)

        val zoomRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(8), 0, 0)
        }
        zoomRow.addView(makeLabel("HW Zoom", 68))
        zoomSeekBar = SeekBar(this).apply {
            max = 30
            progress = 0
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    val z = 1.0f + (p / 10.0f)
                    zoomValueText.text = String.format("%.1fx", z)
                    if (fromUser) streamingService?.engine?.setZoom(z)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        zoomValueText = makeValueLabel("1.0x")
        zoomRow.addView(zoomSeekBar)
        zoomRow.addView(zoomValueText)
        sensorCard.addView(zoomRow)
        container.addView(sensorCard)

        // 5. White Balance & Focus Card
        val wbCard = makeSectionCard("WHITE BALANCE & AUTOFOCUS")
        val wbRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val wbNames = listOf("AWB", "Day", "Cloud", "Tung", "Fluo")
        wbButtons.clear()
        for (i in wbNames.indices) {
            val b = makePillButton(wbNames[i], i == 0) {
                streamingService?.engine?.setWhiteBalance(i)
            }.apply {
                layoutParams = LinearLayout.LayoutParams(0, dp(34), 1f).apply {
                    if (i < wbNames.lastIndex) rightMargin = dp(4)
                }
            }
            wbButtons.add(b)
            wbRow.addView(b)
        }
        wbCard.addView(wbRow)

        val afRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(8), 0, 0)
        }
        val afNames = listOf("AF-C Video", "AF-S Photo", "Lock AF", "∞ Infinity")
        afButtons.clear()
        for (i in afNames.indices) {
            val b = makePillButton(afNames[i], i == 0) {
                streamingService?.engine?.setFocusMode(i)
            }.apply {
                layoutParams = LinearLayout.LayoutParams(0, dp(34), 1f).apply {
                    if (i < afNames.lastIndex) rightMargin = dp(4)
                }
            }
            afButtons.add(b)
            afRow.addView(b)
        }
        wbCard.addView(afRow)
        container.addView(wbCard)

        scroll.addView(container)
        rootFrame.addView(scroll)

        // 6. Full-screen OLED Saver Overlay (Pure #000000 black, tap anywhere to wake)
        oledOverlay = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            visibility = View.GONE
            isClickable = true
            isFocusable = true
            setOnClickListener { setOledSaverMode(false) }
        }
        val oledHint = TextView(this).apply {
            text = "⚡ PXL CAM STREAMING 1080P\nOLED Power Saver Active • Tap Anywhere to Wake"
            setTextColor(Color.parseColor("#66FF8400"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            gravity = Gravity.CENTER
            typeface = Typeface.MONOSPACE
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER
            )
        }
        oledOverlay.addView(oledHint)
        rootFrame.addView(oledOverlay)

        return rootFrame
    }

    private fun setOledSaverMode(enabled: Boolean) {
        oledOverlay.visibility = if (enabled) View.VISIBLE else View.GONE
        val lp = window.attributes
        lp.screenBrightness = if (enabled) 0.01f else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        window.attributes = lp
    }

    override fun onEngineStateChanged(
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
    ) {
        runOnUiThread {
            statusBadge.text = if (connectedClients > 0) {
                "LIVE TO MAC/OBS ($connectedClients) • TCP :9876"
            } else {
                "READY • TCP :9876 • 64-PT sRGB ISP"
            }
            telemetryText.text = String.format(
                "1920x1080 • %.1f FPS • H.264 HW 12M • EV %s",
                if (fps > 0f) fps else 30.0f,
                if (evOffset >= 0) "+$evOffset" else "$evOffset"
            )
            stylePillButton(btnBackLens, cameraIndex == 0)
            stylePillButton(btnFrontLens, cameraIndex == 1)
            btnFlash.text = if (flashOn) "⚡ Flash ${torchStrengthPercent}%" else "⚡ Flash OFF"
            stylePillButton(btnFlash, flashOn)

            val effectivePct = if (flashOn) torchStrengthPercent.coerceIn(1, 100) else 0
            flashSeekBar.progress = effectivePct
            flashValueText.text = if (effectivePct == 0) "OFF" else "$effectivePct%"

            evSeekBar.progress = (evOffset + 8).coerceIn(0, 16)
            evValueText.text = if (evOffset > 0) "+$evOffset" else "$evOffset"

            val zProg = ((zoomRatio - 1.0f) * 10.0f).toInt().coerceIn(0, 30)
            zoomSeekBar.progress = zProg
            zoomValueText.text = String.format("%.1fx", zoomRatio)

            for (i in wbButtons.indices) stylePillButton(wbButtons[i], i == wbMode)
            for (i in afButtons.indices) stylePillButton(afButtons[i], i == afMode)
        }
    }

    private fun showPrivacyAndLicensesDialog() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.privacy_policy_title))
            .setMessage(
                getString(R.string.privacy_policy_body) +
                    "\n\nOpen Source Software:\n" +
                    "• AndroidX Core / AppCompat / Material Components (Apache License 2.0)\n" +
                    "• Kotlin Standard Library (Apache License 2.0)"
            )
            .setPositiveButton("Close", null)
            .show()
    }

    private fun showPermissionRationaleDialog() {
        AlertDialog.Builder(this)
            .setTitle("Camera Permission Required")
            .setMessage("Pxl Cam requires Camera access to stream hardware-encoded 1080p video to your Mac and OBS Studio.")
            .setPositiveButton("Open Settings") { _, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", packageName, null)
                }
                startActivity(intent)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun makeSectionCard(title: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = roundedBox("#12141C", "#24FFFFFF", dp(10))
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dp(10)
            }
            val header = TextView(this@MainActivity).apply {
                text = title
                setTextColor(Color.parseColor("#FF8400"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 10.5f)
                typeface = Typeface.create("sans-serif-black", Typeface.BOLD)
                setPadding(0, 0, 0, dp(8))
            }
            addView(header)
        }
    }

    private fun makePillButton(label: String, active: Boolean, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setPadding(dp(8), dp(2), dp(8), dp(2))
            stylePillButton(this, active)
            setOnClickListener { onClick() }
        }
    }

    private fun stylePillButton(btn: Button, active: Boolean) {
        if (active) {
            btn.setTextColor(Color.parseColor("#FF8400"))
            btn.typeface = Typeface.DEFAULT_BOLD
            btn.background = roundedBox("#2EFF8400", "#FF8400", dp(8))
        } else {
            btn.setTextColor(Color.parseColor("#F4F5F8"))
            btn.typeface = Typeface.DEFAULT
            btn.background = roundedBox("#1AFFFFFF", "#2EFFFFFF", dp(8))
        }
    }

    private fun makeLabel(text: String, widthDp: Int): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(Color.parseColor("#CFD4E0"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            layoutParams = LinearLayout.LayoutParams(dp(widthDp), LinearLayout.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun makeValueLabel(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(Color.parseColor("#FF8400"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            typeface = Typeface.MONOSPACE
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(dp(44), LinearLayout.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun roundedBox(fillHex: String, strokeHex: String, radiusPx: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = radiusPx.toFloat()
            setColor(Color.parseColor(fillHex))
            setStroke(dp(1), Color.parseColor(strokeHex))
        }
    }

    private fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()
}

