package com.pxlcam.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

/**
 * Google Play Store–compliant Foreground Service (type: camera|connectedDevice)
 * that keeps the 1080p 30fps Camera2 + MediaCodec H.264 hardware stream active
 * while connected to macOS Pxl Cam.app and OBS Studio.
 */
class PxlCamStreamingService : Service() {

    companion object {
        const val CHANNEL_ID = "pxl_cam_streaming_channel"
        const val NOTIFICATION_ID = 9876
        const val ACTION_START = "com.pxlcam.app.ACTION_START"
        const val ACTION_STOP = "com.pxlcam.app.ACTION_STOP"
    }

    inner class LocalBinder : Binder() {
        fun getService(): PxlCamStreamingService = this@PxlCamStreamingService
    }

    private val binder = LocalBinder()
    lateinit var engine: PxlCamEngine
        private set
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        engine = PxlCamEngine(applicationContext)
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PxlCam::StreamingWakeLock").apply {
            setReferenceCounted(false)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            engine.stop()
            if (wakeLock?.isHeld == true) wakeLock?.release()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
            )
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        if (wakeLock?.isHeld == false) {
            wakeLock?.acquire(12 * 60 * 60 * 1000L)
        }
        engine.start(1920, 1080)
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        engine.stop()
        if (wakeLock?.isHeld == true) wakeLock?.release()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_desc)
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        val openIntent = Intent(this, MainActivity::class.java)
        val pendingOpen = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, PxlCamStreamingService::class.java).apply {
            action = ACTION_STOP
        }
        val pendingStop = PendingIntent.getService(
            this,
            1,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setOngoing(true)
            .setContentIntent(pendingOpen)
            .addAction(android.R.drawable.ic_media_pause, "Stop Stream", pendingStop)
            .build()
    }
}

