package com.idr.navigation

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
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * IDR Foreground Navigation Service.
 *
 * Keeps the IDR Dead Reckoning engine and sensor streams alive during driving,
 * even when the application is backgrounded or the screen is turned off.
 */
class IdrNavigationService : Service() {

    companion object {
        private const val TAG = "IdrNavService"
        private const val NOTIFICATION_CHANNEL_ID = "idr_navigation_channel"
        private const val NOTIFICATION_ID = 1001

        const val ACTION_START = "com.idr.navigation.ACTION_START"
        const val ACTION_STOP = "com.idr.navigation.ACTION_STOP"
    }

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())
    private var pollJob: Job? = null

    private var wakeLock: PowerManager.WakeLock? = null
    lateinit var idrNative: IdrNative
        private set
    private var sensorManager: IdrSensorManager? = null

    private val _navigationState = MutableStateFlow(IdrNative.NavigationState())
    val navigationState: StateFlow<IdrNative.NavigationState> = _navigationState.asStateFlow()

    private val _diagnostics = MutableStateFlow(IdrNative.Diagnostics())
    val diagnostics: StateFlow<IdrNative.Diagnostics> = _diagnostics.asStateFlow()

    inner class LocalBinder : Binder() {
        fun getService(): IdrNavigationService = this@IdrNavigationService
    }

    override fun onCreate() {
        super.onCreate()
        idrNative = IdrNative()
        idrNative.create()
        sensorManager = IdrSensorManager(this, idrNative)

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "IDR:NavigationWakeLock"
        ).apply {
            setReferenceCounted(false)
        }

        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> stopNavigation()
            ACTION_START, null -> startNavigation()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun startNavigation() {
        wakeLock?.acquire(4 * 60 * 60 * 1000L) // 4 hours timeout max
        idrNative.start()
        sensorManager?.start()

        val notification = buildNotification("Initializing Dead Reckoning...", "Acquiring GNSS fix")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        // Poll navigation state at 10 Hz
        pollJob?.cancel()
        pollJob = serviceScope.launch {
            var lastNotificationUpdate = 0L
            while (isActive) {
                if (idrNative.isRunning()) {
                    val state = idrNative.getNavigationState()
                    val diag = idrNative.getDiagnostics()
                    _navigationState.value = state
                    _diagnostics.value = diag

                    // Update ongoing notification every 1 second
                    val now = System.currentTimeMillis()
                    if (now - lastNotificationUpdate > 1000L) {
                        lastNotificationUpdate = now
                        updateNotification(state)
                    }
                }
                delay(100L) // 10 Hz
            }
        }

        Log.i(TAG, "IDR Navigation Service started")
    }

    private fun stopNavigation() {
        pollJob?.cancel()
        pollJob = null

        sensorManager?.stop()
        idrNative.stop()

        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        Log.i(TAG, "IDR Navigation Service stopped")
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "IDR Dead Reckoning Navigation",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows real-time navigation status, speed, and accuracy"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(title: String, text: String): Notification {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_NAVIGATION)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotification(state: IdrNative.NavigationState) {
        val modeStr = when (state.mode) {
            2 -> "GNSS Only"
            3 -> "GNSS + INS Fusion"
            4 -> "Degraded GNSS"
            5 -> "Pure Dead Reckoning"
            else -> "Aligning Sensors"
        }
        val speedKmh = state.speedMps * 3.6
        val title = "IDR: $modeStr"
        val content = String.format("%.0f km/h | Acc: ±%.1fm", speedKmh, state.horizontalAccuracyM)

        val notification = buildNotification(title, content)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        stopNavigation()
        idrNative.destroy()
        super.onDestroy()
    }
}
