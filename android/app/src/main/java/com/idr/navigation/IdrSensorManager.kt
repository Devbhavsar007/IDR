package com.idr.navigation

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log

/**
 * IDR Sensor Manager.
 *
 * Captures high-frequency IMU (accel, gyro, gravity), barometer, and GNSS
 * measurements on a dedicated background thread and marshals them into [IdrNative].
 *
 * Threading model:
 *   SensorEventListener and LocationListener callbacks run on a dedicated HandlerThread
 *   to avoid dropping IMU samples (100–200 Hz) due to UI thread contention.
 */
class IdrSensorManager(
    private val context: Context,
    private val idrNative: IdrNative
) : SensorEventListener, LocationListener {

    companion object {
        private const val TAG = "IdrSensorManager"
        private const val IMU_RATE_US = 10_000 // 100 Hz (10,000 microseconds)
    }

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager: LocationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    // Background thread for sensor event dispatching
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null

    // Sensors
    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravitySensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val pressureSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)

    // Temporary sample buffers
    private val latestAccel = DoubleArray(3)
    private val latestGyro = DoubleArray(3)
    private val latestGravity = DoubleArray(3)
    private var hasGravity = false
    private var hasAccel = false
    private var hasGyro = false

    // Satellite tracking
    @Volatile
    var satelliteCount: Int = 0
        private set

    private var gnssStatusCallback: GnssStatus.Callback? = null

    @Volatile
    var isRunning: Boolean = false
        private set

    /**
     * Start high-rate sensor and GNSS acquisition.
     */
    @SuppressLint("MissingPermission")
    fun start() {
        if (isRunning) return

        sensorThread = HandlerThread("IdrSensorThread", Thread.MAX_PRIORITY).apply {
            start()
            sensorHandler = Handler(looper)
        }

        val handler = sensorHandler

        // Register IMU sensors
        accelerometer?.let {
            sensorManager.registerListener(this, it, IMU_RATE_US, handler)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, IMU_RATE_US, handler)
        }
        gravitySensor?.let {
            sensorManager.registerListener(this, it, IMU_RATE_US, handler)
        }
        pressureSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL, handler)
        }

        // Register GNSS
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L, // 1 Hz minimum interval
                    0f,    // 0m distance threshold
                    this,
                    sensorThread?.looper
                )
            }

            // Satellite status callback
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                val callback = object : GnssStatus.Callback() {
                    override fun onSatelliteStatusChanged(status: GnssStatus) {
                        var usedInFix = 0
                        for (i in 0 until status.satelliteCount) {
                            if (status.usedInFix(i)) {
                                usedInFix++
                            }
                        }
                        satelliteCount = usedInFix
                    }
                }
                gnssStatusCallback = callback
                locationManager.registerGnssStatusCallback(callback, handler)
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "Location permission missing: ${e.message}")
        }

        isRunning = true
        Log.i(TAG, "Sensor acquisition started (100Hz IMU)")
    }

    /**
     * Stop sensor and GNSS acquisition.
     */
    fun stop() {
        if (!isRunning) return

        sensorManager.unregisterListener(this)
        try {
            locationManager.removeUpdates(this)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                gnssStatusCallback?.let {
                    locationManager.unregisterGnssStatusCallback(it)
                }
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "Error removing location updates: ${e.message}")
        }

        sensorThread?.quitSafely()
        sensorThread = null
        sensorHandler = null

        isRunning = false
        Log.i(TAG, "Sensor acquisition stopped")
    }

    // ── SensorEventListener ──

    override fun onSensorChanged(event: SensorEvent) {
        val tNs = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccel[0] = event.values[0].toDouble()
                latestAccel[1] = event.values[1].toDouble()
                latestAccel[2] = event.values[2].toDouble()
                hasAccel = true

                // Feed IMU on accelerometer tick if gyroscope is also available
                if (hasGyro) {
                    idrNative.feedImu(
                        tNs,
                        latestAccel,
                        latestGyro,
                        if (hasGravity) latestGravity else null
                    )
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyro[0] = event.values[0].toDouble()
                latestGyro[1] = event.values[1].toDouble()
                latestGyro[2] = event.values[2].toDouble()
                hasGyro = true
            }
            Sensor.TYPE_GRAVITY -> {
                latestGravity[0] = event.values[0].toDouble()
                latestGravity[1] = event.values[1].toDouble()
                latestGravity[2] = event.values[2].toDouble()
                hasGravity = true
            }
            Sensor.TYPE_PRESSURE -> {
                val pressureHpa = event.values[0].toDouble()
                idrNative.feedBaro(tNs, pressureHpa)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle sensor calibration changes if necessary
    }

    // ── LocationListener ──

    override fun onLocationChanged(location: Location) {
        val tNs = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR1) {
            location.elapsedRealtimeNanos
        } else {
            SystemClock.elapsedRealtimeNanos()
        }

        val speedMps = if (location.hasSpeed()) location.speed else 0f
        val bearingDeg = if (location.hasBearing()) location.bearing else 0f
        val hAccM = if (location.hasAccuracy()) location.accuracy else 25.0f
        val vAccM = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && location.hasVerticalAccuracy()) {
            location.verticalAccuracyMeters
        } else {
            hAccM * 1.5f
        }

        val fixType = if (location.hasAltitude()) 2 else 1 // 2=3D, 1=2D

        idrNative.feedGnss(
            timestampNs = tNs,
            latDeg = location.latitude,
            lonDeg = location.longitude,
            altM = location.altitude,
            speedMps = speedMps,
            bearingDeg = bearingDeg,
            hAccuracyM = hAccM,
            vAccuracyM = vAccM,
            fixType = fixType,
            satCount = satelliteCount
        )
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
