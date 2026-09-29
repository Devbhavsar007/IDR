package com.idr.navigation

/**
 * IDR — Intelligent Dead Reckoning with GNSS Fusion.
 *
 * Kotlin binding to the native C++ IdrEngine via JNI.
 *
 * Usage:
 *   val idr = IdrNative()
 *   idr.create()
 *   idr.setVehicleProfile("car")
 *   idr.start()
 *
 *   // In sensor listeners:
 *   idr.feedImu(timestamp, accel, gyro, gravity)
 *   idr.feedGnss(timestamp, lat, lon, alt, speed, bearing, hAcc, vAcc, fixType, satCount)
 *
 *   // From UI thread:
 *   val state = idr.getNavigationState()
 *
 *   // Cleanup:
 *   idr.stop()
 *   idr.destroy()
 */
class IdrNative {

    companion object {
        init {
            System.loadLibrary("idr-native")
        }
    }

    // ── Native handle ──
    private var nativeHandle: Long = 0

    // ── Lifecycle ──

    /** Create the native engine. Call from Application.onCreate(). */
    external fun create(): Boolean

    /** Destroy the native engine. Call from onDestroy(). */
    external fun destroy()

    /** Set vehicle profile: "car", "bike", "auto_rickshaw", "bus", "truck" */
    external fun setVehicleProfile(profile: String)

    /** Load the ONNX model for AI inference. Returns true on success. */
    external fun loadModel(modelPath: String): Boolean

    /** Start the engine. */
    external fun start()

    /** Stop the engine. */
    external fun stop()

    // ── Sensor input ──

    /**
     * Feed an IMU sample.
     * @param timestampNs  Monotonic nanosecond timestamp (SystemClock.elapsedRealtimeNanos())
     * @param accel        Accelerometer [x, y, z] in m/s²
     * @param gyro         Gyroscope [x, y, z] in rad/s
     * @param gravity      Gravity sensor [x, y, z] in m/s² (may be null)
     */
    external fun feedImu(
        timestampNs: Long,
        accel: DoubleArray,
        gyro: DoubleArray,
        gravity: DoubleArray?
    )

    /**
     * Feed a GNSS measurement.
     * @param timestampNs  Monotonic nanosecond timestamp
     * @param latDeg       WGS84 latitude (degrees)
     * @param lonDeg       WGS84 longitude (degrees)
     * @param altM         Altitude MSL (meters)
     * @param speedMps     Speed (m/s)
     * @param bearingDeg   Bearing (degrees CW from North)
     * @param hAccuracyM   Horizontal accuracy (meters, 68%)
     * @param vAccuracyM   Vertical accuracy (meters, 68%)
     * @param fixType      Fix type (0=none, 1=2D, 2=3D, 3=RTK)
     * @param satCount     Number of satellites used
     */
    external fun feedGnss(
        timestampNs: Long,
        latDeg: Double,
        lonDeg: Double,
        altM: Double,
        speedMps: Float,
        bearingDeg: Float,
        hAccuracyM: Float,
        vAccuracyM: Float,
        fixType: Int,
        satCount: Int
    )

    /**
     * Feed a barometer sample.
     * @param timestampNs  Monotonic nanosecond timestamp
     * @param pressureHpa  Atmospheric pressure in hPa
     */
    external fun feedBaro(timestampNs: Long, pressureHpa: Double)

    // ── Output ──

    /**
     * Get the latest navigation state.
     * Returns a NavigationState data class with all fields populated.
     * Thread-safe — can be called from UI thread.
     */
    external fun getNavigationState(): NavigationState

    /**
     * Get engine diagnostics for debugging overlay.
     */
    external fun getDiagnostics(): Diagnostics

    // ── Status ──

    external fun isRunning(): Boolean

    // ── Data classes ──

    data class NavigationState(
        val latitudeDeg: Double = 0.0,
        val longitudeDeg: Double = 0.0,
        val altitudeM: Double = 0.0,
        val speedMps: Double = 0.0,
        val headingDeg: Double = 0.0,
        val pitchDeg: Double = 0.0,
        val rollDeg: Double = 0.0,
        val horizontalAccuracyM: Double = 999.0,
        val verticalAccuracyM: Double = 999.0,
        val velocityAccuracyMps: Double = 999.0,
        val headingAccuracyDeg: Double = 999.0,
        val mode: Int = 0,  // 0=INIT, 1=GNSS_ONLY, 2=GNSS_INS, 3=DR, 4=DEGRADED
        val gnssConfidence: Double = 0.0,
        val isStationary: Boolean = false,
        val alignmentQuality: Int = 0,  // 0=NONE, 1=GRAVITY, 2=COARSE, 3=CONVERGED
        val matchedRoadId: Long = -1,
        val matchedRoadConfidence: Double = 0.0,
        val roadLevel: Int = 0,
        val timestampNs: Long = 0
    )

    data class Diagnostics(
        val mode: Int = 0,
        val gnssIntegrity: Int = 0,
        val alignmentQuality: Int = 0,
        val isStationary: Boolean = false,
        val modelLoaded: Boolean = false,
        val horizontalAccuracyM: Double = 999.0,
        val imuCount: Long = 0
    )
}
