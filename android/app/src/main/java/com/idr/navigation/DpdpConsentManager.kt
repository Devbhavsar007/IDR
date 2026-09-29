package com.idr.navigation

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.os.Build

/**
 * DPDP Act 2023 Compliance Manager.
 *
 * Implements consent flows, data minimization, and privacy controls
 * as required by the Digital Personal Data Protection Act 2023 (India).
 *
 * Key requirements:
 *   1. Explicit consent before collecting location/sensor data
 *   2. Purpose limitation — data used only for navigation
 *   3. Data minimization — no cloud upload by default
 *   4. Right to erasure — user can delete all stored data
 *   5. Consent withdrawal — user can revoke at any time
 *   6. Data principal rights — export, correction, grievance
 */
class DpdpConsentManager(private val context: Context) {

    companion object {
        private const val PREFS_NAME = "idr_dpdp_consent"
        private const val KEY_CONSENT_GIVEN = "consent_given"
        private const val KEY_CONSENT_TIMESTAMP = "consent_timestamp"
        private const val KEY_CONSENT_VERSION = "consent_version"
        private const val KEY_SENSOR_LOGGING = "sensor_logging_consent"
        private const val KEY_CLOUD_UPLOAD = "cloud_upload_consent"
        private const val KEY_ANALYTICS = "analytics_consent"
        private const val KEY_CRASH_REPORTS = "crash_reports_consent"

        /** Current consent document version. Bump when T&C change. */
        const val CURRENT_CONSENT_VERSION = 1
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Data processing purposes (DPDP §4) */
    enum class Purpose {
        NAVIGATION,       // Core — required for app functionality
        SENSOR_LOGGING,   // Optional — record IMU for model improvement
        CLOUD_UPLOAD,     // Optional — upload anonymized trajectories
        ANALYTICS,        // Optional — usage analytics
        CRASH_REPORTS     // Optional — crash/error reporting
    }

    /** Consent state */
    data class ConsentState(
        val consentGiven: Boolean = false,
        val consentVersion: Int = 0,
        val consentTimestamp: Long = 0,
        val sensorLogging: Boolean = false,
        val cloudUpload: Boolean = false,
        val analytics: Boolean = false,
        val crashReports: Boolean = false
    ) {
        /** Whether core navigation consent is valid and up-to-date */
        val isValid: Boolean
            get() = consentGiven && consentVersion >= CURRENT_CONSENT_VERSION
    }

    /** Get current consent state */
    fun getConsentState(): ConsentState {
        return ConsentState(
            consentGiven = prefs.getBoolean(KEY_CONSENT_GIVEN, false),
            consentVersion = prefs.getInt(KEY_CONSENT_VERSION, 0),
            consentTimestamp = prefs.getLong(KEY_CONSENT_TIMESTAMP, 0),
            sensorLogging = prefs.getBoolean(KEY_SENSOR_LOGGING, false),
            cloudUpload = prefs.getBoolean(KEY_CLOUD_UPLOAD, false),
            analytics = prefs.getBoolean(KEY_ANALYTICS, false),
            crashReports = prefs.getBoolean(KEY_CRASH_REPORTS, false)
        )
    }

    /** Record user consent (call after showing consent dialog) */
    fun grantConsent(
        sensorLogging: Boolean = false,
        cloudUpload: Boolean = false,
        analytics: Boolean = false,
        crashReports: Boolean = false
    ) {
        prefs.edit().apply {
            putBoolean(KEY_CONSENT_GIVEN, true)
            putInt(KEY_CONSENT_VERSION, CURRENT_CONSENT_VERSION)
            putLong(KEY_CONSENT_TIMESTAMP, System.currentTimeMillis())
            putBoolean(KEY_SENSOR_LOGGING, sensorLogging)
            putBoolean(KEY_CLOUD_UPLOAD, cloudUpload)
            putBoolean(KEY_ANALYTICS, analytics)
            putBoolean(KEY_CRASH_REPORTS, crashReports)
            apply()
        }
    }

    /** Withdraw all consent (DPDP §6) — stops all optional data processing */
    fun withdrawConsent() {
        prefs.edit().apply {
            putBoolean(KEY_CONSENT_GIVEN, false)
            putBoolean(KEY_SENSOR_LOGGING, false)
            putBoolean(KEY_CLOUD_UPLOAD, false)
            putBoolean(KEY_ANALYTICS, false)
            putBoolean(KEY_CRASH_REPORTS, false)
            apply()
        }
    }

    /** Right to erasure (DPDP §12) — delete all locally stored data */
    fun eraseAllData() {
        // Clear consent preferences
        prefs.edit().clear().apply()

        // Delete sensor logs
        context.filesDir.resolve("sensor_logs").deleteRecursively()

        // Delete cached map data
        context.cacheDir.resolve("maps").deleteRecursively()

        // Delete ML model cache
        context.filesDir.resolve("models").deleteRecursively()

        // Delete any exported trajectories
        context.filesDir.resolve("exports").deleteRecursively()
    }

    /** Check if a specific purpose is consented */
    fun isConsentedFor(purpose: Purpose): Boolean {
        val state = getConsentState()
        if (!state.isValid) return false

        return when (purpose) {
            Purpose.NAVIGATION -> true  // Implied by core consent
            Purpose.SENSOR_LOGGING -> state.sensorLogging
            Purpose.CLOUD_UPLOAD -> state.cloudUpload
            Purpose.ANALYTICS -> state.analytics
            Purpose.CRASH_REPORTS -> state.crashReports
        }
    }

    /** Required Android runtime permissions */
    fun requiredPermissions(): List<String> {
        val perms = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            perms.add(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }

        return perms
    }

    /** Generate a data processing notice (for display in consent dialog) */
    fun getConsentNotice(): String {
        return """
            |IDR Navigation — Data Processing Notice
            |
            |Under the Digital Personal Data Protection Act, 2023 (India),
            |we inform you that this app processes the following data:
            |
            |REQUIRED (for navigation):
            |• GPS/GNSS location data
            |• Accelerometer and gyroscope sensor data
            |• Barometric pressure sensor data
            |
            |All sensor data is processed LOCALLY on your device.
            |No data is sent to any server unless you explicitly opt in.
            |
            |OPTIONAL (with your consent):
            |• Sensor data logging for navigation improvement
            |• Anonymized trajectory upload for model training
            |• Usage analytics
            |• Crash/error reports
            |
            |Your rights under DPDP Act 2023:
            |• Right to access your data
            |• Right to correction
            |• Right to erasure (delete all data)
            |• Right to withdraw consent at any time
            |• Right to grievance redressal
            |
            |Data Fiduciary: [Your Organization Name]
            |Contact: dpo@idr-navigation.in
        """.trimMargin()
    }
}
