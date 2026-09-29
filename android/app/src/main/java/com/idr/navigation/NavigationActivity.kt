package com.idr.navigation

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * IDR Main Navigation Activity.
 *
 * Provides real-time navigation visualization, mode alerts, uncertainty metrics,
 * and diagnostics for intelligent dead reckoning.
 */
class NavigationActivity : ComponentActivity() {

    private lateinit var consentManager: DpdpConsentManager
    private var navigationService: IdrNavigationService? = null
    private var isBound = false

    // UI elements
    private lateinit var tvModeBanner: TextView
    private lateinit var tvSpeed: TextView
    private lateinit var tvHeading: TextView
    private lateinit var tvAccuracy: TextView
    private lateinit var tvCoordinates: TextView
    private lateinit var tvDiagnostics: TextView
    private lateinit var btnToggleNav: Button

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineLocationGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        if (fineLocationGranted) {
            checkConsentAndStartService()
        } else {
            Toast.makeText(this, "Location permission is required for navigation", Toast.LENGTH_LONG).show()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as IdrNavigationService.LocalBinder
            navigationService = localBinder.getService()
            isBound = true
            observeNavigationData()
            updateButtonState(true)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            navigationService = null
            isBound = false
            updateButtonState(false)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        consentManager = DpdpConsentManager(this)

        setContentView(buildContentView())
        requestRequiredPermissions()
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val allGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

        if (allGranted) {
            checkConsentAndStartService()
        } else {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }

    private fun checkConsentAndStartService() {
        if (!consentManager.getConsentState().isValid) {
            DpdpConsentDialog(this, consentManager) { granted ->
                if (granted) {
                    bindNavigationService()
                } else {
                    Toast.makeText(this, "Consent is required under DPDP Act 2023", Toast.LENGTH_SHORT).show()
                }
            }.show()
        } else {
            bindNavigationService()
        }
    }

    private fun bindNavigationService() {
        val serviceIntent = Intent(this, IdrNavigationService::class.java).apply {
            action = IdrNavigationService.ACTION_START
        }
        startService(serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun observeNavigationData() {
        val service = navigationService ?: return

        lifecycleScope.launch {
            service.navigationState.collectLatest { state ->
                val speedKmh = state.speedMps * 3.6
                tvSpeed.text = String.format("%.0f", speedKmh)
                tvHeading.text = String.format("Heading: %.0f°", state.headingDeg)
                tvAccuracy.text = String.format("Accuracy: ±%.1f m", state.horizontalAccuracyM)
                tvCoordinates.text = String.format("Lat: %.6f, Lon: %.6f", state.latitudeDeg, state.longitudeDeg)

                when (state.mode) {
                    3 -> { // GNSS_INS
                        tvModeBanner.text = "● GNSS + INS FUSION ACTIVE"
                        tvModeBanner.setBackgroundColor(Color.parseColor("#1B5E20")) // Deep Green
                    }
                    5 -> { // DEAD_RECKONING
                        tvModeBanner.text = "⚡ PURE DEAD RECKONING (GPS OUTAGE)"
                        tvModeBanner.setBackgroundColor(Color.parseColor("#0D47A1")) // Deep Blue
                    }
                    4 -> { // DEGRADED
                        tvModeBanner.text = "⚠ DEGRADED GNSS (MULTIPATH / INTERFERENCE)"
                        tvModeBanner.setBackgroundColor(Color.parseColor("#E65100")) // Amber
                    }
                    else -> {
                        tvModeBanner.text = "ALIGNING SENSORS..."
                        tvModeBanner.setBackgroundColor(Color.parseColor("#37474F")) // Gray
                    }
                }
            }
        }

        lifecycleScope.launch {
            service.diagnostics.collectLatest { diag ->
                val alignStr = when (diag.alignmentQuality) {
                    3 -> "Converged"
                    2 -> "Coarse"
                    1 -> "Gravity"
                    else -> "None"
                }
                tvDiagnostics.text = String.format(
                    "Diagnostics: IMU Updates: %d | Alignment: %s | Stationary: %b",
                    diag.imuCount, alignStr, diag.isStationary
                )
            }
        }
    }

    private fun updateButtonState(running: Boolean) {
        btnToggleNav.text = if (running) "Stop Navigation" else "Start Navigation"
        btnToggleNav.setBackgroundColor(if (running) Color.parseColor("#B71C1C") else Color.parseColor("#1B5E20"))
    }

    private fun buildContentView(): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#121212")) // Dark theme
            setPadding(32, 48, 32, 48)

            // Mode banner
            tvModeBanner = TextView(this@NavigationActivity).apply {
                text = "INITIALIZING..."
                setTextColor(Color.WHITE)
                textSize = 15f
                setPadding(24, 16, 24, 16)
                textAlignment = View.TEXT_ALIGNMENT_CENTER
                setBackgroundColor(Color.parseColor("#37474F"))
            }
            addView(tvModeBanner)

            // Speedometer layout
            val speedContainer = LinearLayout(this@NavigationActivity).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(0, 48, 0, 32)
                textAlignment = View.TEXT_ALIGNMENT_CENTER
            }
            tvSpeed = TextView(this@NavigationActivity).apply {
                text = "0"
                setTextColor(Color.WHITE)
                textSize = 72f
                textAlignment = View.TEXT_ALIGNMENT_CENTER
            }
            val tvSpeedUnit = TextView(this@NavigationActivity).apply {
                text = "km / h"
                setTextColor(Color.LTGRAY)
                textSize = 16f
                textAlignment = View.TEXT_ALIGNMENT_CENTER
            }
            speedContainer.addView(tvSpeed)
            speedContainer.addView(tvSpeedUnit)
            addView(speedContainer)

            // Heading & Accuracy
            tvHeading = TextView(this@NavigationActivity).apply {
                text = "Heading: 0°"
                setTextColor(Color.WHITE)
                textSize = 18f
                setPadding(0, 8, 0, 8)
            }
            tvAccuracy = TextView(this@NavigationActivity).apply {
                text = "Accuracy: ±0.0 m"
                setTextColor(Color.LTGRAY)
                textSize = 16f
                setPadding(0, 8, 0, 8)
            }
            tvCoordinates = TextView(this@NavigationActivity).apply {
                text = "Lat: 0.000000, Lon: 0.000000"
                setTextColor(Color.LTGRAY)
                textSize = 14f
                setPadding(0, 8, 0, 16)
            }
            addView(tvHeading)
            addView(tvAccuracy)
            addView(tvCoordinates)

            // Diagnostics line
            tvDiagnostics = TextView(this@NavigationActivity).apply {
                text = "Diagnostics: Initializing..."
                setTextColor(Color.GRAY)
                textSize = 12f
                setPadding(0, 16, 0, 32)
            }
            addView(tvDiagnostics)

            // Toggle button
            btnToggleNav = Button(this@NavigationActivity).apply {
                text = "Start Navigation"
                setTextColor(Color.WHITE)
                textSize = 16f
                setBackgroundColor(Color.parseColor("#1B5E20"))
                setOnClickListener {
                    if (isBound) {
                        unbindService(serviceConnection)
                        stopService(Intent(this@NavigationActivity, IdrNavigationService::class.java))
                        isBound = false
                        updateButtonState(false)
                    } else {
                        bindNavigationService()
                    }
                }
            }
            addView(btnToggleNav)
        }
    }

    override fun onDestroy() {
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
        super.onDestroy()
    }
}
