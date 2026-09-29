package com.idr.navigation.map

import com.idr.navigation.NavigationState
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * OfflineMapController
 *
 * Manages rendering and state projection for offline dead-reckoning navigation.
 * Renders the vehicle position marker, heading directional chevron,
 * multi-level flyover/underpass layer colors, and the 95% confidence covariance ellipse.
 */
class OfflineMapController {

    data class ErrorEllipse(
        val semiMajorAxisMeters: Double,
        val semiMinorAxisMeters: Double,
        val orientationDegrees: Double
    )

    data class MapVisualState(
        val latitude: Double,
        val longitude: Double,
        val altitudeMeters: Double,
        val headingDegrees: Double,
        val speedKmh: Double,
        val layer: Int,
        val layerColorHex: String,
        val errorEllipse: ErrorEllipse,
        val isDeadReckoning: Boolean
    )

    // Layer-specific styling constants for Indian highway corridors
    companion object {
        const val COLOR_FLYOVER_ELEVATED = "#10B981" // Emerald Green (Layer 1+)
        const val COLOR_SURFACE_AT_GRADE = "#3B82F6" // Electric Blue (Layer 0)
        const val COLOR_UNDERPASS_TUNNEL = "#F59E0B" // Amber / Gold (Layer -1)

        // 95% confidence scale factor for 2-DOF bivariate normal distribution
        // sqrt(chi2inv(0.95, 2)) = sqrt(5.991) ≈ 2.4477
        const val CHI2_95_SCALE = 2.4477
    }

    private var currentVisualState: MapVisualState? = null

    /**
     * Compute visual state from native fused navigation state.
     *
     * @param navState Raw state from C++ IdrEngine
     * @param activeLayer Road layer from HMM map-matcher (-1: underpass, 0: surface, 1: flyover)
     */
    fun processNavigationUpdate(navState: NavigationState, activeLayer: Int = 0): MapVisualState {
        // Calculate 95% confidence uncertainty ellipse
        val sigmaH = navState.horizontalAccuracyMeters.coerceAtLeast(0.5)
        val ellipse95 = ErrorEllipse(
            semiMajorAxisMeters = sigmaH * CHI2_95_SCALE,
            semiMinorAxisMeters = (sigmaH * 0.8) * CHI2_95_SCALE,
            orientationDegrees = navState.headingDeg
        )

        val layerColor = when {
            activeLayer > 0 -> COLOR_FLYOVER_ELEVATED
            activeLayer < 0 -> COLOR_UNDERPASS_TUNNEL
            else -> COLOR_SURFACE_AT_GRADE
        }

        val state = MapVisualState(
            latitude = navState.latitude,
            longitude = navState.longitude,
            altitudeMeters = navState.altitude,
            headingDegrees = navState.headingDeg,
            speedKmh = navState.speedKmh,
            layer = activeLayer,
            layerColorHex = layerColor,
            errorEllipse = ellipse95,
            isDeadReckoning = navState.isDeadReckoningActive
        )

        currentVisualState = state
        return state
    }

    /**
     * Extrapolate vehicle position forward in time during UI rendering frames.
     * Ensures fluid 60-120 fps animations between high-rate filter ticks.
     */
    fun extrapolatePosition(deltaSeconds: Double): Pair<Double, Double>? {
        val current = currentVisualState ?: return null
        if (deltaSeconds <= 0.0 || current.speedKmh <= 0.5) {
            return Pair(current.latitude, current.longitude)
        }

        val speedMps = current.speedKmh / 3.6
        val distanceMeters = speedMps * deltaSeconds

        // Forward geodesic extrapolation
        val headingRad = Math.toRadians(current.headingDegrees)
        val deltaNorthMeters = distanceMeters * cos(headingRad)
        val deltaEastMeters = distanceMeters * sin(headingRad)

        // Earth radius ~6378137m
        val latRad = Math.toRadians(current.latitude)
        val deltaLatDeg = deltaNorthMeters / 111132.954
        val deltaLonDeg = deltaEastMeters / (111132.954 * cos(latRad))

        return Pair(current.latitude + deltaLatDeg, current.longitude + deltaLonDeg)
    }

    fun getCurrentState(): MapVisualState? = currentVisualState
}
