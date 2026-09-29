// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// IdrEngine v2 — Top-level orchestrator with full Phase 2 integration.
//
// Owns all subsystems: INS, EKF, alignment, ZUPT, calibration,
// NHC, GNSS integrity. Provides the single entry point for
// feeding sensor data and retrieving navigation output.

#pragma once

#include "idr/alignment/phone_alignment.h"
#include "idr/calibration/online_calibrator.h"
#include "idr/constraints/nhc.h"
#include "idr/constraints/zupt.h"
#include "idr/engine/navigation_state.h"
#include "idr/fusion/eskf.h"
#include "idr/inertial/strapdown_ins.h"
#include "idr/integrity/gnss_integrity.h"
#include "idr/math_utils/geo_utils.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/sensors/sensor_types.h"
#include "idr/timing/timestamp_validator.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <cmath>
#include <functional>

namespace idr {

/// Engine configuration.
struct EngineConfig {
    VehicleProfile vehicle = profiles::kCar;
    ProcessNoise process_noise;
    IntegrityConfig integrity;
    AlignmentConfig alignment;
    ZuptConfig zupt;
    CalibrationConfig calibration;
    double output_rate_hz = 10.0;

    // NHC
    double nhc_lateral_sigma = 0.1;
    double nhc_vertical_sigma = 0.1;
};

/// IdrEngine — the top-level orchestrator.
///
/// Usage:
///   IdrEngine engine(config);
///   engine.setOutputCallback(callback);
///
///   // Feed data as it arrives:
///   engine.processImu(imu_sample);
///   engine.processGnss(gnss_measurement);
///   engine.processBaro(baro_sample);
///
///   // NavigationState is delivered via callback.
class IdrEngine {
public:
    using OutputCallback = std::function<void(const NavigationState&)>;

    explicit IdrEngine(const EngineConfig& config = {})
        : config_(config)
        , imu_validator_(100.0, 0.1, 1e-6)
        , integrity_monitor_(config.integrity)
        , alignment_(config.alignment)
        , zupt_detector_(config.zupt)
        , calibrator_(config.calibration) {
        ekf_.setProcessNoise(config.process_noise);
    }

    /// Set callback for navigation state output.
    void setOutputCallback(OutputCallback cb) {
        output_cb_ = std::move(cb);
    }

    /// Process a new IMU sample.
    /// This is the high-rate entry point (~100-200 Hz).
    void processImu(const ImuSample& imu) {
        // ── Timestamp validation ──
        auto tv = imu_validator_.validate(imu.timestamp_ns);
        if (!tv.valid) return;

        // ── Alignment: feed gravity ──
        Vec3d gravity_vec = imu.hasGravity() ? imu.gravity : imu.accel;
        alignment_.updateGravity(gravity_vec);

        // ── First-time initialization ──
        if (!ins_.isInitialized()) {
            imu_init_count_++;
            return;  // Wait for GNSS to initialize position
        }

        double dt = tv.first_sample
            ? (imu_validator_.estimatedRateHz() > 0.0 ? 1.0 / imu_validator_.estimatedRateHz() : 0.01)
            : tv.dt_sec;
        if (dt <= 0.0 || !std::isfinite(dt)) {
            dt = 0.01;
        }

        // ── ZUPT detection ──
        auto zupt_result = zupt_detector_.update(imu);

        // ── Online calibration ──
        if (zupt_result.is_stationary) {
            Vec3d g_dir = gravity_vec.normalized();
            calibrator_.addStationarySample(imu.accel, imu.gyro, g_dir);
        }

        // ── Apply phone-to-vehicle rotation ──
        ImuSample body_imu = imu;
        if (alignment_.quality() >= AlignmentQuality::GRAVITY) {
            Quaterniond R_vp = alignment_.rotation();
            body_imu.accel = R_vp * imu.accel;
            body_imu.gyro = R_vp * imu.gyro;
        }

        // ── Apply bias correction from calibrator ──
        const auto& cal = calibrator_.state();
        if (cal.accel_calibrated) {
            body_imu.accel -= cal.accel_bias;
        }
        if (cal.gyro_calibrated) {
            body_imu.gyro -= cal.gyro_bias;
        }

        // ── INS propagation ──
        ins_.propagate(body_imu, dt);

        // ── EKF prediction ──
        Vec3d accel_body = body_imu.accel - ins_.state().accel_bias;
        ekf_.predict(ins_.state(), accel_body, dt);

        // ── ZUPT pseudo-measurement ──
        if (zupt_result.zupt_available) {
            applyZupt();
        }

        // ── NHC (adaptive, every IMU step) ──
        if (!zupt_result.is_stationary) {
            applyNhc();
        }

        // ── Confidence decay ──
        alignment_.decayConfidence(dt);

        // ── Output at configured rate ──
        imu_count_++;
        int output_every = static_cast<int>(
            imu_validator_.estimatedRateHz() / config_.output_rate_hz);
        if (output_every < 1) output_every = 1;

        if (imu_count_ % static_cast<uint32_t>(output_every) == 0) {
            emitNavigationState(zupt_result);
        }
    }

    /// Process a new GNSS measurement.
    void processGnss(const GnssMeasurement& gnss) {
        if (!ins_.isInitialized()) {
            if (gnss.isValid()) {
                initializeFromGnss(gnss);
            }
            return;
        }

        // ── Update alignment with GNSS heading ──
        if (gnss.hasBearing() && gnss.speed_mps > 1.0f) {
            double heading_rad = static_cast<double>(gnss.bearing_deg) * constants::kDegToRad;
            alignment_.updateHeading(heading_rad, static_cast<double>(gnss.speed_mps));
        }

        // ── Update ZUPT with GNSS speed ──
        zupt_detector_.updateGnssSpeed(static_cast<double>(gnss.speed_mps));

        // ── Integrity assessment ──
        double ekf_heading = quat::toEulerZYX(ins_.state().attitude)[0];
        auto assessment = integrity_monitor_.assess(
            gnss,
            ins_.state().position,
            ins_.state().velocity,
            ekf_heading,
            ekf_.horizontalAccuracy()
        );

        gnss_integrity_ = assessment.state;

        if (!assessment.use_measurement) {
            spdlog::debug("GNSS rejected: integrity={}",
                          static_cast<int>(assessment.state));
            return;
        }

        // ── Convert GNSS to ENU ──
        Vec3d gnss_enu = geo::llaToEnu(
            gnss.latitude_deg, gnss.longitude_deg, gnss.altitude_m,
            origin_lat_, origin_lon_, origin_alt_
        );

        // ── GNSS position update ──
        double sigma_h = std::max(static_cast<double>(gnss.horizontal_accuracy_m), 1.0);
        double sigma_v = std::max(static_cast<double>(gnss.vertical_accuracy_m), 2.0);

        bool pos_accepted = ekf_.updateGnssPosition(
            gnss_enu, ins_.state().position, sigma_h, sigma_v
        );

        // ── GNSS velocity update ──
        if (gnss.speed_mps > 0.5f) {
            Vec3d gnss_vel_enu = Vec3d::Zero();
            if (gnss.hasBearing()) {
                double bearing_rad = static_cast<double>(gnss.bearing_deg) * constants::kDegToRad;
                double spd = static_cast<double>(gnss.speed_mps);
                gnss_vel_enu.x() = spd * std::sin(bearing_rad);
                gnss_vel_enu.y() = spd * std::cos(bearing_rad);
            }
            double sigma_vel = std::max(static_cast<double>(gnss.speed_accuracy_mps), 0.3);
            ekf_.updateGnssVelocity(gnss_vel_enu, ins_.state().velocity, sigma_vel);
        }

        // ── Apply correction to nominal state ──
        if (pos_accepted) {
            ekf_.applyToNominal(ins_);
        }

        updateNavigationMode();
    }

    /// Process a barometer sample.
    void processBaro(const BaroSample& baro) {
        if (!baro.isValid()) return;
        last_pressure_hpa_ = baro.pressure_hpa;

        if (!std::isfinite(ref_pressure_hpa_)) {
            ref_pressure_hpa_ = baro.pressure_hpa;
        }

        if (ins_.isInitialized()) {
            // Altitude offset from origin in meters (hypsometric approximation)
            double baro_alt_enu = 44330.0 * (1.0 - std::pow(baro.pressure_hpa / ref_pressure_hpa_, 0.190284));
            bool accepted = ekf_.updateBaroAltitude(baro_alt_enu, ins_.state().position.z(), 1.0);
            if (accepted) {
                ekf_.applyToNominal(ins_);
            }
        }
    }

    /// Process a wheel speed / vehicle odometry sample.
    void processWheelSpeed(const WheelSpeedSample& wheel) {
        if (!wheel.isValid() || !ins_.isInitialized()) return;

        bool accepted = ekf_.updateWheelSpeed(wheel.speed_mps, ins_.state(), wheel.accuracy_mps);
        if (accepted) {
            ekf_.applyToNominal(ins_);
        }
    }

    /// Process wheel speed with raw parameters.
    void processWheelSpeed(Timestamp timestamp_ns, double speed_mps, double accuracy_mps = 0.2) {
        WheelSpeedSample sample;
        sample.timestamp_ns = timestamp_ns;
        sample.speed_mps = speed_mps;
        sample.accuracy_mps = accuracy_mps;
        processWheelSpeed(sample);
    }

    /// Process a calibrated magnetometer sample.
    void processMagnetometer(const MagSample& mag) {
        if (!mag.isValid() || !ins_.isInitialized()) return;

        double field_mag = mag.field_ut.norm();
        // Reject magnetic anomaly: Earth's typical surface field is 25-65 µT
        if (field_mag < 20.0 || field_mag > 75.0) {
            return;
        }

        // Only compute heading if vehicle frame is leveled
        if (alignment_.quality() < AlignmentQuality::COARSE) {
            return;
        }

        // Rotate magnetic vector into leveled body frame
        Vec3d b_body = alignment_.rotation().toRotationMatrix().transpose() * mag.field_ut;
        double mag_heading = std::atan2(-b_body.y(), b_body.x());

        Vec3d euler = quat::toEulerZYX(ins_.state().attitude);
        double nom_yaw = euler.x();

        bool accepted = ekf_.updateMagneticHeading(mag_heading, nom_yaw, 0.1);
        if (accepted) {
            ekf_.applyToNominal(ins_);
        }
    }

    // ── Accessors ──
    const NavigationState& lastOutput() const { return last_output_; }
    const NominalState& nominalState() const { return ins_.state(); }
    NavigationMode navigationMode() const { return nav_mode_; }
    GnssIntegrity gnssIntegrity() const { return gnss_integrity_; }
    AlignmentQuality alignmentQuality() const { return alignment_.quality(); }
    bool isStationary() const { return zupt_detector_.isStationary(); }
    bool isInitialized() const { return ins_.isInitialized(); }
    uint32_t imuCount() const { return imu_count_ + static_cast<uint32_t>(imu_init_count_); }


private:
    void initializeFromGnss(const GnssMeasurement& gnss) {
        origin_lat_ = gnss.latitude_deg;
        origin_lon_ = gnss.longitude_deg;
        origin_alt_ = gnss.altitude_m;
        origin_set_ = true;

        NominalState init;
        init.timestamp_ns = gnss.timestamp_ns;
        init.position = Vec3d::Zero();

        if (gnss.hasBearing() && gnss.speed_mps > 1.0f) {
            double bearing = static_cast<double>(gnss.bearing_deg) * constants::kDegToRad;
            double spd = static_cast<double>(gnss.speed_mps);
            init.velocity.x() = spd * std::sin(bearing);
            init.velocity.y() = spd * std::cos(bearing);
        }

        // Initial attitude from alignment if available
        if (alignment_.quality() >= AlignmentQuality::GRAVITY) {
            init.attitude = alignment_.rotation();
        }

        // Apply calibration biases if available
        const auto& cal = calibrator_.state();
        if (cal.accel_calibrated) init.accel_bias = cal.accel_bias;
        if (cal.gyro_calibrated) init.gyro_bias = cal.gyro_bias;

        ins_.initialize(init);
        nav_mode_ = NavigationMode::GNSS_ONLY;

        ZuptResult zr;
        zr.is_stationary = zupt_detector_.isStationary();
        emitNavigationState(zr);

        spdlog::info("IDR Engine initialized: origin=({:.6f}, {:.6f}, {:.1f}), "
                     "alignment={}, cal_accel={}, cal_gyro={}",
                     origin_lat_, origin_lon_, origin_alt_,
                     static_cast<int>(alignment_.quality()),
                     cal.accel_calibrated, cal.gyro_calibrated);
    }

    void applyZupt() {
        // ZUPT: velocity = [0, 0, 0] with very low sigma
        Eigen::Matrix<double, 3, eskf::kStateSize> H =
            Eigen::Matrix<double, 3, eskf::kStateSize>::Zero();
        H.block<3, 3>(0, eskf::kVelIdx) = Mat3d::Identity();

        Vec3d z = -ins_.state().velocity;  // Innovation: 0 - current velocity

        double sigma = zupt_detector_.zuptSigma();
        Eigen::Matrix3d R = sigma * sigma * Eigen::Matrix3d::Identity();

        if (ekf_.update<3>(H, z, R)) {
            ekf_.applyToNominal(ins_);
        }
    }

    void applyNhc() {
        MotionClassDistribution mc;
        mc.probabilities[static_cast<int>(MotionClass::NORMAL_DRIVING)] = 0.8f;
        mc.probabilities[static_cast<int>(MotionClass::STATIONARY)] = 0.1f;
        mc.probabilities[static_cast<int>(MotionClass::ACCELERATION)] = 0.1f;

        double speed = ins_.state().velocity.norm();
        double yaw_rate = 0.0;
        double lat_accel = 0.0;

        auto nhc = computeAdaptiveNhc(
            config_.vehicle, mc, yaw_rate, speed, lat_accel,
            alignment_.alignmentState()
        );

        ekf_.updateNhc(ins_.state(), nhc.lateral_sigma, nhc.vertical_sigma);
    }

    void updateNavigationMode() {
        switch (gnss_integrity_) {
            case GnssIntegrity::HEALTHY:
                nav_mode_ = NavigationMode::GNSS_INS;
                break;
            case GnssIntegrity::DEGRADED:
                nav_mode_ = NavigationMode::DEGRADED;
                break;
            case GnssIntegrity::SUSPECT:
            case GnssIntegrity::DENIED:
                nav_mode_ = NavigationMode::DEAD_RECKONING;
                break;
        }
    }

    void emitNavigationState(const ZuptResult& zupt) {
        NavigationState state;

        if (origin_set_) {
            auto geo = geo::enuToLla(
                ins_.state().position, origin_lat_, origin_lon_, origin_alt_
            );
            state.latitude_deg  = geo.latitude_deg;
            state.longitude_deg = geo.longitude_deg;
            state.altitude_m    = geo.altitude_m;
        }

        state.velocity_east_mps  = ins_.state().velocity.x();
        state.velocity_north_mps = ins_.state().velocity.y();
        state.velocity_down_mps  = -ins_.state().velocity.z();
        state.speed_mps = ins_.state().velocity.norm();

        Vec3d euler = quat::toEulerZYX(ins_.state().attitude);
        state.heading_deg = angle::wrap2Pi(euler[0]) * constants::kRadToDeg;
        state.pitch_deg = euler[1] * constants::kRadToDeg;
        state.roll_deg  = euler[2] * constants::kRadToDeg;

        state.horizontal_accuracy_m = ekf_.horizontalAccuracy();
        auto vel_sigma = ekf_.velocitySigma();
        state.velocity_accuracy_mps = vel_sigma.norm();
        auto att_sigma = ekf_.attitudeSigma();
        state.heading_accuracy_deg = att_sigma.z() * constants::kRadToDeg;
        state.vertical_accuracy_m = ekf_.positionSigma().z();

        state.mode = nav_mode_;
        state.gnss_confidence = (gnss_integrity_ == GnssIntegrity::HEALTHY) ? 1.0 :
                                (gnss_integrity_ == GnssIntegrity::DEGRADED) ? 0.5 : 0.0;
        state.is_stationary = zupt.is_stationary;
        state.alignment_quality = static_cast<uint8_t>(alignment_.quality());

        state.timestamp_ns = ins_.state().timestamp_ns;
        state.update_count = imu_count_;

        last_output_ = state;

        if (output_cb_) {
            output_cb_(state);
        }
    }

    // ── Subsystems ──
    EngineConfig config_;
    StrapdownIns ins_;
    ErrorStateEkf ekf_;
    TimestampValidator imu_validator_;
    GnssIntegrityMonitor integrity_monitor_;
    PhoneToVehicleAlignment alignment_;
    ZuptDetector zupt_detector_;
    OnlineSensorCalibrator calibrator_;

    // ── State ──
    NavigationMode nav_mode_ = NavigationMode::INIT;
    GnssIntegrity gnss_integrity_ = GnssIntegrity::DENIED;
    NavigationState last_output_;

    // ── ENU origin ──
    double origin_lat_ = 0.0, origin_lon_ = 0.0, origin_alt_ = 0.0;
    bool origin_set_ = false;

    // ── Counters ──
    int imu_init_count_ = 0;
    uint32_t imu_count_ = 0;

    // ── Sensors ──
    double last_pressure_hpa_ = std::numeric_limits<double>::quiet_NaN();
    double ref_pressure_hpa_ = std::numeric_limits<double>::quiet_NaN();

    // ── Callbacks ──
    OutputCallback output_cb_;
};

}  // namespace idr
