#!/usr/bin/env python3
"""
IDR Phase 11 — GNSS Anti-Spoofing and Continuous Integrity Verification Suite
Simulates intentional coordinate offset jumps, deceptive trajectory curvatures,
and synthetic C/N0 uniformity patterns to verify spoofing defense algorithms.
"""

import math
import pytest
from typing import Dict, List, Any


class PySpoofingDetector:
    def __init__(self,
                 weight_velocity: float = 0.35,
                 weight_curvature: float = 0.30,
                 weight_clock: float = 0.15,
                 weight_cn0: float = 0.20):
        self.w_vel = weight_velocity
        self.w_curv = weight_curvature
        self.w_clk = weight_clock
        self.w_cn0 = weight_cn0
        self.suspect_epochs = 0
        self.alert_epochs = 0

    def check_velocity(self, gnss_vel: float, imu_vel: float, sigma: float = 0.5) -> float:
        mismatch = abs(gnss_vel - imu_vel)
        return min(max(mismatch / (3.0 * sigma), 0.0), 1.0)

    def check_curvature(self, gnss_heading_rate: float, gyro_yaw_rate: float) -> float:
        diff = abs(gnss_heading_rate - gyro_yaw_rate)
        return min(max(diff / (3.0 * 0.05), 0.0), 1.0)

    def check_cn0(self, cn0_values: List[float]) -> float:
        if len(cn0_values) < 5:
            return 0.0
        mean = sum(cn0_values) / len(cn0_values)
        variance = sum((x - mean) ** 2 for x in cn0_values) / len(cn0_values)
        std_dev = math.sqrt(variance)
        if std_dev < 2.0:
            return 0.85
        elif std_dev < 3.0:
            return 0.30
        return 0.0

    def compute_score(self, vel_score: float, curv_score: float, cn0_score: float, clk_score: float = 0.0) -> float:
        return self.w_vel * vel_score + self.w_curv * curv_score + self.w_cn0 * cn0_score + self.w_clk * clk_score

    def update(self, score: float) -> str:
        if score >= 0.80:
            self.alert_epochs += 1
            self.suspect_epochs += 1
        elif score >= 0.60:
            self.suspect_epochs += 1
            self.alert_epochs = max(0, self.alert_epochs - 1)
        else:
            self.suspect_epochs = max(0, self.suspect_epochs - 1)
            self.alert_epochs = max(0, self.alert_epochs - 1)

        if self.alert_epochs >= 5:
            return "DENIED"
        elif self.suspect_epochs >= 3:
            return "SUSPECT"
        return "HEALTHY"


def test_clean_gnss_signals_pass_integrity():
    detector = PySpoofingDetector()
    vel_s = detector.check_velocity(15.2, 15.0, 0.5)
    curv_s = detector.check_curvature(0.01, 0.012)
    cn0_s = detector.check_cn0([28.0, 34.0, 42.0, 48.0, 31.0, 39.0])
    score = detector.compute_score(vel_s, curv_s, cn0_s)

    assert score < 0.25
    state = detector.update(score)
    assert state == "HEALTHY"


def test_velocity_spoofing_attack_detected():
    detector = PySpoofingDetector()
    # Attacker reports 25 m/s while vehicle is stationary at a red light
    vel_s = detector.check_velocity(25.0, 0.0, 0.5)
    assert vel_s > 0.90


def test_curvature_spoofing_attack_detected():
    detector = PySpoofingDetector()
    # Attacker reports heading straight, but vehicle is undergoing sharp roundabout turn (0.2 rad/s)
    curv_s = detector.check_curvature(0.0, 0.20)
    assert curv_s > 0.90


def test_cn0_uniformity_spoofing_attack_detected():
    detector = PySpoofingDetector()
    # Typical software-defined radio (SDR) GPS spoofer with uniform RF attenuation
    uniform_cn0 = [41.2, 41.1, 41.3, 41.0, 41.2, 41.1]
    cn0_s = detector.check_cn0(uniform_cn0)
    assert cn0_s >= 0.85


def test_hysteresis_state_machine_transition():
    detector = PySpoofingDetector()
    # Consecutive severe attacks
    states = [detector.update(0.85) for _ in range(6)]
    assert states[0] == "HEALTHY"
    assert states[2] == "SUSPECT"
    assert states[4] == "DENIED"
    assert states[5] == "DENIED"
