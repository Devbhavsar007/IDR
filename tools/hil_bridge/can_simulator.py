#!/usr/bin/env python3
"""
IDR — Hardware-in-the-Loop (HIL) Simulator & CAN Telemetry Bridge
Simulates real-time vehicle CAN bus frames (OBD-II PID 0x0D speed, 4-wheel speeds, steering angle)
synchronized with high-rate IMU and GNSS fixes with configurable environmental fault injection.
"""

import math
import struct
import time
from typing import Dict, List, Any, Optional, Tuple


class CanMessage:
    def __init__(self, arbitration_id: int, data: bytes, timestamp_s: float):
        self.arbitration_id = arbitration_id
        self.data = data
        self.timestamp_s = timestamp_s

    def hex_payload(self) -> str:
        return self.data.hex().upper()


class VehicleCanSimulator:
    # Standard automotive CAN IDs
    CAN_ID_OBD2_RESPONSE = 0x7E8     # OBD-II response
    CAN_ID_WHEEL_SPEEDS = 0x0B4      # Individual wheel pulse counters
    CAN_ID_STEERING_ANGLE = 0x025    # Steering sensor
    CAN_ID_YAW_RATE = 0x120          # Chassis stability unit

    def __init__(self, dt_s: float = 0.01):
        self.dt_s = dt_s
        self.current_time_s = 0.0

        # Vehicle dynamics state
        self.speed_mps = 0.0
        self.heading_rad = 0.0
        self.steering_angle_deg = 0.0
        self.yaw_rate_rads = 0.0
        self.elevation_m = 0.0

        # Scenario triggers
        self.gnss_outage_active = False

    def set_speed(self, speed_kmh: float):
        self.speed_mps = speed_kmh / 3.6

    def set_steering(self, angle_deg: float):
        self.steering_angle_deg = angle_deg
        # Bicycle model approximation for yaw rate: r = (v / L) * tan(delta)
        wheelbase_m = 2.7
        self.yaw_rate_rads = (self.speed_mps / wheelbase_m) * math.tan(math.radians(angle_deg * 0.06))

    def step(self) -> List[CanMessage]:
        self.current_time_s += self.dt_s
        self.heading_rad += self.yaw_rate_rads * self.dt_s

        messages = []

        # 1. OBD-II PID 0x0D Speed (Service 01 Mode 01 PID 0D)
        speed_kmh_int = min(255, max(0, int(round(self.speed_mps * 3.6))))
        # Payload: Length (3), Mode (0x41), PID (0x0D), Value (speed_kmh), padding (0xAA)
        obd_payload = bytes([0x03, 0x41, 0x0D, speed_kmh_int, 0xAA, 0xAA, 0xAA, 0xAA])
        messages.append(CanMessage(self.CAN_ID_OBD2_RESPONSE, obd_payload, self.current_time_s))

        # 2. 4-Wheel Speeds (FL, FR, RL, RR in 0.01 km/h)
        speed_raw = int(self.speed_mps * 3.6 * 100)
        # Differentials during cornering
        track_width_m = 1.5
        delta_v = (self.yaw_rate_rads * track_width_m * 0.5) * 3.6 * 100
        fl = max(0, int(speed_raw - delta_v))
        fr = max(0, int(speed_raw + delta_v))
        rl = max(0, int(speed_raw - delta_v))
        rr = max(0, int(speed_raw + delta_v))
        wheel_payload = struct.pack(">HHHH", fl, fr, rl, rr)
        messages.append(CanMessage(self.CAN_ID_WHEEL_SPEEDS, wheel_payload, self.current_time_s))

        # 3. Steering Wheel Angle (scale: 0.1 deg/LSB, signed 16-bit)
        steering_raw = int(self.steering_angle_deg * 10)
        steering_payload = struct.pack(">hH", steering_raw, 0)
        messages.append(CanMessage(self.CAN_ID_STEERING_ANGLE, steering_payload, self.current_time_s))

        return messages

    def parse_obd_speed(self, message: CanMessage) -> Optional[float]:
        """Decode OBD-II vehicle speed in m/s from standard response."""
        if message.arbitration_id != self.CAN_ID_OBD2_RESPONSE:
            return None
        data = message.data
        if len(data) >= 4 and data[1] == 0x41 and data[2] == 0x0D:
            speed_kmh = float(data[3])
            return speed_kmh / 3.6
        return None

    def parse_wheel_speeds(self, message: CanMessage) -> Optional[Tuple[float, float, float, float]]:
        """Decode 4 wheel speeds in m/s from chassis CAN frame."""
        if message.arbitration_id != self.CAN_ID_WHEEL_SPEEDS or len(message.data) < 8:
            return None
        fl, fr, rl, rr = struct.unpack(">HHHH", message.data)
        to_mps = lambda v: (v / 100.0) / 3.6
        return (to_mps(fl), to_mps(fr), to_mps(rl), to_mps(rr))
