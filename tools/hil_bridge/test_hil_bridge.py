import pytest
from tools.hil_bridge.can_simulator import VehicleCanSimulator, CanMessage


def test_can_obd_speed_generation_and_decoding():
    sim = VehicleCanSimulator(dt_s=0.01)
    sim.set_speed(60.0) # 60 km/h

    messages = sim.step()
    assert len(messages) >= 3

    # Find OBD response
    obd_msgs = [m for m in messages if m.arbitration_id == VehicleCanSimulator.CAN_ID_OBD2_RESPONSE]
    assert len(obd_msgs) == 1

    decoded_speed_mps = sim.parse_obd_speed(obd_msgs[0])
    assert decoded_speed_mps is not None
    # 60 km/h = 16.6667 m/s, integer precision is ±0.3 m/s
    assert pytest.approx(decoded_speed_mps, abs=0.4) == (60.0 / 3.6)


def test_can_wheel_speed_differentials_during_turning():
    sim = VehicleCanSimulator(dt_s=0.01)
    sim.set_speed(50.0)
    sim.set_steering(45.0) # Steering to right

    messages = sim.step()
    wheel_msgs = [m for m in messages if m.arbitration_id == VehicleCanSimulator.CAN_ID_WHEEL_SPEEDS]
    assert len(wheel_msgs) == 1

    speeds = sim.parse_wheel_speeds(wheel_msgs[0])
    assert speeds is not None
    fl, fr, rl, rr = speeds

    # Outside wheels should rotate faster than inside wheels
    assert fr > 0
    assert fl > 0
    assert abs(fr - fl) > 0.05
