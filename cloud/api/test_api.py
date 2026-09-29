"""Unit tests for IDR Cloud Telemetry & DPDP Compliance API."""

from fastapi.testclient import TestClient
from cloud.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dpdp_compliant"] is True


def test_telemetry_rejected_without_dpdp_consent():
    payload = {
        "session_id": "sess_test_123",
        "device_pseudonym": "dev_unconsented_999",
        "vehicle_type": "car",
        "duration_sec": 120.0,
        "distance_km": 2.5,
        "avg_speed_kmh": 45.0,
        "outage_count": 2,
        "max_outage_sec": 15.0,
        "pct_dead_reckoning": 25.0,
    }
    response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 403
    assert "DPDP Act" in response.json()["detail"]


def test_consent_grant_and_telemetry_flow():
    pseudonym = "dev_consented_456"

    # 1. Grant consent
    consent_req = {
        "device_pseudonym": pseudonym,
        "consent_version": 1,
        "purpose_navigation": True,
        "purpose_logging": True,
        "purpose_cloud_upload": True,
        "action": "GRANT",
    }
    c_resp = client.post("/api/v1/dpdp/consent", json=consent_req)
    assert c_resp.status_code == 200
    assert c_resp.json()["status"] == "consent_recorded"

    # 2. Ingest telemetry
    payload = {
        "session_id": "sess_consented_001",
        "device_pseudonym": pseudonym,
        "vehicle_type": "auto_rickshaw",
        "duration_sec": 300.0,
        "distance_km": 4.2,
        "avg_speed_kmh": 28.0,
        "outage_count": 3,
        "max_outage_sec": 22.0,
        "pct_dead_reckoning": 35.0,
    }
    t_resp = client.post("/api/v1/telemetry", json=payload)
    assert t_resp.status_code == 201
    assert t_resp.json()["status"] == "ingested"

    # 3. Check summary
    s_resp = client.get("/api/v1/analytics/summary")
    assert s_resp.status_code == 200
    assert s_resp.json()["total_sessions"] >= 1

    # 4. Right to Erasure
    e_resp = client.post("/api/v1/dpdp/erase", json={"device_pseudonym": pseudonym})
    assert e_resp.status_code == 200
    assert e_resp.json()["status"] == "data_erased"


def test_latest_model_release():
    response = client.get("/api/v1/models/latest")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "sha256" in data
    assert "signature" in data
