# IDR (Intelligent Dead Reckoning) — API Reference Manual

## 1. Edge / Serial Binary Protocol (40-Byte Packed Struct)

For embedded microcontrollers (ESP32, STM32), companion telematics boxes, and high-rate local UDP streaming, IDR defines a 40-byte packed binary packet with big-endian wire format:

```
+---------------+---------------+--------------------+--------------------+
| Magic: 'I','D'| SeqNum: uint16| TimestampUs: uint64| LatE7: int32       |
| (0-1: 2 B)    | (2-3: 2 B)    | (4-11: 8 B)        | (12-15: 4 B)       |
+---------------+---------------+--------------------+--------------------+
| LonE7: int32  | AltCm: int32  | SpeedCmS: uint16   | HeadingCdeg: uint16|
| (16-19: 4 B)  | (20-23: 4 B)  | (24-25: 2 B)       | (26-27: 2 B)       |
+---------------+---------------+--------------------+--------------------+
| AccMms: uint16| NavMode: uint8| MatchedRoad: uint32| CRC16: uint16      |
| (28-29: 2 B)  | (30: 1 B)     | (31-34: 4 B)       | (35-36: 2 B)       |
+---------------+---------------+--------------------+--------------------+
```

### Struct Field Definitions

| Field | Type | Description |
|:---|:---|:---|
| `magic` | `uint8_t[2]` | Always `{'I', 'D'}` (0x49, 0x44) |
| `sequence_number` | `uint16_t` | Monotonically increasing packet counter |
| `timestamp_us` | `uint64_t` | Epoch timestamp in microseconds |
| `latitude_e7` | `int32_t` | WGS84 Latitude scaled by $10^7$ (e.g. $285832000 \rightarrow 28.5832^\circ$) |
| `longitude_e7`| `int32_t` | WGS84 Longitude scaled by $10^7$ (e.g. $772185000 \rightarrow 77.2185^\circ$) |
| `altitude_cm` | `int32_t` | Ellipsoidal / MSL height in centimeters |
| `speed_cms` | `uint16_t` | Ground speed in centimeters per second ($100\text{ cm/s} = 1.0\text{ m/s}$) |
| `heading_cdeg`| `uint16_t` | Heading angle in centi-degrees ($0 \text{ to } 35999 \rightarrow 0^\circ \text{ to } 359.99^\circ$) |
| `accuracy_mms`| `uint16_t` | Horizontal $1\sigma$ position uncertainty in millimeters |
| `nav_mode` | `uint8_t` | Bitmask: `0x00`=Init, `0x01`=GNSS_Fix, `0x02`=DR_Active, `0x04`=Map_Matched |
| `matched_road_id` | `uint32_t` | OSM Way ID or local road graph unique segment identifier |
| `crc16` | `uint16_t` | CCITT-FALSE CRC over the preceding bytes |

---

## 2. Android Native JNI Bridge (`IdrNativeBridge`)

The JNI layer binds Kotlin code to `libidr-native.so` (or `libidr-native.dll` in local testing):

### Methods

#### `nativeCreateEngine(vehicleProfile: Int): Long`
Instantiates a new C++ `idr::IdrEngine` configured for the specified vehicle type:
- `0`: Standard Car / Taxi
- `1`: Motorcycle / Two-Wheeler
- `2`: Auto-Rickshaw (3-Wheeler)
- `3`: Bus / Heavy Commercial Vehicle
- Returns: Pointer address to the native engine instance (64-bit handle).

#### `nativeDestroyEngine(handle: Long): Unit`
Safely destroys the engine and frees native memory.

#### `nativeProcessImu(handle: Long, timestampNs: Long, ax: Double, ay: Double, az: Double, gx: Double, gy: Double, gz: Double): Unit`
Feeds calibrated 100 Hz accelerometer ($\text{m/s}^2$) and gyroscope ($\text{rad/s}$) measurements into the strapdown mechanization and auto-alignment filter.

#### `nativeProcessGnss(handle: Long, timestampNs: Long, lat: Double, lon: Double, alt: Double, speed: Double, heading: Double, hAcc: Double, vAcc: Double): Unit`
Injects GNSS fixes into the ESKF measurement queue.

#### `nativeGetNavigationState(handle: Long): NavigationState`
Returns the current fused state snapshot:
```kotlin
data class NavigationState(
    val timestampNs: Long,
    val latitude: Double,
    val longitude: Double,
    val altitude: Double,
    val speedKmh: Double,
    val headingDeg: Double,
    val horizontalAccuracyMeters: Double,
    val isDeadReckoningActive: Boolean,
    val isZeroVelocityDetected: Boolean,
    val alignmentQuality: Int, // 0: None, 1: Coarse, 2: Fine
    val matchedRoadId: Long,
    val matchedRoadName: String
)
```

#### `nativeLoadOsmGeoJson(handle: Long, geoJsonPath: String, refLat: Double, refLon: Double, refAlt: Double): Int`
Loads a road network file into the native spatial index. Returns the number of imported road segments.

---

## 3. Cloud REST Telemetry API

Base URL: `http://<host>:8000`

### 3.1 Health Check
- **Endpoint:** `GET /health`
- **Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "idr-telemetry-service",
  "version": "1.0.0"
}
```

### 3.2 DPDP Consent Grant
- **Endpoint:** `POST /api/v1/dpdp/consent`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "device_pseudonym": "dev-bhavsar-pixel8",
  "consent_version": "v1.0",
  "data_processing_granted": true,
  "analytics_granted": true,
  "model_improvement_granted": true
}
```
- **Response:** `200 OK`
```json
{
  "status": "recorded",
  "device_pseudonym": "dev-bhavsar-pixel8",
  "active": true
}
```

### 3.3 Telemetry Ingestion (DPDP Gated)
- **Endpoint:** `POST /api/v1/telemetry`
- **Description:** Strictly rejects with HTTP 403 if the pseudonym lacks an active DPDP consent record.
- **Request Body:**
```json
{
  "device_pseudonym": "dev-bhavsar-pixel8",
  "session_id": "session-delhi-001",
  "vehicle_type": "auto_rickshaw",
  "start_timestamp": 1759132800000,
  "end_timestamp": 1759133400000,
  "total_distance_m": 4500.5,
  "gnss_outage_seconds": 180.0,
  "max_dr_drift_m": 4.2,
  "samples_count": 60000,
  "consent_hash": "consent-token-sha256"
}
```
- **Response:** `201 Created`
```json
{
  "status": "ingested",
  "session_id": "session-delhi-001",
  "recorded_at": "2026-09-29T10:00:00Z"
}
```

### 3.4 DPDP Right to Erasure
- **Endpoint:** `POST /api/v1/dpdp/erase`
- **Request Body:**
```json
{
  "device_pseudonym": "dev-bhavsar-pixel8"
}
```
- **Response:** `200 OK`
```json
{
  "status": "erased",
  "device_pseudonym": "dev-bhavsar-pixel8",
  "sessions_deleted": 1
}
```

### 3.5 Model Release Distribution
- **Endpoint:** `GET /api/v1/models/latest?vehicle_type=car`
- **Response:** `200 OK`
```json
{
  "version": "1.0.0",
  "model_name": "idr_motion_model.onnx",
  "sha256": "4a7f...",
  "signature_ed25519": "c9a0...",
  "download_url": "https://models.idr.internal/v1.0.0/model.onnx"
}
```

---

## 4. Offline Vector Tile Binary Format (`.idrtile`)

For zero-latency, internet-independent map rendering and dead-reckoning snapping, IDR packages regional road networks into `.idrtile` binary packages:

### Header
- `magic`: `b"IDRT"` (4 bytes)
- `version`: `uint16` (2 bytes, current = 1)
- `road_count`: `uint32` (4 bytes)
- `bounding_box`: `min_lat, min_lon, max_lat, max_lon` (4 × `float64` = 32 bytes)

### Road Records (Repeated `road_count` times)
- `road_id`: `uint32` (4 bytes)
- `layer`: `int8` (1 byte, `-1`: underpass, `0`: surface, `1`: elevated flyover)
- `one_way`: `uint8` (1 byte, boolean flag)
- `speed_limit_x10`: `uint16` (2 bytes, speed limit in km/h × 10)
- `name_length`: `uint16` (2 bytes)
- `name_utf8`: Variable string bytes
- `coords_count`: `uint16` (2 bytes)
- `coords_array`: Array of (`lat_float32`, `lon_float32`, `alt_float32`) (12 bytes per coordinate point)

---

## 5. Hardware-in-the-Loop CAN Telemetry (`can_simulator.py`)

Simulates standard automotive CAN network frames for vehicle test benches:

| CAN Identifier | Description | DLC | Cycle Time |
|:---|:---|:---:|:---:|
| `0x7E8` | OBD-II Mode 01 PID 0x0D Vehicle Speed ($0-255\text{ km/h}$) | 8 B | 100 ms |
| `0x0B4` | 4-Wheel Pulse Speed Sensors (FL, FR, RL, RR in $0.01\text{ km/h}$) | 8 B | 10 ms |
| `0x025` | Steering Wheel Angle Sensor ($\pm 780.0^\circ$ at $0.1^\circ/\text{LSB}$) | 4 B | 10 ms |
| `0x120` | Chassis Stability Yaw Rate ($\text{rad/s}$) and Lateral Accel ($g$) | 8 B | 10 ms |

