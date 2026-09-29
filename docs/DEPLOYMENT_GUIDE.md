# IDR (Intelligent Dead Reckoning) — Production Deployment Guide

## 1. Android Mobile Deployment

### 1.1 Prerequisites
- Android Studio Ladybug / Hedgehog or newer
- Android SDK 34 (Android 14) with NDK 26.1+
- CMake 3.22.1+

### 1.2 Android Native Shared Library Build
The native C++ engine compiles into `libidr-native.so` for Android ABIs (`arm64-v8a`, `armeabi-v7a`, `x86_64`):

```bash
cd android
./gradlew assembleRelease
```
The resulting AAR/APK bundles:
- `libidr-native.so` (optimized with `-O3 -ffast-math`)
- `libonnxruntime.so` (for neural motion inference)
- High-rate background sensor service: `IdrNavigationService`
- Bilingual DPDP consent dialogs (English and Hindi)

### 1.3 Android Permissions & Power Optimizations
Ensure your `AndroidManifest.xml` grants:
- `android.permission.HIGH_SAMPLING_RATE_SENSORS`: Allows unthrottled 100 Hz IMU collection.
- `android.permission.FOREGROUND_SERVICE_LOCATION`: Prevents OS battery killers from terminating dead-reckoning during long tunnel transits.
- `android.permission.ACCESS_FINE_LOCATION`: For GNSS reference fixes.

---

## 2. Embedded & Edge Device Deployment

### 2.1 Microcontroller Target (ESP32 / STM32 / RTOS)
For telematics units running FreeRTOS or bare-metal C++:
1. Configure UART at **115200 or 921600 baud**.
2. Stream 40-byte binary packets at 100 Hz.
3. Verify byte alignment: Ensure compiler flag `-fpack-struct` or `#pragma pack(push, 1)` matches `idr::edge::BinaryPacket`.
4. Validate CRC16-CCITT checksum on each received frame.

### 2.2 Linux Edge Companion (Raspberry Pi CM4 / Jetson Orin Nano)
Compile standalone engine with edge backend:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SIMULATION=ON
cmake --build . -j$(nproc)
```

Run replay or real-time serial listener:
```bash
./simulation/idr-replay --input /dev/ttyUSB0 --output replay_out.csv --vehicle-type auto_rickshaw
```

---

## 3. Cloud Backend & Monitoring Deployment

### 3.1 Docker Compose Deployment
The IDR cloud telemetry and model distribution stack runs with Docker Compose:

```bash
cd cloud/docker
docker-compose up -d --build
```

Verify running containers:
```bash
docker-compose ps
```

Endpoints exposed:
- `http://localhost:8000/health`: REST API health probe
- `http://localhost:8000/docs`: Interactive OpenAPI / Swagger UI
- `http://localhost:9090`: Prometheus Monitoring Dashboard

### 3.2 Prometheus Metric Ingestion
Prometheus polls `/api/v1/analytics/summary` every 10 seconds:
- `total_sessions`: Total driving trips recorded
- `active_dpdp_consents`: Number of pseudonymized devices with legal consent
- `total_distance_km`: Cumulative distance driven under dead-reckoning
- `average_gnss_outage_seconds`: Duration of satellite-denied navigation handled

---

## 4. DPDP Act 2023 Compliance Operations

### 4.1 Zero-PII Policy
- **No Identity Storage:** Device MAC addresses, IMEI, phone numbers, or user names are never accepted or stored.
- **Client-Side Pseudonymization:** Device IDs are hashed on-device using SHA-256 before transmission.

### 4.2 Handling Right to Erasure Requests
When a user withdraws consent or requests erasure through the mobile UI:
1. Mobile app sends `POST /api/v1/dpdp/erase` with the client token.
2. The server marks the consent record as revoked.
3. All telemetry drive sessions matching the token are permanently purged from the database.
4. An immutable cryptographically hashed audit log entry records the erasure event.
