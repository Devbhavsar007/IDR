# IDR — Intelligent Dead Reckoning with GNSS Fusion

> Real-time vehicle localization engine maintaining continuous position during GNSS degradation/outage using smartphone IMU + AI + classical navigation fusion.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IDR Engine (C++)                      │
│                                                         │
│  Sensors ──► INS Mechanization ──► Error-State EKF      │
│  (IMU/GNSS/  (Strapdown)          (15-state fusion)     │
│   Baro/Mag)                                             │
│              AI Models ──────────► EKF Measurements      │
│              (ONNX Runtime)        (velocity, Δpose)     │
│                                                         │
│              Constraints ─────────► EKF Measurements     │
│              (NHC, ZUPT, Map)      (pseudo-obs)          │
│                                                         │
│  Output: NavigationState @ 10-200 Hz                    │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────┐
    │  Android App        │  │  Edge Binary   │
    │  (Kotlin/Compose)   │  │  (Linux ARM64) │
    │  JNI Bridge         │  │  Serial/UDP    │
    │  MapLibre Offline   │  │  200 Hz IMU    │
    └─────────────────────┘  └────────────────┘
```

## Project Structure

```
idr/
├── core/                    # C++ core library
│   ├── include/idr/         # Public headers
│   │   ├── types/           # Common types, frames, constants
│   │   ├── sensors/         # Sensor data structures
│   │   ├── timing/          # Timestamp validation
│   │   ├── math_utils/      # Geo coords, quaternions, matrices
│   │   ├── inertial/        # Strapdown INS mechanization
│   │   ├── fusion/          # Error-State EKF
│   │   ├── constraints/     # NHC, ZUPT, vehicle profiles
│   │   ├── integrity/       # GNSS integrity monitor
│   │   ├── neural/          # ONNX inference bridge
│   │   ├── calibration/     # Sensor calibration
│   │   ├── alignment/       # Phone-to-vehicle alignment
│   │   ├── map_graph/       # Road graph + spatial index
│   │   ├── map_matching/    # HMM map matching
│   │   ├── engine/          # IdrEngine, NavigationState
│   │   └── diagnostics/     # Logging, profiling
│   └── src/                 # Implementation files
├── ml/                      # Python ML pipeline
│   ├── models/              # Model architectures (PyTorch)
│   ├── training/            # Training loops
│   ├── evaluation/          # Metrics, benchmarks
│   ├── export/              # ONNX export
│   └── tests/               # ML unit tests
├── android/idr-mobile/      # Android application
├── edge/                    # Edge deployment
├── simulation/              # Replay engine
├── tests/                   # C++ tests (GoogleTest)
├── maps/                    # Map schemas
├── tools/                   # CLI utilities
└── docs/                    # Documentation
```

## Building

### Prerequisites
- CMake ≥ 3.20
- C++17 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- vcpkg for dependency management

### Build
```bash
# Install dependencies
vcpkg install

# Configure
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build -j$(nproc)

# Test
cd build && ctest --output-on-failure
```

### Python ML Environment
```bash
cd ml
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,mlops]"
```

## Key Design Principles

1. **AI produces measurements, not positions** — The AI model outputs velocity, displacement, and uncertainty. The EKF is always the final authority.
2. **Graceful degradation** — Every sensor and constraint can fail independently without crashing the system.
3. **Continuous operation** — No hard resets; smooth transitions between GNSS+INS and pure dead reckoning.
4. **Phone-agnostic** — Works across budget to flagship Android devices with adaptive quality.

## License

Proprietary — All rights reserved.
