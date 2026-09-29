# IDR (Intelligent Dead Reckoning) — System Architecture & Mathematical Foundations

## 1. System Overview

**IDR (Intelligent Dead Reckoning with GNSS Fusion)** is an industrial-grade, hybrid physics-neural navigation engine specifically engineered for extreme navigation environments, such as Indian urban canyons, multi-level flyover networks, underpasses, tunnels, and irregular vehicle dynamics (auto-rickshaws, two-wheelers, buses).

```
                      +---------------------------------------+
                      |   Sensors (100 Hz IMU, 1-10 Hz GNSS)  |
                      +---------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
   +--------------------------+                        +--------------------------+
   |   Phone-to-Vehicle       |                        |   Temporal Multi-Task    |
   |   Dynamic Alignment      |                        |   Motion Model (TCN+GRU) |
   +--------------------------+                        +--------------------------+
                |                                                   |
                v                                                   v
   +-------------------------------------------------------------------------------+
   |                      15-State Error-State Kalman Filter                       |
   |   - Strapdown INS Mechanization (Quaternion kinematics @ 100 Hz)              |
   |   - Non-Holonomic Constraints (NHC: Body lateral/vertical zero-velocity)      |
   |   - Adaptive Zero-Velocity Updates (ZUPT / ZARU)                              |
   |   - Innovation Mahalanobis Gate & GNSS Integrity Monitor                      |
   +-------------------------------------------------------------------------------+
                                          |
                                          v
   +-------------------------------------------------------------------------------+
   |               HMM Viterbi Map Matcher with Multi-Layer Graph                  |
   |   - Spatial Grid Indexing (O(1) candidate lookup)                             |
   |   - Grade-separated Flyover/Underpass Layer Disambiguation                     |
   +-------------------------------------------------------------------------------+
                                          |
                                          v
   +-------------------------------------------------------------------------------+
   |            Edge Serial / UDP Stream  <--->  Android JNI Foreground App        |
   +-------------------------------------------------------------------------------+
```

---

## 2. 15-State Error-State Kalman Filter (ESKF)

### 2.1 State Vector Formulation

The continuous error state $\delta \mathbf{x} \in \mathbb{R}^{15}$ is formulated as:
$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n \\ \delta \mathbf{v}^n \\ \delta \boldsymbol{\theta}^n \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix}$$

where:
- $\delta \mathbf{p}^n = [\delta x, \delta y, \delta z]^T$: Position error in the local Navigation frame (ENU, in meters).
- $\delta \mathbf{v}^n = [\delta v_E, \delta v_N, \delta v_U]^T$: Velocity error in the local Navigation frame (in m/s).
- $\delta \boldsymbol{\theta}^n = [\delta \phi, \delta \theta, \delta \psi]^T$: Small-angle orientation error vector (roll, pitch, yaw in radians).
- $\mathbf{b}_a = [b_{ax}, b_{ay}, b_{az}]^T$: Accelerometer bias drift (in $\text{m/s}^2$).
- $\mathbf{b}_g = [b_{gx}, b_{gy}, b_{gz}]^T$: Gyroscope bias drift (in $\text{rad/s}$).

### 2.2 Error-State Dynamics & Transition Matrix

The continuous-time error differential equations are:
$$\delta \dot{\mathbf{p}}^n = \delta \mathbf{v}^n$$
$$\delta \dot{\mathbf{v}}^n = -[\mathbf{f}^n \times] \delta \boldsymbol{\theta}^n - \mathbf{R}_b^n \mathbf{b}_a + \mathbf{w}_v$$
$$\delta \dot{\boldsymbol{\theta}}^n = -\mathbf{R}_b^n \mathbf{b}_g + \mathbf{w}_\theta$$
$$\dot{\mathbf{b}}_a = \mathbf{w}_{ba}, \quad \dot{\mathbf{b}}_g = \mathbf{w}_{bg}$$

where:
- $[\mathbf{f}^n \times]$ is the skew-symmetric matrix of specific force in the navigation frame.
- $\mathbf{R}_b^n$ is the rotation matrix from body to navigation frame parameterized by nominal quaternion $\mathbf{q}$.
- $\mathbf{w}_v, \mathbf{w}_\theta, \mathbf{w}_{ba}, \mathbf{w}_{bg}$ are white Gaussian noise processes.

In discrete time with interval $\Delta t$, the state transition matrix $\mathbf{F} \in \mathbb{R}^{15 \times 15}$ is:
$$\mathbf{F} = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & -[\mathbf{f}^n \times] \Delta t & -\mathbf{R}_b^n \Delta t & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & -\mathbf{R}_b^n \Delta t \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3
\end{bmatrix}$$

### 2.3 Error-to-Nominal Injection and Reset

Following every measurement update $\mathbf{K} (\mathbf{z} - \hat{\mathbf{z}})$, the error state $\delta \mathbf{x}$ is injected into the nominal state:
$$\mathbf{p}^n \leftarrow \mathbf{p}^n + \delta \mathbf{p}^n$$
$$\mathbf{v}^n \leftarrow \mathbf{v}^n + \delta \mathbf{v}^n$$
$$\mathbf{q} \leftarrow \mathbf{q} \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \boldsymbol{\theta}^n \end{bmatrix}, \quad \mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}$$
$$\mathbf{b}_a \leftarrow \mathbf{b}_a + \delta \mathbf{b}_a, \quad \mathbf{b}_g \leftarrow \mathbf{b}_g + \delta \mathbf{b}_g$$

Once injected, the error state is mathematically reset to zero: $\delta \mathbf{x} \leftarrow \mathbf{0}_{15 \times 1}$, and error covariance is updated via the reset Jacobian.

### 2.4 Right-Invariant Extended Kalman Filter (RI-EKF) on $SE_2(3)$

To resolve the classical EKF's **linearization inconsistency** during aggressive maneuvers (e.g., sharp 180° turns in congested alleys) and extended GNSS blackouts, IDR implements an alternative **Right-Invariant EKF** on the matrix Lie group $SE_2(3)$:

#### 2.4.1 Lie Group State & Error
$$\mathbf{X} = \begin{bmatrix} \mathbf{R} & \mathbf{v} & \mathbf{p} \\ \mathbf{0}_{1 \times 3} & 1 & 0 \\ \mathbf{0}_{1 \times 3} & 0 & 1 \end{bmatrix} \in SE_2(3), \quad \boldsymbol{\eta} = \hat{\mathbf{X}} \mathbf{X}^{-1} \in SE_2(3)$$

#### 2.4.2 Trajectory-Independent Error Dynamics
Unlike the standard ESKF where the error transition matrix $\mathbf{F}$ depends on the estimated orientation and specific force $[\mathbf{f}^n \times]$, the Right-Invariant continuous error Jacobian $\mathbf{A} \in \mathbb{R}^{15 \times 15}$ satisfies:
$$\mathbf{A} = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}} \\
[\mathbf{g} \times] & \mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}} & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix}$$
Here, $[\mathbf{g} \times]$ depends **only on the constant gravity vector**, completely removing linearization error from trajectory uncertainty!

#### 2.4.3 Closed-Form Group Action Update
Following measurement correction $\boldsymbol{\xi} \in \mathfrak{se}_2(3)$, the group state is updated via exact Lie group actions:
$$\hat{\mathbf{R}} \leftarrow \exp([\boldsymbol{\xi}_\theta \times]) \hat{\mathbf{R}}$$
$$\hat{\mathbf{v}} \leftarrow \exp([\boldsymbol{\xi}_\theta \times]) \hat{\mathbf{v}} + \boldsymbol{\xi}_v$$
$$\hat{\mathbf{p}} \leftarrow \exp([\boldsymbol{\xi}_\theta \times]) \hat{\mathbf{p}} + \boldsymbol{\xi}_p$$

---


## 3. Physical Constraints & Vehicle Dynamics

### 3.1 Non-Holonomic Constraints (NHC)
For land vehicles (cars, buses, auto-rickshaws, two-wheelers) under standard traction conditions, lateral slip and vertical bounce are close to zero in the vehicle frame $V$:
$$v_y^V \approx 0, \quad v_z^V \approx 0$$

Given vehicle-to-navigation rotation $\mathbf{R}_V^n$, the measurement model is:
$$\mathbf{z}_{\text{nhc}} = \begin{bmatrix} (\mathbf{R}_V^n)_{2, :}^T \mathbf{v}^n \\ (\mathbf{R}_V^n)_{3, :}^T \mathbf{v}^n \end{bmatrix} + \mathbf{v}_{\text{nhc}}$$

The measurement matrix $\mathbf{H}_{\text{nhc}} \in \mathbb{R}^{2 \times 15}$ maps directly to velocity and orientation errors.

### 3.2 Vehicle Type Adaptations

| Vehicle Type | NHC Lateral $\sigma_y$ | NHC Vertical $\sigma_z$ | Max Yaw Rate | Notes |
|:---|:---|:---|:---|:---|
| **Motorcycle / Scooter** | $0.45\text{ m/s}$ | $0.50\text{ m/s}$ | $2.5\text{ rad/s}$ | High lean angles; relaxed NHC during turns |
| **Auto-Rickshaw (3W)** | $0.25\text{ m/s}$ | $0.35\text{ m/s}$ | $1.8\text{ rad/s}$ | Sharp turning radius, high chassis vibration |
| **Standard Car** | $0.10\text{ m/s}$ | $0.10\text{ m/s}$ | $1.0\text{ rad/s}$ | Strict lateral constraint |
| **Heavy Bus / Truck** | $0.05\text{ m/s}$ | $0.05\text{ m/s}$ | $0.6\text{ rad/s}$ | High inertia, low roll rate |

### 3.3 Dynamic Phone-to-Vehicle Auto-Alignment

Smartphones mounted on dashboards, windshields, or handlebar holders have an arbitrary rotation $\mathbf{R}_P^V$ relative to the vehicle body:
1. **Coarse Leveling (Gravity Vector Estimation):** Low-pass filtering stationary accelerometer readings yields the gravity unit vector $\mathbf{g}_P$. The pitch and roll angles are resolved to align phone $Z$ with true vertical.
2. **Azimuth Alignment (Forward Acceleration Integration):** When the vehicle accelerates or brakes during GNSS tracking, forward acceleration $\mathbf{a}_{\text{fwd}}^V = [a_x^V, 0, 0]^T$ correlates with GNSS velocity derivative $\dot{\mathbf{v}}_{\text{gnss}}$. Cross-correlation resolves the remaining yaw rotation angle between the leveled phone frame and the vehicle longitudinal axis.

### 3.4 Multi-Sensor Aiding & Odometry Models

#### 3.4.1 Barometric Altimetry Aiding
Barometric pressure $P$ is converted to relative ENU altitude using the hypsometric approximation:
$$h_{\text{baro}} = 44330.0 \cdot \left(1 - \left(\frac{P}{P_0}\right)^{0.190284}\right)$$
The observation model $z_{\text{baro}} = p_z^n + v_{\text{baro}}$ is gated through a 1-DOF $\chi^2$ innovation test, anchoring the vertical state during tunnel transits and resolving flyover deck transitions.

#### 3.4.2 Wheel Speed Odometry (CAN / OBD-II)
Wheel speed sensors report longitudinal vehicle speed $v_{\text{wheel}} = v_x^V$. The observation model links body forward velocity with navigation velocity:
$$z_{\text{odo}} = (\mathbf{R}_V^n)_{1, :}^T \mathbf{v}^n - v_{\text{wheel}}$$
The measurement Jacobian $\mathbf{H}_{\text{odo}} \in \mathbb{R}^{1 \times 15}$ observes both velocity and attitude errors, eliminating dead reckoning velocity drift during prolonged satellite outages.

#### 3.4.3 Calibrated Magnetometer Heading
Magnetic field vectors are leveled into the vehicle horizontal plane $\mathbf{b}_V = (\mathbf{R}_P^V)^T \mathbf{b}_P$, yielding magnetic heading $\psi_{\text{mag}} = \text{atan2}(-b_y, b_x)$. High-magnitude anomaly rejection ($\|b\| \notin [20, 75]\text{ \mu T}$) filters out steel bridges and railway line interference.

---


## 4. Multi-Task Deep Neural Motion Model

### 4.1 Architecture
The neural branch processes 1-second sliding windows (100 samples) of calibrated 6-DOF IMU data:
- **TCN Backbone:** 3 residual dilated convolution blocks (kernel size 3, dilation factors 1, 2, 4) with 64 channels, spatial dropout, and layer normalization.
- **Recurrent Bridge:** 2-layer Bidirectional GRU (hidden size 64) extracting temporal context.
- **Multi-Task Heads:**
  1. `speed_ms`: Scalar forward speed estimate.
  2. `heading_rad`: Vehicle heading angle.
  3. `step_displacement_m`: High-resolution incremental forward distance $\Delta d$.
  4. `mode_logits`: 4-class classification (`STATIONARY`, `DRIVING`, `WALKING`, `IRREGULAR_ROAD`).
  5. `log_var`: Heteroscedastic uncertainty estimate $\log(\sigma^2)$.

### 4.2 Heteroscedastic Gaussian Loss
To avoid overconfidence during erratic movements (e.g. pothole hits or speed bumps), heads are trained with Gaussian Negative Log-Likelihood:
$$\mathcal{L}(y, \hat{y}, s) = \frac{1}{2} \exp(-s) \|y - \hat{y}\|^2 + \frac{1}{2} s$$
where $s = \log(\sigma^2)$. If the network cannot predict displacement accurately, it learns to increase uncertainty $s$, scaling up measurement covariance $\mathbf{R}$ when feeding the EKF.

---

## 5. Map-Matching: Hidden Markov Model (HMM) Viterbi

### 5.1 Spatial Polyline Indexing
Road networks are stored in a regular 2D spatial grid (default cell size: 50m). Roads are discretized along segments at intervals $\le \text{cell\_size} / 2$, allowing $O(1)$ candidate retrieval for any arbitrary ENU query coordinate.

### 5.2 Emission Probability
The emission probability of dead-reckoned position $\mathbf{x}_t$ given candidate road $r_i$ considers Euclidean distance, heading alignment, and grade separation:
$$P(\mathbf{x}_t \mid r_i) = \frac{1}{\sqrt{2\pi}\sigma_d} \exp\left(-\frac{d(\mathbf{x}_t, r_i)^2}{2\sigma_d^2}\right) \cdot \cos^k(\psi_t - \psi_{r_i}) \cdot \exp\left(-\frac{|\Delta z|^2}{2\sigma_z^2}\right)$$

### 5.3 Multi-Level Elevation Disambiguation
Indian grade-separated road corridors (e.g., Delhi Barapullah Elevated Corridor vs. Ring Road Surface vs. AIIMS Underpass) share overlapping 2D coordinates. IDR tracks barometric altitude and GNSS ellipsoidal height to apply an exponential penalty to candidate segments with mismatched `layer` attributes (-1, 0, +1).
