# CIE Exercise AI I: Data Preprocessing for Biomechanical Sensor Data

## 1. Laboratory Overview and Engineering Context

The exercise **CIE Exercise AI I: Data Preprocessing** from the *Institut für Allgemeine Mechanik (IAM)* at RWTH Aachen University applies the foundational data acquisition and preprocessing principles established in **Theory of AI I** to practical engineering analysis. 

The practical pipeline centers on simulated acceleration time-series data captured from an **Inertial Measurement Unit (IMU)** during human gait analysis (`imu_walk.csv`). The workflow spans five progressive engineering tasks:
1. Handling missing and irregular sensor measurements through physically consistent imputation.
2. Filtering uninformative and redundant variables through domain-driven feature selection.
3. Feature normalization via standardization ($z$-score) to condition data for machine learning algorithms.
4. Tri-axial signal magnitude synthesis and digital frequency-domain filtering (low-pass, high-pass, band-pass).
5. Signal envelope extraction and adaptive gait peak identification for event segmentation.

---

## 2. Task 1: Missing Value Imputation in Biomechanical Time Series

### Problem Statement
The dataset `imu_walk.csv` contains simulated gait acceleration records with missing observations ($NaN$ values) caused by typical sensor artifacts such as wireless packet dropouts, intermittent power dips, or hardware read latencies.

### Imputation Methods and Comparative Trade-Offs

| Imputation Strategy | Algorithmic Mechanism | Viability in Gait IMU Time Series | Engineering Justification & Limitations |
| :--- | :--- | :--- | :--- |
| **Row Deletion (Listwise Deletion)** | Drops any timestamp row containing missing entries | **Unsuitable** | Breaks fixed sampling rate intervals ($\Delta t$), destroys temporal continuity, and distorts downstream frequency filtering and peak detection. |
| **Global Column Mean Imputation** | Replaces missing entries with the overall feature average | **Unsuitable** | Induces sharp, unphysical discontinuities and step shifts in oscillatory gait cycles, artificially suppressing local variance and dynamic range. |
| **Local Temporal Interpolation** (Linear / Spline / Neighbor Averaging) | Estimates missing points from adjacent temporal samples: $\hat{x}_t \approx \frac{x_{t-1} + x_{t+1}}{2}$ | **Optimal** | Preserves time-series continuity, maintains fixed sampling frequency, and accurately tracks smooth biomechanical movement trajectories across brief dropouts. |

---

## 3. Task 2: Feature Selection and Redundancy Reduction

### Problem Statement
The raw dataset includes seven distinct recorded variables:
* Time step: $t$
* Tri-axial accelerations: $a_x, a_y, a_z$
* Auxiliary telemetry: remaining battery voltage $\text{battery\_v}$
* Redundant measurement: x-axis acceleration from a secondary sensor $a_{x,\text{additional}}$
* Derived measurement: mean acceleration across x and y axes $\text{mean\_ax\_ay}$

### Preprocessing and Dimensionality Reduction Strategy
To optimize data quality and eliminate detrimental collinearity prior to machine learning model ingestion, the variables are categorized and filtered:

```
Raw Telemetry ──> Identify Invariant / Irrelevant Variables (battery_v) ──> Remove
              ──> Identify Redundant Collinear Features (ax_additional, mean_ax_ay) ──> Prune
              ──> Retain Independent Physical State Vector (t, ax, ay, az)
```

1. **Elimination of Low-Variance / Irrelevant Telemetry (`battery_v`):**
   * Battery voltage exhibits near-zero short-term variance across a typical gait trial and bears no causal physical relationship to limb kinematic patterns.
   * Retaining it adds irrelevant noise dimensions and risks spurious statistical correlations.
2. **Pruning Collinear and Derived Redundancies (`ax_additional`, `mean_ax_ay`):**
   * The secondary measurement $a_{x,\text{additional}}$ has near-perfect collinearity ($\rho \approx 1.0$) with $a_x$, while $\text{mean\_ax\_ay}$ is a linear combination of existing primary features.
   * Redundant features inflate model parameter counts, degrade numerical stability during matrix inversion in linear models, and skew distance-based learning metrics.
3. **Retained Physical State Representation:**
   * Preserving the primary orthogonal kinematic channels ($t, a_x, a_y, a_z$) maintains full biomechanical fidelity while minimizing feature dimensionality.

---

## 4. Task 3: Feature Standardization and Distribution Analysis

### Problem Statement
Raw sensor channels exhibit widely disparate physical scales and gravitational offsets (for example, dynamic vertical acceleration $a_z$ oscillates around the baseline gravity component $\approx 9.81\,\text{m/s}^2$, whereas horizontal axes $a_x, a_y$ oscillate around $0\,\text{m/s}^2$).

### Mathematical Formulation
Features are standardized using `StandardScaler` from `sklearn.preprocessing`, transforming each variable to zero mean and unit variance ($z$-score):

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ denotes the empirical mean and $\sigma$ denotes the sample standard deviation.

### Distribution and Time-Series Dynamics
* **Horizontal Axes ($a_x, a_y$):** Raw signals centered near $0\,\text{m/s}^2$ with standard deviations typically below unity are scaled up to standard normal variance $[-2, +2]$, expanding feature resolution.
* **Vertical Axis ($a_z$):** Centers the raw distribution from $\approx 10\,\text{m/s}^2$ down to $0.0$, stripping static gravitational bias and aligning vertical movement amplitude with lateral and forward acceleration dynamics.
* **Algorithm Conditioning:** Prevents neural network optimizers and gradient descent steps from disproportionately prioritizing axes with large baseline offsets.

---

## 5. Task 4: Acceleration Magnitude and Frequency-Domain Filtering

### Problem Statement
Evaluating gait dynamics independent of precise IMU coordinate frame orientation requires calculating the invariant acceleration magnitude, followed by frequency filtering to isolate distinct movement bands.

### Magnitude Synthesis
The invariant Euclidean acceleration magnitude $a_{\text{total}}$ is synthesized across orthogonal axes:

$$a_{\text{total}} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

### Digital Filtering Regimes and Threshold Behavior

| Filter Type | Target Frequency Band | Signal Impact & Engineering Application |
| :--- | :--- | :--- |
| **Low-Pass Filter** | Retains low frequencies below cutoff ($f < f_c$) | Removes high-frequency electrical sensor noise, chassis vibration, and skin-artifact jitter; preserves fundamental gait periodicity. |
| **High-Pass Filter** | Retains high frequencies above cutoff ($f > f_c$) | Eliminates baseline gravitational drift, sensor bias shifts, and steady-state posture offsets; isolates rapid impact events. |
| **Band-Pass Filter** | Retains targeted intermediate band ($f_{\text{low}} < f < f_{\text{high}}$) | Strips both low-frequency drift and high-frequency noise simultaneously, isolating rhythmic cyclic movement signatures. |

---

## 6. Task 5: Signal Envelope Extraction and Adaptive Peak Detection

### Problem Statement
Extracting step cadence and stride timing requires isolating the amplitude envelope of gait impacts and reliably identifying heel-strike peaks amidst background noise.

### Analytical Procedure
1. **Band-Pass Pre-filtering:** A band-pass filter between $0.5\,\text{Hz}$ and $6.0\,\text{Hz}$ is applied to $a_{\text{total}}$ to isolate typical human gait frequencies ($1$–$2\,\text{Hz}$ fundamental stride frequency plus primary harmonics).
2. **Envelope Computation:** The signal envelope is computed by taking the absolute value (full-wave rectification) of the band-pass filtered magnitude:

$$\text{envelope} = \left| a_{\text{total, filtered}} \right|$$

3. **Adaptive Threshold Peak Detection:**
   * Fixed hard-coded thresholds fail when subject walking speed, ground stiffness, or body mass vary across trials.
   * An adaptive threshold $T_{\text{adaptive}}$ is constructed using robust statistical estimators:

$$T_{\text{adaptive}} = \text{median}(\text{envelope}) + k \cdot \text{MAD}$$

where $\text{MAD} = \text{median}(|\text{envelope} - \text{median}(\text{envelope})|)$ is the Median Absolute Deviation, and $k$ is a tuning sensitivity multiplier.

### Engineering Gait Interpretation
* **Peak Significance:** Detected prominent envelope peaks correspond directly to **initial foot-ground impact events (heel strikes)** in the human gait cycle.
* **Temporal Kinematics:** Inter-peak intervals provide quantitative metrics of **step duration, stride-to-stride cadence, and bilateral gait symmetry**.
* **Detection Factors:** Peak detection accuracy is governed by the sensitivity multiplier $k$, minimum inter-peak distance constraints (enforcing a physiological refractory period to prevent double-counting split peaks), and sensor placement firmness on the anatomical segment.
