# Theory of AI II: Neural Networks, Supervised Learning, and Training Strategies

## 1. Introduction and Historical Evolution of Neural Architectures

The progression of computational intelligence from biological metaphors into mathematical learning engines spans several technological cycles. Neural networks model functional relationships directly from data rather than relying on hand-crafted analytical formulations, motivated by modern computational capacities including graphics processing units (GPUs) and tensor processing units (TPUs) alongside high-throughput digital acquisition streams.

The evolution of artificial neural networks proceeded through several defining phases:
* **The Biological Model (1943):** Warren McCulloch and Walter Pitts formalized nervous activity as a logical computing model, establishing several governing physical assumptions:
  * Neuronal activity is strictly an "all-or-none" binary process.
  * A fixed number of active excitatory synapses is required to excite a neuron, independent of position or past activity.
  * Synaptic delay represents the only significant transmission latency.
  * Any inhibitory synapse completely prevents excitation of the neuron at any time.
  * Network topology remains invariant over time.
  * Functional behavior is determined entirely by synaptic connectivity.
* **Rosenblatt's Perceptron (1958):** Frank Rosenblatt developed the first computational classifier capable of iteratively updating connection strengths to categorize inputs.
* **Critique on Rosenblatt's Perceptron (1969):** Marvin Minsky and Seymour Papert published a mathematical critique proving the structural impossibility of single-layer perceptrons solving non-linearly separable problems (such as the XOR function), ushering in the first "AI Winter."
* **Multilayer Networks and Backpropagation (1986):** David Rumelhart, Geoffrey Hinton, and Ronald Williams popularized the generalized delta rule and backpropagation algorithm using continuous, differentiable activation functions, resolving single-layer limitations.
* **Recurrent Architectures (1997):** Recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) cells emerged to handle temporal sequence dependencies.
* **The Modern Deep Learning Era (2012–Present):** Accelerated by GPU parallelization, breakthroughs such as AlexNet (2012), the Transformer attention mechanism (2017), and large language models (ChatGPT, 2022) established deep neural networks as foundational engineering tools.

---

## 2. The Perceptron Binary Classifier and Linear Separability

The single-layer perceptron represents the elemental building block of artificial neural networks, functioning geometrically as an affine hyperplane classifier.

### Mathematical Formulation
The neuron processes an input vector $\mathbf{p} = [p_1, p_2, \dots, p_n]^T \in \mathbb{R}^n$ via a linear dot product with an internal weight vector $\mathbf{w} \in \mathbb{R}^n$, shifted by a scalar bias $b \in \mathbb{R}$:

$$v = \mathbf{w}^T \mathbf{p} + b = \sum_{i=1}^n w_i p_i + b$$

The scalar potential $v$ passes through a nonlinear hard-limiter activation function $\phi(v)$ to yield the discrete output classification $\vartheta \in \{0, 1\}$:

$$f(\mathbf{p}) = \begin{cases} 1 & \text{if } \mathbf{w}^T \mathbf{p} + b > 0 \\ 0 & \text{otherwise} \end{cases}$$

Geometrically, the equation $\mathbf{w}^T \mathbf{p} + b = 0$ defines a decision hyperplane dividing the input feature space into two half-spaces corresponding to classes $K_1$ and $K_2$.

### Vectorized Perceptron Learning Algorithm
Given a training dataset $\{(\mathbf{p}_1, t_1), (\mathbf{p}_2, t_2), \dots, (\mathbf{p}_q, t_q)\}$ with target classifications $t_k \in \{0, 1\}$, training proceeds iteratively by updating network parameters in response to output errors:

1. Present input vector $\mathbf{p}_k$ to the network.
2. Evaluate current prediction $\vartheta_k = \phi(\mathbf{w}^T \mathbf{p}_k + b)$.
3. Compute classification error $e_k = t_k - \vartheta_k$.
4. Update weights and bias according to the error direction:
   * **Rule 1:** If $t = 1$ and $\vartheta = 0$ ($e = +1$), update: $\mathbf{w}^{\text{new}} = \mathbf{w}^{\text{old}} + \mathbf{p}$, and $b^{\text{new}} = b^{\text{old}} + 1$.
   * **Rule 2:** If $t = 0$ and $\vartheta = 1$ ($e = -1$), update: $\mathbf{w}^{\text{new}} = \mathbf{w}^{\text{old}} - \mathbf{p}$, and $b^{\text{new}} = b^{\text{old}} - 1$.
   * **Rule 3:** If $t = \vartheta$ ($e = 0$), parameters remain unchanged: $\mathbf{w}^{\text{new}} = \mathbf{w}^{\text{old}}$, and $b^{\text{new}} = b^{\text{old}}$.

For multi-neuron layers with weight matrices $\mathbf{W}$ and bias vectors $\mathbf{b}$, the general update rule incorporating learning rate $\alpha \in (0, 1]$ is:

$$\mathbf{W}^{\text{new}} = \mathbf{W}^{\text{old}} + \alpha \mathbf{e} \mathbf{p}^T$$

$$\mathbf{b}^{\text{new}} = \mathbf{b}^{\text{old}} + \alpha \mathbf{e}$$

Setting $\alpha$ excessively high causes parameter updates to oscillate erratically around optimal decision boundaries; setting $\alpha$ excessively low impairs convergence speed across practical training horizons.

### Worked Example: Hand Calculation of the Logical AND Gate
The logical AND gate problem demonstrates perceptron convergence on a linearly separable binary logic problem. Starting parameters: $\mathbf{w}^{(0)} = [0, 0]^T$, $b^{(0)} = 0$, learning rate $\alpha = 1$.

| Epoch | Input $\mathbf{p} = (p_1, p_2)$ | Target $t$ | Linear Sum $\mathbf{w}^T \mathbf{p} + b$ | Output $\vartheta$ | Error $e = t - \vartheta$ | Weight $\mathbf{w}$ After Update | Bias $b$ After Update |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $(0, 0)$ | 0 | $0$ | 0 | 0 | $(0, 0)$ | $0$ |
| **1** | $(0, 1)$ | 0 | $0$ | 0 | 0 | $(0, 0)$ | $0$ |
| **1** | $(1, 0)$ | 0 | $0$ | 0 | 0 | $(0, 0)$ | $0$ |
| **1** | $(1, 1)$ | 1 | $0$ | 0 | **+1** | $(1, 1)$ | $1$ |
| **2** | $(0, 0)$ | 0 | $1$ | 1 | **−1** | $(1, 1)$ | $0$ |
| **2** | $(0, 1)$ | 0 | $1$ | 1 | **−1** | $(1, 0)$ | $−1$ |
| **2** | $(1, 0)$ | 0 | $0$ | 0 | 0 | $(1, 0)$ | $−1$ |
| **2** | $(1, 1)$ | 1 | $0$ | 0 | **+1** | $(2, 1)$ | $0$ |

Following 6 complete epochs across all input permutations, the perceptron converges to optimal weights $\mathbf{w} = [2, 1]^T$ and bias $b = -2$. The resulting decision line $2p_1 + p_2 - 2 = 0$ separates the positive instance $(1, 1)$ from all other coordinates.

---

## 3. Multilayer Perceptrons and the Universal Approximation Theorem

To overcome the linear separability constraint identified by Minsky and Papert, modern architectures incorporate multiple intermediate layers equipped with continuous nonlinear activation functions.

### Network Architecture and Activation Dynamics
A multilayer perceptron (MLP) cascades computational representations through hidden layers:

$$\mathbf{a}^{(l)} = \phi^{(l)} \left( \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \right)$$

Common activation functions include:
* **Rectified Linear Unit (ReLU):** $\phi(z) = \max(0, z)$, standard for deep hidden layers due to non-saturating gradients for positive activations.
* **Sigmoid (Logistic):** $\phi(z) = \frac{1}{1 + e^{-z}}$, maps inputs to the real interval $(0, 1)$, suited for binary output classification and gating.
* **Hyperbolic Tangent (Tanh):** $\phi(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, provides zero-centered activations between $(-1, 1)$.
* **Softmax:** $\hat{y}_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$, normalizes class logits into an output probability distribution over mutually exclusive categories.

### The Universal Approximation Theorem
The Universal Approximation Theorem provides the formal theoretical justification for feedforward architectures:

> A standard feedforward neural network comprising a single hidden layer with a sufficient finite number of neurons and continuous, bounded, monotonically increasing nonlinear activation functions can approximate any continuous multivariate function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

While the theorem guarantees the existence of a parameter configuration capable of approximating complex mappings, it does not imply that standard gradient-based optimization algorithms will discover those optimal parameters.

---

## 4. Supervised Learning: Metrics, Losses, and Evaluation

Supervised learning frames neural model fitting as an empirical risk minimization problem.

### Metrics versus Loss Functions
* **Performance Metrics ($\mathcal{M}$):** Evaluate operational performance from an engineering standpoint (such as overall classification accuracy, recall, or precision). Many metrics cannot serve as optimization objectives because their derivatives are zero almost everywhere (step functions) or possess extreme gradients.
* **Loss Functions ($\mathcal{L}$):** Continuous, differentiable surrogate objectives derived from principles such as maximum likelihood estimation. Loss functions provide stable, non-zero gradient signals across parameter space to guide weight updates.

### Common Loss Formulations
* **Mean Squared Error (MSE):** Used in continuous numerical regression tasks following linear or non-saturating output activations:

$$\mathcal{L}_{\text{MSE}}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{N_y} \sum_{i=1}^{N_y} (y_i - \hat{y}_i)^2$$

* **Binary Cross-Entropy (BCE):** Used in binary classification following a logistic sigmoid activation layer ($\hat{y} \in [0, 1]$):

$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -y \log \hat{y} - (1 - y) \log (1 - \hat{y})$$

* **Multiclass Categorical Cross-Entropy (MCE):** Used in mutually exclusive categorical classification following a softmax layer:

$$\mathcal{L}_{\text{MCE}}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{k} y_k \log \hat{y}_k$$

### Classification Metrics and the Confusion Matrix
For discrete classification tasks, predictions are categorized against ground truth labels within a confusion matrix:

| Actual \ Predicted | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | **True Positive (TP)** | **False Negative (FN)** |
| **Actual Negative** | **False Positive (FP)** | **True Negative (TN)** |

Derived evaluation metrics reflect different diagnostic priorities:
* **Accuracy:** $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$, measures overall correctness across all classes. Highly misleading when evaluating imbalanced datasets.
* **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$, measures the reliability of positive predictions, penalizing false alarms (False Positives).
* **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$, measures the proportion of actual positive cases identified, penalizing missed detections (False Negatives).
* **F-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$, the harmonic mean of precision and recall.

#### Engineering Case Study: The Asymmetric Cost of Errors
In the classic breast cancer recurrence benchmark comprising 286 clinical patients (201 non-recurring, 85 recurring):
* An uncalibrated model predicting negative recurrence across all patients achieves 70.3% classification accuracy while failing entirely from a diagnostic standpoint.
* False Negatives carry high clinical severity: a missed recurrence sends an affected patient home without treatment. Conversely, False Positives trigger secondary confirmatory screening without life-threatening risk. Consequently, engineering model selection must optimize for **Recall** rather than raw Accuracy.

### Threshold Selection, ROC Curves, and AUC
Binary classifiers output a continuous probability estimate $\hat{y} = F(\mathbf{x}) \in [0, 1]$. Assigning final class membership requires defining a decision threshold hyperparameter $\tau \in [0, 1]$:

$$\hat{y}_{\text{class}} = \begin{cases} 1 & \text{if } \hat{y} \ge \tau \\ 0 & \text{if } \hat{y} < \tau \end{cases}$$

Varying $\tau$ from $1$ down to $0$ shifts the balance between True Positives and False Positives:
* **Receiver Operating Characteristic (ROC) Curve:** Traces the True Positive Rate ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$) along the vertical axis against the False Positive Rate ($\frac{\text{FP}}{\text{FP} + \text{TN}}$) along the horizontal axis across all possible threshold values $\tau$.
* **Area Under the Curve (AUC):** Measures the aggregate two-dimensional area beneath the ROC curve. An AUC of $1.0$ indicates perfect separation between classes across all thresholds, whereas an AUC of $0.5$ denotes performance equivalent to random guessing.

---

## 5. Backpropagation and Gradient-Based Optimization

Deep neural networks optimize internal weights and biases by solving high-dimensional non-convex minimization problems via the backpropagation algorithm.

### The Backpropagation Formulation
The scalar objective function $\mathcal{L}$ is minimized with respect to parameter matrices $\mathbf{W}^{(l)}$ and bias vectors $\mathbf{b}^{(l)}$ across all layers $l \in \{1, \dots, L\}$. Applying the multivariable chain rule propagates error signals backward from the network output.

For an arbitrary node $j$ at hidden layer $l$ receiving linear potential $V_j = \sum_i W_{ji} y_i + b_j$, the local gradient operator $\delta_j$ is defined as:

$$\delta_j = \frac{\partial \mathcal{L}}{\partial V_j} = \frac{\partial \mathcal{L}}{\partial a_j} \cdot \frac{\partial a_j}{\partial V_j} = \frac{\partial \mathcal{L}}{\partial a_j} \phi'(V_j)$$

Parameter updates are proportional to the negative gradient of the loss:

$$\Delta W_{ji}^{(l)} = - \alpha \frac{\partial \mathcal{L}}{\partial W_{ji}^{(l)}} = - \alpha \delta_j^{(l)} a_i^{(l-1)}$$

For the final output layer $L$, the local gradient reflects output error:

$$\delta_j^{(L)} = (y_j - \hat{y}_j) \phi'(V_j^{(L)})$$

For intermediate hidden layers $l < L$, error signals backpropagate from downstream nodes in layer $l+1$:

$$\delta_j^{(l)} = \phi'(V_j^{(l)}) \sum_{k} \delta_k^{(l+1)} W_{kj}^{(l+1)}$$

### Gradient Descent Variants and Optimization Challenges
Vanilla gradient descent updates model parameters along the steepest negative slope of the loss surface:

$$\mathbf{\Theta}_{k+1} = \mathbf{\Theta}_k - \alpha \nabla_{\mathbf{\Theta}} \mathcal{L}(\mathbf{\Theta}_k)$$

In high-dimensional parameter spaces, loss landscapes exhibit complex topologies including local minima, plateaus, and saddle points with disparate directional curvatures. Several optimizer variants address these dynamics:

| Optimizer | Algorithmic Mechanism | Engineering Characteristic |
| :--- | :--- | :--- |
| **Stochastic Gradient Descent (SGD)** | Evaluates gradients on single random samples | Induces high stochastic noise; escapes shallow local minima but exhibits high variance. |
| **Momentum** | Incorporates a fraction of the previous step's update vector: $\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \alpha \nabla \mathcal{L}$ | Accelerates along consistent directional gradients and dampens cross-canyon oscillations. |
| **Nesterov Accelerated Gradient (NAG)** | Evaluates gradient look-aheads based on current momentum velocity | Prevents overshooting valleys by adjusting trajectory before applying full updates. |
| **Adagrad** | Scales learning rates inversely with the sum of historical squared gradients | Performs well on sparse features but monotonically decays learning rates to zero. |
| **Adadelta / RMSprop** | Restricts historical gradient accumulation to an exponential moving average window | Resolves Adagrad's premature learning rate decay, navigating non-stationary surfaces. |

---

## 6. Practical Training Diagnosis, Validation, and Capacity Tuning

Ensuring generalization performance requires structured diagnostic protocols and rigorous data partitioning.

### Optimization versus Generalization
* **Optimization:** The numerical minimization of an empirical loss objective on a given training sample.
* **Learning (Generalization):** Ensuring the learned mapping accurately predicts outcomes on unseen evaluation data drawn from the operational distribution.

### Batch Size Dynamics
Training efficiency and gradient dynamics depend heavily on batch size selection:
* **Full Batch Learning:** Uses the entire dataset per update. Yields deterministic gradients and parallel vectorization, but demands substantial memory and tends to settle into sharp, non-generalizing local minima.
* **Stochastic (Online) Learning:** Updates parameters after every individual observation. Computationally inefficient and noisy, but regularizes training by preventing lock-in to shallow minima.
* **Mini-Batch Gradient Descent:** The standard practical compromise (typically 32 to 512 samples per batch), balancing hardware vectorization on GPUs with gradient noise that improves generalization.

### Diagnosing Underfitting and Overfitting
Monitoring training and validation loss curves over successive epochs reveals network health:
* **Underfitting (High Bias / Low Variance):** Occurs when network capacity is insufficient to capture underlying functional dependencies. Both training and validation errors plateau at unacceptable thresholds.
  * *Diagnostic Remediation:* Increase network capacity by adding hidden layers, expanding layer units, or utilizing more expressive architectures.
* **Overfitting (High Variance / Low Bias):** Occurs when network capacity is excessively high relative to training sample volume, causing weights to memorize stochastic sample noise. Training loss continues downward while validation loss diverges upward.
  * *Diagnostic Remediation:* Apply early stopping, introduce regularization, prune layer capacity, or expand training variety via data augmentation.
* **Loss Oscillations:** In stochastic and mini-batch settings, raw loss trajectories fluctuate heavily; applying **moving average filters** clarifies the underlying trend.

### Dataset Partitioning and Biomechanical Dependency Rules
Datasets must be split into three independent subsets:

| Dataset Partition | Target Proportion (Small Datasets) | Target Allocation (Large Datasets) | Operational Objective |
| :--- | :--- | :--- | :--- |
| **Training Set** | 70% | ~980,000 samples | Optimizes internal weights $\mathbf{W}$ and biases $\mathbf{b}$ via gradient backpropagation. |
| **Validation Set** | 15% | ~10,000 samples | Evaluates intermediate generalization, guides architecture choices, and tunes hyperparameters. |
| **Test Set** | 15% | ~10,000 samples | Provides an unbiased final assessment of real-world generalization performance. |

#### Critical Engineering Rule: Subject-Wise vs. Sample-Wise Partitioning
In standard machine learning benchmarks with independent and identically distributed (i.i.d.) observations, random sample-wise splitting is customary. 

However, in **biomechanical and sensor-based engineering datasets (such as IMU or gait recordings)**, samples originating from the same trial or subject share strong temporal autocorrelations and biometric signatures:
* **The Error of Sample-Wise Splitting:** Randomly allocating individual sensor time frames across train, validation, and test sets causes severe **data leakage**, as idiosyncratic subject gait patterns contaminate the training set. This artificially inflates validation accuracy while leading to catastrophic failure on new individuals.
* **The Subject-Wise Mandate:** Biomechanical datasets must be partitioned strictly **subject-wise**, ensuring all measurements from a given subject belong exclusively to a single dataset split.

### The Golden Rule of Model Training
When engineering neural network pipelines, practitioners must adhere to a strict sequence of priorities:

$$\textbf{First, resolve Underfitting} \longrightarrow \textbf{Then, resolve Overfitting}$$

1. Ensure the network architecture possess sufficient representational capacity to drive training error near zero.
2. Once the capacity constraint is resolved and the model achieves convergence, apply regularization, architectural pruning, and data augmentation to close the generalization gap.
