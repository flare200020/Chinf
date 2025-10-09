# Channel Matters: Estimating Channel Influence for Multivariate Time Series

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://nips.cc/)

[cite_start]This repository contains the official implementation for the paper: **Channel Matters: Estimating Channel Influence for Multivariate Time Series** [cite: 1, 2][cite_start], accepted at NeurIPS 2025[cite: 29].

[cite_start]Our work introduces **Channel-wise Influence (ChInf)**, the first method designed to estimate the influence of individual data channels in Multivariate Time Series (MTS)[cite: 18]. [cite_start]ChInf serves as an efficient post-hoc interpretability tool that quantifies how modifying training data channels impacts model performance, without requiring expensive retraining[cite: 14].

[cite_start]We demonstrate the effectiveness of ChInf by applying it to two critical MTS tasks: anomaly detection and data pruning, where our ChInf-based methods achieve top-1 rankings compared to existing approaches[cite: 20, 21].

## 🚀 Key Contributions

* [cite_start]**Novel Method**: We propose **Channel-wise Influence (ChInf)**, a new data-centric approach to quantify the influence of individual channels in Multivariate Time Series (MTS)[cite: 18, 44].
* [cite_start]**Practical Applications**: We derive two channel-wise algorithms from ChInf for classic MTS tasks[cite: 19]:
    1.  [cite_start]**MTS Anomaly Detection**: Using channel-level self-influence as a robust anomaly score[cite: 45, 119].
    2.  [cite_start]**MTS Channel Pruning**: A new data pruning strategy to select a representative subset of channels, reducing data size and computational cost while maintaining performance[cite: 45, 137].
* [cite_start]**State-of-the-Art Performance**: Extensive experiments show that our ChInf-based methods achieve superior performance, ranking first on various benchmark datasets for both anomaly detection and data pruning tasks[cite: 21, 48].

## 🛠️ Methodology Overview

[cite_start]Traditional influence functions like TracIn calculate the influence of an entire data sample, failing to distinguish the impact of individual channels in MTS[cite: 40, 84, 85]. [cite_start]To address this, we formulate ChInf to estimate the influence between specific channels[cite: 86].

### Channel-wise Influence (ChInf)

[cite_start]The influence between a training channel $c_i'$ and a test channel $c_j$ is defined as[cite: 98]:
$$CIF(c_{i}', c_{j}) := \eta\nabla_{\theta}L(c_{i}';\theta)^{\top}\nabla_{\theta}L(c_{j};\theta)$$
[cite_start]This allows us to construct a **Channel-wise Influence Matrix ($M_{CInf}$)**, where each element represents how training on channel *i* helps reduce the loss for channel *j*[cite: 92, 103].

### Applications

1.  [cite_start]**Anomaly Detection**: We use the self-influence of each channel (the diagonal elements of $M_{CInf}$) as an anomaly score[cite: 121]. [cite_start]A high self-influence score indicates that a channel's data point is anomalous relative to the learned patterns of the model[cite: 117]. [cite_start]The final anomaly score for a given timestamp is the maximum score across all channels[cite: 123].

2.  [cite_start]**Channel Pruning**: To create a smaller, representative dataset, we rank channels by their self-influence scores and select a subset at regular intervals[cite: 148]. [cite_start]This ensures the pruned dataset contains a diverse set of channels, which is more effective than simply choosing the most influential ones[cite: 149]. [cite_start]Our results show we can preserve model performance with as little as 50% of the original channels[cite: 259].

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/flare200020/Chinf.git](https://github.com/flare200020/Chinf.git)
    cd Chinf
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    (Please create a `requirements.txt` file with all necessary packages like PyTorch, NumPy, etc.)
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Usage

### Anomaly Detection

To run the ChInf-based anomaly detection, use the following command. The anomaly scores will be computed based on the self-influence of each channel.

```bash
# Example: Run anomaly detection on the SWaT dataset using the GCN-LSTM model
python run_anomaly_detection.py --dataset SWAT --model GCN-LSTM
# Get started
