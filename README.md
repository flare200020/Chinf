# Channel Matters: Estimating Channel Influence for Multivariate Time Series

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://nips.cc/)

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

To run the ChInf-based anomaly detection, use the following command. 

```bash
bash ./scripts/anomaly_detection/SWaT.sh
# Get started
