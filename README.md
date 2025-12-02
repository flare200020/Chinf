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
# Dataset
For detailed information about dataset acquisition, preprocessing procedures, and data organization,  
please refer to the [`datasets` section](https://github.com/ssarfraz/QuoVadisTAD) of the **QuoVadisTAD** repository.  
We sincerely thank the authors of *QuoVadisTAD* for their excellent open-source contribution and dataset preparation pipeline. 

# Get started

To run the ChInf-based anomaly detection, use the following command. 
```bash
bash ./scripts/anomaly_detection/SWaT.sh
```

## 📜 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{
wang2025channel,
title={Channel Matters: Estimating Channel Influence for Multivariate Time Series},
author={Muyao Wang and Zeke Xie and Bo Chen and Hongwei Liu and James Kwok},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=U2AF01VJyg}
}
```
