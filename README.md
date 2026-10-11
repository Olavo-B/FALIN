# FALIN: An FPGA Accelerator for Lightweight Image Networks using Transformed Temporal Data

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- ![FALIN](misc/docs/img/logo.png) -->
<!-- Center image -->
<p align="center">
  <img src="misc/docs/img/logo.png" alt="FALIN"/>
</p>

Processing time-series on FPGAs via recurrent models is resource-heavy. FALIN proposes an area-efficient pipeline by transforming 1D temporal data into 2D images. We use a lightweight CNN for feature extraction and an XGBoost classifier, bypassing standard MLPs. This mapping shifts CNN operations to DSPs and unrolled trees to LUTs. Evaluated against temporal baselines, FALIN drastically reduces area footprint and latency while preserving high classification accuracy.

## Overview

FALIN (Fast and Lightweight Image Networks) is a framework for Time-Series Classification on resource-constrained edge devices. It works in two steps:

1. **Time series to image:** each 1D signal is converted into a 2D image (using STFT or GAF) and resized to a fixed size.
2. **Lightweight classification:** a small CNN extracts features from the image, and an XGBoost classifier makes the final decision.

### Key Results

Tested on ten datasets from the UCR Time Series Archive, FALIN needs about **22.6M multiply–accumulate operations** per inference, versus roughly **270M** for the LSTM-FCN baseline. This is an order-of-magnitude reduction in computation, while keeping competitive accuracy on periodic signals such as ECG5000, Wafer, and FordA.

## Repository Structure

```
FALIN/
├── src/                      # Source code
├── misc/docs/img/            # Figures and logo
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Usage

```bash
# Clone the repository
git clone https://github.com/lesc-ufv/FALIN.git
cd FALIN

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Citation

If you use FALIN in your research, please cite:



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Funding provided by Fapemig APQ-01577-22, CAPES, CNPq, IEEE IES, SBC, and UFV Lato Sensu in Robotica.
- The [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) for the benchmark datasets.

## Contact

- **Olavo Barros** - olavo.barros@ufv.br
- **Michel Silva** - michel.m.silva@ufv.br
- **Jose Augusto M. Nacif** - jnacif@ufv.br
- **Ricardo Ferreira** - ricardo@ufv.br
- **Racyus D. G. Pacífico** - racyus.pacifico@ufop.edu.br

Universidade Federal de Viçosa (UFV) and Universidade Federal de Ouro Preto (UFOP).
