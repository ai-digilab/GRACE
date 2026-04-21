# GRACE: Adaptive Backbone Scaling for Memory-Efficient Class Incremental Learning

> **Official PyTorch implementation** of the paper:  
> *"Grow, Assess, Compress: Adaptive Backbone Scaling for Memory-Efficient Class Incremental Learning"*

[![arXiv 2603.08426](https://img.shields.io/badge/arXiv-2603.08426-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.08426)[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

---

## 📋 Overview

**GRACE** is a dynamic framework for **Class Incremental Learning (CIL)** that intelligently manages model capacity through a cyclic **"GRow, Assess, ComprEss"** strategy. Unlike traditional expansion-based methods that suffer from uncontrolled parameter growth, GRACE:

- ✅ **Evaluates backbone utilization** using normalized effective rank (`eRank`)
- ✅ **Adaptively expands** the network only when capacity saturation is detected
- ✅ **Compresses redundant backbones** via dual-level knowledge distillation
- ✅ **Reduces memory footprint by up to 73%** compared to purely expansionist models
- ✅ **Surpasses state-of-the-art by up to 4 accuracy points** on memory-aligned evaluations

### Key Contributions

1. **Dynamic Capacity Management**: Saturation-aware mechanism using effective rank to decide when to expand architecture
2. **Enhanced Compression Phase**: Importance-aware student initialization + dual-level (logit + feature) knowledge distillation
3. **Resource-Aware Versatility**: Tunable expansion rate to control model growth

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ai-digilab/GRACE.git
cd GRACE
pip install -r requirements.txt
```

### Running Experiments

GRACE uses JSON configuration files located in `exps/` to define experimental settings.

```bash
# Example: CIFAR-100, Base-0, Increment-10 protocol
python main.py --config ./exps/grace_cifar100_b0i10.json
```

#### Key Hyperparameters

| Parameter       | Description                                        |
| --------------- | -------------------------------------------------- |
| `t_base`        | Saturation threshold for expansion decision (`τ₁`) |
| `t_decay`       | Decay factor when compression occurs (`ρ`)         |
| `memory_size`   | Exemplar buffer capacity (`M`)                     |
| `c_kd_factor`   | Weight for logit-level distillation                |
| `c_pred_factor` | Weight for feature-level distillation              |

---

## 📁 Code Structure
```bash
GRACE/
├── 📁 convs/                    # Convolutional network architectures
│   ├── cifar_resnet.py          # ResNet variants optimized for CIFAR datasets
│   ├── linears.py               # Linear layer utilities and custom heads
│   └── resnet.py                # Standard ResNet backbone implementations
│
├── 📁 exps/                     # Experiment configuration files (JSON)
│
├── 📁 models/                   # Core model implementations
│   ├── base.py                  # Abstract base class for continual learning models
│   └── grace.py                 # GRACE algorithm implementation
│
├── 📁 utils/                    # Utility modules
│   ├── autoaugment.py           # AutoAugment policy
│   ├── data.py                  # Dataset wrappers and preprocessing
│   ├── data_manager.py          # Manages incremental data splits & loaders
│   ├── factory.py               # Factory for instantiating models
│   ├── inc_net.py               # Incremental network for GRACE
│   ├── ops.py                   # Common operations
│   └── toolkit.py               # Helpers: parameter counting, metrics
│
├── main.py                      # 🚀 Entry point: CLI arg parsing + config loading
└── trainer.py                   # 🔄 Training loop, evaluation & logging logic
```

---

## 🙏 Acknowledgements

- Built upon the [PyCIL](https://github.com/G-U-N/PyCIL) framework for class-incremental learning

---

## 📚 Citation

If you use GRACE in your research, please cite our paper:

```bibtex
@article{garcia2026grow,
  title={Grow, Assess, Compress: Adaptive Backbone Scaling for Memory-Efficient Class Incremental Learning},
  author={Garcia-Casta{\~n}eda, Adrian and Irureta, Jon and Imaz, Jon and Lojo, Aizea},
  journal={arXiv preprint arXiv:2603.08426},
  year={2026}
}
```
