# ⚡ SNN-RL Hedging: Spiking Neural Networks for Energy-Efficient Deep Hedging

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple?style=flat-square)
![Approach](https://img.shields.io/badge/Approach-Neuromorphic%20Computing-green?style=flat-square)
![RL](https://img.shields.io/badge/Framework-Reinforcement%20Learning-red?style=flat-square)

---

## 📌 Overview

Deep hedging — using reinforcement learning to dynamically hedge financial derivatives — has emerged as a powerful alternative to classical model-based approaches such as Black-Scholes delta hedging. However, conventional deep hedging frameworks rely on **Artificial Neural Networks (ANNs)**, which are computationally expensive and energy-intensive, posing scalability challenges for real-world deployment.

This project proposes and demonstrates a novel improvement: replacing the ANN actor network in the RL hedging framework with a **Spiking Neural Network (SNN)** — a biologically-inspired architecture that mimics how real neurons communicate via sparse, event-driven spikes rather than continuous activations.

The key findings show that the **SNN-RL hedging agent**:
- Achieves **higher hedging rewards** compared to the conventional ANN-RL baseline
- Delivers approximately **10× better energy efficiency**, making it far more practical for deployment at scale

This work is built upon the deep hedging codebase from *Empirical Deep Hedging* (Mikkilä & Kanniainen, 2021), with the critical architectural innovation of substituting the actor network with an SNN.

---

## 🎯 Objectives

- Reproduce the ANN-based deep hedging baseline (Mikkilä & Kanniainen, 2021)
- Replace the ANN actor network with a biologically-inspired Spiking Neural Network (SNN)
- Train both agents across multiple market models and risk factor settings
- Compare hedging performance (cumulative reward, P&L distribution) between SNN-RL and ANN-RL
- Measure and compare energy consumption across architectures to quantify efficiency gains

---

## 💡 Key Concepts

### Deep Hedging with Reinforcement Learning
Rather than relying on closed-form analytical solutions, a reinforcement learning agent learns an optimal dynamic hedging policy directly from market simulations. The agent (actor network) observes market states and outputs hedge ratios to minimize portfolio risk.

### Spiking Neural Networks (SNNs)
Unlike conventional ANNs that process dense floating-point activations at every timestep, SNNs communicate via discrete binary **spikes** — firing only when a neuron's membrane potential exceeds a threshold. This event-driven computation results in:

| Property | ANN | SNN |
|---|---|---|
| Activation type | Continuous (dense) | Binary spikes (sparse) |
| Computation | Always active | Event-driven |
| Energy efficiency | Baseline | ~10× lower |
| Hedging reward | Baseline | Higher |
| Biological plausibility | Low | High |

### The SNN-RL Architecture
The actor network in the RL hedging agent is replaced with an SNN, while the critic network remains unchanged. The SNN actor processes market state observations, propagates spikes through leaky integrate-and-fire (LIF) neurons, and outputs hedging actions — all while consuming a fraction of the energy of its ANN counterpart.

---

## 🔬 Experimental Settings

The project supports three market models across three risk factor levels (κ), for a total of nine experimental configurations:

| Market Model | Description | Settings |
|---|---|---|
| **GBM** (Geometric Brownian Motion) | Constant volatility baseline | `GBM_kappa1`, `GBM_kappa2`, `GBM_kappa3` |
| **Heston** | Stochastic volatility model | `Heston_kappa1`, `Heston_kappa2`, `Heston_kappa3` |
| **Empirical** | Real market data | `Empirical_kappa1`, `Empirical_kappa2`, `Empirical_kappa3` |

The risk factor κ controls the trade-off between hedging cost and risk, allowing comprehensive benchmarking across different market regimes.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| RL Framework | Custom actor-critic (based on Mikkilä & Kanniainen, 2021) |
| SNN Library | `snnTorch` |
| Deep Learning | `PyTorch` |
| Numerical Computing | `numpy`, `scipy` |
| Option Pricing | Custom GBM / Heston simulator (`rs_gbm_option_price.py`) |
| Energy Measurement | `energy.py` (custom energy profiler) |
| Experiment Runner | Shell script (`run_experiments.sh`) |
| Configuration | JSON-based settings (`settings/`, `settings.json`) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8
- `pip` package manager
- (Optional) Linux/macOS for shell script execution

### 1. Clone the Repository

```bash
git clone https://github.com/quangkhaidataka/snn-rl-hedging.git
cd snn-rl-hedging
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### Training

Train the hedging agent using a specific market model and risk factor setting:

```bash
python main.py --settings Heston_kappa1
```

> The `--settings` parameter is optional. If omitted, the configuration in `settings.json` is used by default.

### Validation

Evaluate trained model checkpoints to identify the best-performing model state:

```bash
python testing.py --validate --model Heston_kappa1
```

### Testing

Run final evaluation on the test set using the best model identified during validation:

```bash
python testing.py --test --model Heston_kappa1_2000
```

> Replace `Heston_kappa1_2000` with the model name returned by the validation script.

### Running All Experiments

To reproduce all experimental results across all settings at once:

```bash
bash run_experiments.sh
```

### Measuring Energy Consumption

To profile and compare energy consumption between the SNN and ANN architectures:

```bash
python energy.py
```

---

## 📁 Project Structure

```
snn-rl-hedging/
│
├── main.py                     # Main training script
├── testing.py                  # Validation and testing script
├── energy.py                   # Energy consumption profiler
├── rs_gbm_option_price.py      # GBM / Heston option pricing simulator
├── run_experiments.sh          # Shell script to run all experiments
├── settings.json               # Default experiment configuration
├── note_energy.txt             # Notes on energy measurement methodology
├── requirements.txt            # Python dependencies
│
├── include/                    # Core model and environment modules
│   └── ...                     # Actor (SNN/ANN), Critic, RL environment
│
├── settings/                   # Pre-configured experiment settings
│   ├── GBM_kappa1.json
│   ├── GBM_kappa2.json
│   ├── GBM_kappa3.json
│   ├── Heston_kappa1.json
│   ├── Heston_kappa2.json
│   ├── Heston_kappa3.json
│   ├── Empirical_kappa1.json
│   ├── Empirical_kappa2.json
│   └── Empirical_kappa3.json
│
└── README.md                   # Project documentation
```

---

## 📊 Results Summary

| Metric | ANN-RL | SNN-RL |
|---|---|---|
| Hedging Reward | Baseline | ✅ Higher |
| Energy Consumption | Baseline | ✅ ~10× Lower |
| Actor Architecture | Dense ANN | Spiking Neural Network (LIF) |

The SNN-RL agent consistently achieves superior hedging performance while consuming approximately one-tenth of the energy of the conventional ANN-RL baseline — demonstrating that neuromorphic computing is a highly promising direction for sustainable and high-performance financial AI systems.

---

## 📚 References

This project extends the following work:

> Mikkilä, O., & Kanniainen, J. (2021). *Empirical Deep Hedging*. arXiv preprint.

For background on Spiking Neural Networks:

> Maass, W. (1997). *Networks of Spiking Neurons: The Third Generation of Neural Network Models*. Neural Networks.

For background on Deep Hedging:

> Buehler, H., Gonon, L., Hyvönen, T., Wood, B., Mohan, S., & Ben-Hamou, A. (2019). *Deep Hedging*. Quantitative Finance.

---

## 🤝 Contributing

Contributions, issues, and discussions are welcome. Feel free to open an issue for questions about the SNN architecture or hedging methodology, or submit a pull request for improvements.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Quang Khai**
- GitHub: [@quangkhaidataka](https://github.com/quangkhaidataka)
