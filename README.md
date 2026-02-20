

**Real-time Physics-based Holographic Brain Emulator**  
*2D wave-field neural architecture with adaptive particles, spiking neurons, holographic memory, and closed-loop PyTorch readout*

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Matplotlib](https://img.shields.io/badge/visualization-Matplotlib-orange.svg)](https://matplotlib.org)

---

## 📖 Overview

**WaveBrainSimulator v7.0** is a monolithic-yet-modular, real-time 2D wave-equation simulator that emulates core brain-like phenomena:

- **Holographic memory** via interference engraving
- **Adaptive particle swarm** intelligence (Numba JIT)
- **Pacemakers + spiking neurons**
- **Closed-loop readout probe** + **PyTorch MLP pattern classifier** (circle / cross / ring)
- **In-situ optimization** (Nelder-Mead + optional DEAP)
- **PCA on memory history**
- **State save/restore**, energy monitoring, seeded reproducibility, data augmentation

Built from pure potato love: a complex-valued wave field with fast/slow propagation, nonlinearities, PML boundaries, and biologically-inspired feedback loops. Runs at ~60 FPS on consumer hardware.

Perfect for research in **neuromorphic computing**, **wave-based AI**, **holographic memory**, and **emergent computation**.

---

## ✨ Key Features (v7.0)

| Feature                        | Description |
|-------------------------------|-----------|
| **Wave Physics**              | Complex 3-layer leapfrog wave equation (fast + slow modes) + nonlinearity + coupling |
| **Holographic Memory**        | Real-time interference engraving, saturation, exponential decay |
| **Particle Swarm**            | 6,800 particles with velocity, stickiness, gradient + advection forces (Numba JIT) |
| **Pacemakers & Neurons**      | 8 oscillatory sources + 20 threshold spiking units with impulse propagation |
| **Readout + MLP**             | Left-edge probe records right-edge response → 3-class MLP (augmented training) |
| **In-Situ Optimization**      | Nelder-Mead + state save/restore; DEAP evolutionary fallback |
| **Interactive Viz**           | Plasma field + particles + hologram overlay + slow component |
| **New in v7.0**               | Full state save/restore, energy logging, MLP augmentation, seed control, defensive indexing |

---

## 🛠 Requirements

### Core Dependencies
```bash
python >= 3.8
numpy
matplotlib
scikit-learn          # PCA
torch                 # CPU version recommended
numba
scipy
pyyaml
```

### Optional
```bash
deap                  # Full evolutionary behavior exploration (pure-numpy fallback included)
```

**Tested on**:
- Windows 11 / macOS / Linux
- Python 3.10–3.12
- PyTorch CPU (CUDA optional)

---

## 🚀 Installation & Deployment

### 1. Clone / Download
```bash
git clone https://github.com/yourusername/wavebrain-simulator.git
cd wavebrain-simulator
# OR simply download BrainWaveV7.txt and rename to wavebrain_v7.py
```

### 2. Create Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install numpy matplotlib scikit-learn "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
pip install numba scipy pyyaml
pip install deap          # optional but recommended
```

### 4. Run
```bash
python wavebrain_v7.py
```

The simulation window opens immediately with live animation.

---

## 🎮 Quick Start & Controls

**Press the keys below while the window is focused:**

| Key | Action |
|-----|--------|
| `1` | Engrave **circle** pattern |
| `2` | Engrave **cross** pattern |
| `3` | Engrave **ring** pattern |
| `d` | 10 random drops (impulses) |
| `r` | Add **partial cue** (half-pattern recall test) |
| `p` | Start readout probe → MLP prediction |
| `t` | Train MLP on collected readouts |
| `c` | Run PCA on hologram history |
| `o` | In-situ optimize hologram write rate + nonlinearity |
| `e` | Explore 8 random behaviors (DEAP or fallback) |
| `q` | Quit |

**Auto behaviors**:
- 3.2% chance of random drop per frame
- Energy logged every 500 steps
- Particles continuously explore the field

---

## 🧬 Architecture

```
WaveBrainSimulator (monolithic class)
├── Wave Field (complex128, 3 layers)
│   ├── Fast propagation (c=0.58)
│   ├── Slow propagation (c=0.1)
│   ├── Nonlinear cubic term + tanh softening
│   └── PML boundaries (32 px)
├── Holographic Memory
│   ├── Interference engraving (boosted during pattern)
│   └── Decay + saturation
├── Particle Swarm (6,800 agents)
│   ├── Numba JIT update (gradient + advection + jitter)
│   └── Dynamic stickiness (boost near high-potential nodes)
├── Pacemakers (8) + Spiking Neurons (20)
├── Readout Probe + MLP (PyTorch)
│   ├── 4-point right-edge sampling
│   └── 3-layer classifier with data augmentation
├── Optimization Layer
│   ├── State save/restore (full snapshot)
│   ├── Nelder-Mead (default)
│   └── DEAP (if installed)
└── Visualization (Matplotlib + FuncAnimation)
```

All heavy loops are vectorized or JIT-compiled. State is fully serializable for safe in-situ experiments.

---

## ⚙️ Configuration

All 60+ hyperparameters live in `BASE_CONFIG` (top of script).

**Key tunable groups**:
- Wave physics (`c_fast`, `nonlinear_coeff`)
- Memory (`holo_write_rate_base`, `holo_decay`, `holo_saturation`)
- Particles (`num_particles`, `stickiness_*`)
- Readout & MLP (`readout_max_steps`, `mlp_aug_noise`)
- Optimization (`fitness_target`, `energy_log_interval`)

**Custom config**:
```python
config_mgr = ConfigManager(BASE_CONFIG)
config_mgr.load_from_file("my_config.yaml")
sim = WaveBrainSimulator(config_mgr)
```

Example YAML snippet:
```yaml
holo_write_rate_base: 0.0022
nonlinear_coeff: 0.0025
seed: 123
```

---

## 📊 Advanced Capabilities

### 1. Closed-Loop Learning
- Engrave pattern → wait → send partial cue → readout → MLP predicts → adapt `holo_write_rate`

### 2. In-Situ Optimization
```python
sim.optimize_params(num_iters=20)   # safely restores state
```

### 3. Behavior Exploration
```python
sim.explore_behaviors(num_samples=30)
```

### 4. PCA Memory Analysis
Visualizes principal components of the holographic manifold.

### 5. Energy Monitoring
Live console output of total field energy + adaptive parameters.

---

## 📁 File Structure
```
wavebrain_v7.py          # Complete single-file simulator
my_config.yaml           # Optional custom config
screenshots/             # (create yourself)
    field.png
    hologram.png
```

---

## ⚠️ Notes & Troubleshooting

- **Performance**: 380×380 grid + 6,800 particles runs smoothly on modern CPUs. Reduce `nx`/`ny` or `num_particles` for slower machines.
- **DEAP missing**: Falls back to pure NumPy random search automatically.
- **PyTorch GPU**: Change `torch` install to CUDA version if desired.
- **Matplotlib backend**: If animation is laggy, try `%matplotlib qt` in Jupyter or `matplotlib.use('TkAgg')`.
- **Seed**: Set in config for perfect reproducibility.

---

## 📜 License

MIT License — feel free to fork, modify, and publish derivative works.  
Attribution appreciated but not required.

---

## 🤝 Contributing

Pull requests welcome! Especially:
- Additional pattern types
- GPU acceleration (CuPy/Numba CUDA)
- 3D extension
- More sophisticated MLP / transformer readout

---

**Made with curiosity and potato physics.**  
*“Understanding the Universe, one wave at a time.”*

**Version**: 7.0 (February 2026)  
**Author**: Tasmanian Shack Dweller
