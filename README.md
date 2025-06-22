# RL-based-precoder-design-in-a-2-2-MIMO-OFDM-system-with-256-QAM-modulation.

This repository implements a **reinforcement learning (RL)** approach to select a precoder matrix in a **2×2 MIMO-OFDM** system with **256-QAM** modulation. The simulation runs on GPU using **CuPy**, and the RL agent is trained using **PPO** from **stable-baselines3**.

---

## 1. System Model 🧮

We transmit over \(N = 64\) subcarriers, with cyclic prefix length \(CP\). For subcarrier \(k\):

$\[
\mathbf{y}[k] = \mathbf{H}[k] \mathbf{W} \mathbf{s}[k] + \mathbf{n}[k]
\] $

- \($\mathbf{H}[k] \in \mathbb{C}^{2 \times 2}\)$:  flat-fading channel  
- $\(\mathbf{W} \in \mathbb{C}^{2 \times 2}\)$:     precoding matrix chosen from a codebook  
- $\(\mathbf{s}[k]\)$:       QAM vector  
- $\(\mathbf{n}[k]\)$:     AWGN

We use:
- 256-QAM mapping/demapping
- OFDM modulation (IFFT/FFT + CP)
- Equalization via ZF (baseline) or RL-chosen precoder

---

## 2. Reinforcement Learning Setup

### State:
$$
s = [\Re(\mathbf{H}), \Im(\mathbf{H}), \text{SNR}]
$$

Total input dimension: 9

### Actions:
Discrete actions $a \in \{0,1\}$ map to codebook precoders:

$$
\mathbf{W}_0 = \mathbf{I}, \quad
\mathbf{W}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

### Reward:
$$
r = -\text{BER}
$$

---

## 3. BER Computation

Empirical:
$$
\text{BER} = \frac{\# \text{bit errors}}{\# \text{total bits}}
$$

Approximate for 256-QAM:
$$
\text{BER} \approx \frac{4}{k} \left(1 - \frac{1}{\sqrt{M}} \right) Q\left(\sqrt{\frac{3k}{M-1} \cdot \text{SNR}}\right)
$$

Where $k = \log_2 M = 8$ and $M = 256$.

---

## 4. Training Loop

```python
from stable_baselines3 import PPO
from env import PrecoderEnv

env = PrecoderEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
