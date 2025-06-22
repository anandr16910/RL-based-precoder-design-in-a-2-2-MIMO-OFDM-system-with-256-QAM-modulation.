# RL-based-precoder-design-in-a-2-2-MIMO-OFDM-system-with-256-QAM-modulation.

This repository implements a **reinforcement learning (RL)** approach to select a precoder matrix in a **2×2 MIMO-OFDM** system with **256-QAM** modulation. The simulation runs on GPU using **CuPy**, and the RL agent is trained using **PPO** from **stable-baselines3**.

---

## 1. System Model 🧮

We transmit over \(N = 64\) subcarriers, with cyclic prefix length \(CP\). For subcarrier \(k\):

$\[
\mathbf{y}[k] = \mathbf{H}[k] \mathbf{W} \mathbf{s}[k] + \mathbf{n}[k]
\] $
$
- \(\mathbf{H}[k] \in \mathbb{C}^{2 \times 2}\): flat-fading channel  
- \(\mathbf{W} \in \mathbb{C}^{2 \times 2}\): precoding matrix chosen from a codebook  
- \(\mathbf{s}[k]\): QAM vector  
- \(\mathbf{n}[k]\): AWGN
$
We use:
- 256-QAM mapping/demapping
- OFDM modulation (IFFT/FFT + CP)
- Equalization via ZF (baseline) or RL-chosen precoder

---

## 2. Reinforcement Learning (RL) Formulation

### State:  
\[
s = [\Re(\mathbf{H}), \Im(\mathbf{H}), \text{SNR}]
\]  
Dimensions: 9

### Actions:  
Choose \(a \in \{0,1\}\) corresponding to precoders
\[
\mathbf{W}_0 = \mathbf{I}, \quad
\mathbf{W}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\]

### Reward:  
\[
r = -\text{BER}
\]  
(you could also use throughput or spectral efficiency)

---

## 3. BER Computation

Empirically:
\[
\text{BER} = \frac{\text{# bit errors}}{\text{# total bits}}
\]

For 256-QAM:
\[
M = 256,\quad k = 8,\quad \text{BER} \approx \frac 4 k \Bigl(1-\tfrac1{\sqrt M}\Bigr) Q\!\Bigl(\sqrt{\tfrac{3k}{M-1}\text{SNR}}\Bigr)
\]

---

## 4. Training and Inference Loop

We use **PPO** (stable-baselines3):

```python
from stable_baselines3 import PPO
from env import PrecoderEnv  # CuPy-based OFDM sim environment

env = PrecoderEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
