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
----
Empirical:


$$ \text{BER} = \frac{\text{bit errors}}{\text{total bits}} $$


Approximate for 256-QAM:
BER ≈ (4 / k) × (1 - 1/√M) × Q(√(3k / (M−1) × SNR))  
where:  
- M = 256  
- k = log₂(M) = 8  
- Q(x) is the tail probability of the standard normal distribution




---

## 4. Training Loop

```python
from stable_baselines3 import PPO
from env import PrecoderEnv

env = PrecoderEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)


---

mimo_ofdm_rl_precoder/
│
├── README.md
├── requirements.txt
├── env.py # PrecoderEnv with CuPy simulation
├── train_rl.py # PPO training script
├── eval_ber.py # BER vs SNR plots
└── src/
└── mimo_ofdm.py # Core functions: qam_map, ofdm_mod/demod, simulate()

