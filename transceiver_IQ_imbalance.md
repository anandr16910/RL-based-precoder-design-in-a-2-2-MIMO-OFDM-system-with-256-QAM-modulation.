# Transceiver IQ Imbalance and Data Converters Notes

## Part 1: Transceiver IQ Imbalance (Tx & Rx)

### 1. Tx IQ Imbalance
* **Concept Overview:** IQ Imbalance in the transmitter (Tx) involves both gain and phase imbalances.
* It causes trouble and harm to the intended receiver and creates co-channel interference.
* **Visual Plot Description (Zero-IF Spectrum):** The notes illustrate a spectrum plot where the "Desired Signal" and "Image" overlap around a center frequency ($f_0$). The gap between the top of the desired signal's power and the image's power is labeled as the Image Rejection Ratio (ISR).

**Mathematical Model for Tx IQ Imbalance**
The complex baseband signal with gain error ($\Delta G$) and phase error ($\Delta\phi$) can be represented as:
$$\tilde{Y}_{BB}(t) = \frac{1}{2} \left[ \left(1-\frac{\Delta G}{2}\right)\cos\frac{\Delta\phi}{2} - j\left(1-\frac{\Delta G}{2}\right)\sin\frac{\Delta\phi}{2} \right] x_I(t) + \frac{1}{2} \left[ -\left(1+\frac{\Delta G}{2}\right)\sin\frac{\Delta\phi}{2} + j\left(1+\frac{\Delta G}{2}\right)\cos\frac{\Delta\phi}{2} \right] x_Q(t)$$

Represented in matrix form:
$$\begin{bmatrix} I_{out} \\ Q_{out} \end{bmatrix} = \begin{bmatrix} \left(1-\frac{\Delta G}{2}\right)\cos\frac{\Delta\phi}{2} & -\left(1+\frac{\Delta G}{2}\right)\sin\frac{\Delta\phi}{2} \\ -\left(1-\frac{\Delta G}{2}\right)\sin\frac{\Delta\phi}{2} & \left(1+\frac{\Delta G}{2}\right)\cos\frac{\Delta\phi}{2} \end{bmatrix} \begin{bmatrix} x_I(t) \\ x_Q(t) \end{bmatrix}$$

**Characterizing Degradation (EVM & ISR)**
To characterize the degradation caused by these imbalances, we evaluate Error Vector Magnitude (EVM) and Image Suppression Ratio (ISR). 

Using small-angle approximations where $\Delta G$ & $\Delta\phi \ll 1$:
* $\cos x \approx 1$
* $\sin x \approx x$
* Neglect $\Delta G \cdot \Delta\phi$ terms for small $x$

This simplifies the ISR relation to:
$$ISR = \frac{\Delta G^2 + \Delta\phi^2}{4}$$

The Signal-to-Noise Ratio (SNR) relates to ISR inversely:
$$SNR_{dB} = \frac{1}{ISR} = \frac{4}{\Delta G^2 + \Delta\phi^2}$$

**Example Calculations:**
* If $\Delta G = 10\%$ and $\Delta\phi = 0^\circ$, $SNR \approx 26$ dB.
* If $\Delta G = 10\%$ and $\Delta\phi = 3^\circ$, $SNR \approx 24.3$ dB.

---

### 2. Rx IQ Imbalance (Case 3)
* **Scenario:** Ideal Transmitter, but Receiver (Rx) has the imbalance.

**Block Diagram (Rx Mixer):**
> $x(t)$ $\longrightarrow$ **[Quad Mixer]** $\longrightarrow y_{RF}(t) \longrightarrow$ **[ $Re\{\}$ ]** $\longrightarrow y(t)$
> *Note: Mixer is driven by the Local Oscillator signal* $e^{j\omega_0 t}$.

The resulting ISR for the receiver follows the same derived ratio as the transmitter:
$$ISR = \frac{|\beta_{Rx}|^2}{|\alpha_{Rx}|^2} = \frac{\Delta G^2 + \Delta\phi^2}{4}$$

---

## Part 2: Data Converters (ADC / DAC)

### Core Concepts
* **Processes:** Sampling, Quantization, and Encoding.
* **Key Metrics & Terminology:** Nyquist criteria, INL (Integral Non-Linearity), DNL (Differential Non-Linearity), Dynamic Range, Sensitivity, Resolution.
* **Sources of Error:** There are two main sources of error: Sample/Hold noise and Quantization noise/error.

**Block Diagram (Analog to Digital):**
> Analog Signal $\longrightarrow$ **[ ADC ]** $\longrightarrow$ Digital Data ($x_q(n)$)

### Quantization Types
There are two primary types of quantizers discussed: **Mid-tread** and **Mid-rise**.
* $n_b$ = number of bits.
* $A_{clip}$ = maximum clipping amplitude.
* $\Delta q$ = Quantization Step Size.

For a Mid-tread quantizer:
$$\Delta q = \frac{2A_{clip}}{2^{n_b} - 1}$$

For a Mid-rise quantizer:
$$\Delta q = \frac{2A_{clip}}{2^{n_b}}$$

**Mid-Tread Quantizer Example ($n_b = 3$, $A_{clip} = 1$)**
For $n_b = 3$, the step size $\Delta q = \frac{2}{7} \approx 0.2857$.

| Input Range $x(n)$ | Quantized Value $x_q(n)$ | Binary Output |
| :--- | :--- | :--- |
| -1.00 to -0.8571 | -1.000 | 000 |
| -0.8571 to -0.5714 | -0.714 | 001 |
| -0.5714 to -0.2857 | -0.4287 | 010 |
| -0.2857 to 0.00 | -0.1428 | 011 |
| 0.00 to 0.2857 | 0.1428 | 100 |
| 0.2857 to 0.5714 | 0.4287 | 101 |
| 0.5714 to 0.8571 | 0.714 | 110 |
| 0.8571 to 1.00 | 1.000 | 111 |

**Mid-Rise Quantizer Example ($n_b = 3$, $A_{clip} = 1$)**
For $n_b = 3$, the step size $\Delta q = \frac{2}{8} = \frac{1}{4}$ (or 0.25).

| Range | Quantized Value $x_q(n)$ | Binary Output |
| :--- | :--- | :--- |
| $< -3/4$ | $-3/4$ | 000 |
| $-3/4$ to $-1/2$ | $-1/2$ | 001 |
| $-1/2$ to $-1/4$ | $-1/4$ | 010 |
| $-1/4$ to 0 | $-1/4$ | 011 |
| 0 to $1/4$ | $1/4$ | 100 |
| $1/4$ to $1/2$ | $1/4$ | 101 |
| $1/2$ to $3/4$ | $1/2$ | 110 |
| $> 3/4$ | $3/4$ | 111 |
