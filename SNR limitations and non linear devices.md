
# Board Notes: SNR Limitations and Non-Linear Devices (18 Nov 2020)

## Part 1: SNR Limitation Due to Quantization Noise
* [cite_start]**Core Concept:** Signal-to-Quantization Noise Ratio (SQNR) is limited by the quantization error/noise introduced during the Analog-to-Digital conversion process[cite: 234, 236, 245].
* [cite_start]**Key Variables:** $b_q$, $A_{clip}$ (clipping amplitude), and $n_b$ (number of bits)[cite: 237].
* [cite_start]**Quantization Step Size ($\Delta q$):** Defined as $\Delta q = \frac{2A_{clip}}{2^{n_b}-1}$ or approximately $\Delta q = \frac{2A_{clip}}{2^{n_b}}$ assuming $2^{n_b} \gg 1$[cite: 238, 239].
* [cite_start]**Noise Distribution:** The quantization error follows a uniform noise distribution between $-\frac{\Delta q}{2}$ and $\frac{\Delta q}{2}$[cite: 241, 242].

**Mathematical Derivation for SQNR:**

[cite_start]The average quantization noise power ($O_N$) is calculated by integrating over the uniform distribution[cite: 243]:
[cite_start]$$O_N = \int_{-\frac{\Delta q}{2}}^{\frac{\Delta q}{2}} \frac{1}{\Delta q} x^2 dx = \frac{\Delta q^2}{12}$$ [cite: 247, 249]

If $P_s$ is the average signal power, the ratio of Signal Power to Quantization Noise Power is:
[cite_start]$$\frac{P_s}{O_N} = \frac{P_s}{\left(\frac{\Delta q^2}{12}\right)} = \frac{12P_s}{\Delta q^2}$$ [cite: 250, 251]

[cite_start]By substituting $\Delta q$ and simplifying, we get the SQNR in decibels (dB)[cite: 252, 253]:
[cite_start]$$(SQNR)_{dB} \approx 10 \cdot 2n_b \log_{10}(2) + 4.76 - 10 \log_{10}\left(\frac{A_{clip}^2}{P_s}\right)$$ [cite: 253]
[cite_start]$$(SQNR)_{dB} \approx 6.02n_b + 4.76 - 10 \log_{10}\left(\frac{A_{clip}^2}{P_s}\right)$$ [cite: 255]
[cite_start]*Note: This results in the standard rule of thumb of roughly "6 dB per bit" for data converters[cite: 257]. Furthermore, the ratio $\frac{A_{clip}^2}{P_s}$ relates to Peak-to-Average Power Ratio (PAPR)[cite: 258, 262].*

### Sine Wave Case
[cite_start]For a standard sine wave, the signal power is $P_s = \frac{A_{clip}^2}{2}$, which means the ratio $\frac{A_{clip}^2}{P_s} = 2$ (or 3 dB)[cite: 265, 270].
Substituting this back into the SQNR equation yields:
[cite_start]$$SQNR \approx 6.02n_b + 1.76$$ [cite: 271]

### Oversampling Ratio (OSR)
[cite_start]When oversampling is applied, the OSR is defined as $OSR = \frac{F_s}{2f_m}$[cite: 272, 276].
The SQNR equation expands to include the oversampling gain:
[cite_start]$$SQNR \approx 6.02n_b + 4.76 - 10 \log_{10}\left(\dots\right) + 10 \log_{10}(OSR)$$ [cite: 282]

---

## Part 2: Receiver (Rx) Noise Margin and Degradation
* [cite_start]**Noise Sources:** The total noise floor is a combination of thermal noise (from the analog front end) and quantization noise (from the ADC)[cite: 283, 284, 285].
* [cite_start]**Thermal Noise Power:** Calculated as $kTB$, or $FGkTB$ when accounting for the Gain ($G$) and Noise Figure ($F$) of the Analog Front End (AFE)[cite: 287, 288, 297].

**Block Diagram Context:**
> [cite_start]**[ AFE (Gain: G, Noise Figure: F) ]** $\longrightarrow$ **[ ADC (adds Quantization Noise $N_2$) ]** $\longrightarrow$ Digital Data [cite: 297, 298, 303]

**Degradation in Rx Sensitivity:**
[cite_start]Total Noise ($N_{o2}$) equals the amplified thermal noise ($N_{o1}$) plus the ADC quantization noise ($N_2$)[cite: 302, 312]:
[cite_start]$$N_{o2} = FGkTB + N_2$$ [cite: 313]

The noise margin (degradation ratio) is defined as:
[cite_start]$$\frac{N_{o2}}{N_{o1}} = \frac{FGkTB + N_2}{FGkTB} = 1 + \frac{N_2}{N_{o1}}$$ [cite: 314, 315]

**Plot Data (Degradation $D_{dB}$ vs Noise Margin $\delta_{dB}$):**
[cite_start]The notes contain a plot showing how the margin ($\delta_{dB}$) impacts total degradation ($D_{dB}$)[cite: 328, 329]:
* If $\delta = -9$ dB $\rightarrow$ $D_{dB} = 3$ dB [cite: 332, 333]
* [cite_start]If $\delta = -6$ dB $\rightarrow$ $D_{dB} = 0.97$ dB [cite: 334, 335]
* If $\delta = -12$ dB $\rightarrow$ $D_{dB} = 0.26$ dB [cite: 337, 338]

---

## Part 3: Non-Linear Devices (NLD) and 1-dB Compression Point
* [cite_start]**System Models:** Devices can be characterized as linear or non-linear[cite: 345, 360, 361].
    * [cite_start]**Linear Model:** $y(t) = \alpha_1 x(t)$, where $\alpha_1$ is the slope[cite: 355, 356].
    * **Non-Linear Model (NLD):** Includes higher-order terms. $y(t) = \alpha_1 x(t) + \alpha_2 x^2(t) - \alpha_3 x^3(t)$[cite: 361, 363].

**Characterizing Non-Linearity (Single-Tone Test):**
To find the 1 dB Compression Point (1dB-CP), we input a single-tone signal $x(t) = A \cos(\omega_1 t)$ into the NLD model[cite: 367, 369, 377].
$$y(t) = \alpha_1 A \cos(\omega_1 t) + \alpha_2 A^2 \cos^2(\omega_1 t) - \alpha_3 A^3 \cos^3(\omega_1 t)$$ [cite: 378]

By expanding the terms, the amplitude of the fundamental frequency component becomes:
$$\text{Fundamental Amplitude} = \left(\alpha_1 A - \frac{3}{4} \alpha_3 A^3\right) \cos(\omega_1 t)$$ [cite: 388]

**Power Comparison:**
* [cite_start]**Ideal Linear Power:** $\frac{\alpha_1^2 A^2}{2}$ [cite: 385]
* **Actual Non-Linear Power:** $\frac{\left(\alpha_1 A - \frac{3}{4} \alpha_3 A^3\right)^2}{2}$ [cite: 390]

**Calculating the 1 dB Compression Point:**
The 1 dB compression point is the input amplitude ($A$) where the actual fundamental power is exactly 1 dB less than the ideal extrapolated linear power[cite: 391, 394].
$$20 \log_{10} \left( \frac{\alpha_1 A - \frac{3}{4} \alpha_3 A^3}{\alpha_1 A} \right) = -1$$ [cite: 395]
$$20 \log_{10} \left( 1 - \frac{3}{4} \frac{\alpha_3}{\alpha_1} A^2 \right) = -1$$ [cite: 396, 399]

Solving for $A^2$:
$$A^2 \approx 0.145 \left(\frac{\alpha_1}{\alpha_3}\right)$$ [cite: 400]

The amplitude at the 1 dB compression point ($A_{1dB}$) is:
$$A_{1dB} = \sqrt{0.145 \left(\frac{\alpha_1}{\alpha_3}\right)}$$ [cite: 405]

**Visual Plot Description (1 dB Compression Point):**
The notes depict an Output vs. Input power plot. A straight line represents the ideal linear response ($\alpha_1$). A second curve follows the linear line initially but then "droops" downwards due to the $-\alpha_3$ term. The point where the actual power curve falls exactly 1 dB below the ideal linear line is marked as the 1 dB Compression Point (CP)[cite: 403, 404, 407, 409]. This dictates how much "backoff" is required for operation[cite: 410, 413].
