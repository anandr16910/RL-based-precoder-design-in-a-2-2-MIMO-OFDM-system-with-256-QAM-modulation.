Here is the extracted text and formulas from the handwritten notes on non-linear RF circuits, formatted in Markdown:

### **1-dB Compression Point & Saturation (Single-Tone)**
*   The top left includes a graph of Output vs. Input for a non-linear (NL) system, marking the linear region, the "1 dB CP" (compression point), and the "Saturation point".
*   Variables marked on the graph include $A_{in\_sat}$, $A_{1dBcp}$, and the "Intercept point $P_{1dBcp}$".
*   The equation for finding the saturation point is given by taking the derivative: $\frac{d}{dA} (\alpha_1 A - \frac{3}{4} \alpha_3 A^3) = 0$.
*   Solving this gives: $\alpha_1 - \frac{9}{4} \alpha_3 A^2 = 0 \Rightarrow A = \pm \sqrt{\frac{4}{9} (\frac{\alpha_1}{\alpha_3})} = \pm \frac{2}{3} \sqrt{\frac{\alpha_1}{\alpha_3}}$.
*   This results in the saturation power: $P_{in\_sat} = \frac{2}{9} (\frac{\alpha_1}{\alpha_3})$.
*   A logarithmic ratio is calculated: $10 \log_{10} \left( \frac{P_{in\_sat}}{P_{1dBcp}} \right) = 10 \log_{10} \left( \frac{(2/9)(\alpha_1/\alpha_3)}{0.0725 (\alpha_1/\alpha_3)} \right) = 4.9 \text{ dB}$.
*   The notes summarize this relationship: "$P_{in-sat}$ is above $P_{1dBcP}$ by $\approx 5 \text{ dB}$" and "$P_{1dBcP}$ is below $P_{in-sat}$ by $\approx 5 \text{ dB}$".
*   An additional note states: "Backoff referenced sometimes w.r.t $P_{in-sat}$ or $P_{1dBcP}$".

### **1 dB Compression Point for 2-Tone Case**
*   The input signal is defined as: $x(t) = A_1 \cos(\omega_1 t) + A_2 \cos(\omega_2 t)$, which is labeled as equation 1, showing a Desired Signal 'DS' and an Interfering Signal.
*   The output non-linear polynomial is: $y(t) = \alpha_1 x(t) + \alpha_2 x^2(t) - \alpha_3 x^3(t)$.
*   Substituting $x(t)$ into the polynomial yields: $y(t) = \alpha_1 [A_1 \cos(\omega_1 t) + A_2 \cos(\omega_2 t)] + \alpha_2 [A_1 \cos(\omega_1 t) + A_2 \cos(\omega_2 t)]^2 - \alpha_3 [A_1 \cos(\omega_1 t) + A_2 \cos(\omega_2 t)]^3$.
*   The notes indicate this expansion results in "13 terms" and labels this as equation 2.

### **Table of Frequencies and Amplitudes (The 13 Terms)**
The document provides a table mapping the 13 resulting terms by their frequency and amplitude:

| Sl no. | Freq. | Amplitude | Notes |
| :--- | :--- | :--- | :--- |
| 1 | DC | $\frac{1}{2} \alpha_2 (A_1^2 + A_2^2)$ | |
| 2 | $\omega_1$ (1 dB CP) DS | $\alpha_1 A_1 - \frac{3}{2} \alpha_3 (\frac{A_1^3}{2} + A_1 A_2^2)$ | Effect of 3rd order NL on fundamentals |
| 3 | $\omega_2$ IS | $\alpha_1 A_2 - \frac{3}{2} \alpha_3 (\frac{A_2^3}{2} + A_2 A_1^2)$ | |
| 4 | $2\omega_1$ | $\frac{\alpha_2 A_1^2}{2}$ | |
| 5 | $2\omega_2$ | $\frac{\alpha_2 A_2^2}{2}$ | |
| 6 | $3\omega_1$ | $-\frac{\alpha_3 A_1^3}{4}$ | |
| 7 | $3\omega_2$ | $-\frac{\alpha_3 A_2^3}{4}$ | |
| 8 | $\omega_1 + \omega_2$ | $\alpha_2 A_1 A_2$ | IM terms begin |
| 9 | $\omega_2 - \omega_1$ (*) (IP2) | $\alpha_2 A_1 A_2$ | Intercept point IP2; Effect of 2nd order NL on IMD |
| 10 | $2\omega_1 + \omega_2$ | $-\frac{3}{4} \alpha_3 A_1^2 A_2$ | |
| 11 | $2\omega_1 - \omega_2$ (*) (IP3) | $-\frac{3}{4} \alpha_3 A_1^2 A_2$ | IP3: Effect of 3rd order NL on IM |
| 12 | $2\omega_2 + \omega_1$ | $-\frac{3}{4} \alpha_3 A_1 A_2^2$ | |
| 13 | $2\omega_2 - \omega_1$ | $-\frac{3}{4} \alpha_3 A_1 A_2^2$ | |

*(Note: All rows and data represented in the table are sourced from the document.)*

### **Block Diagram and IMD Spectrum**
*   A block diagram shows two input tones ($A_1$ at $f_1$, $A_2$ at $f_2$) passing through a Non-Linear Device (NLD) to produce $y(t)$.
*   A frequency spectrum graph illustrates the resulting output, highlighting the "Inband" fundamentals and the various intermodulation products (e.g., DC, $f_2-f_1$, $2f_1-f_2$, $2f_2-f_1$, $2f_1$, $2f_2$, $3f_1$).
*   Annotations point to "Zero-IF receivers" and note that the extraneous peaks are "IMD due to NL".
*   A final equation at the bottom for the 1 dB compression point for the two-tone case is written as: $P = 20 \log \left( \frac{\alpha_1 A_1 - \frac{3}{2}\alpha_3(\frac{A_1^3}{2} + A_1 A_2^2)}{\alpha_1 A_1} \right)$.
