<!DOCTYPE html>
<html lang="en">
<!-- Palette: Energetic & Playful (Deep Navy #0A2342, Brilliant Blue #1768AC, Vibrant Coral #FF8C61, Cyan #25CED1, Light Gray #F4F7F6) -->
<!-- Plan: 
  1. Header: Title and introduction to Signal Processing deep dive.
  2. Section 1 (Quantization): Bar chart showing the 6dB/bit rule for SQNR.
  3. Section 2 (Receiver Noise): Flexbox block diagram of the Rx flow and a line chart of Noise Margin vs. Degradation.
  4. Section 3 (Non-Linear Devices): Line chart comparing Ideal vs. Actual power to illustrate the 1-dB Compression Point.
-->
<!-- Visualizations:
  - SQNR Growth: Bar Chart (Goal: Change/Compare). Best for showing discrete bit levels vs SQNR. (Chart.js, NO SVG).
  - Rx Flow: Block Diagram (Goal: Organize). Best for showing signal path. (HTML/Tailwind, NO SVG/Mermaid).
  - Rx Degradation: Line Chart (Goal: Relationship). Best for continuous curve of margin vs degradation. (Chart.js, NO SVG).
  - 1-dB Compression: Line Chart with 2 datasets (Goal: Compare/Relationship). Best for ideal vs actual droop. (Chart.js, NO SVG).
-->
<!-- Confirmation: NEITHER Mermaid JS NOR SVG were used anywhere in this output. -->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Processing Essentials: SNR & Non-Linearity</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            background-color: #F4F7F6;
            color: #0A2342;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            height: 40vh;
            max-height: 400px;
        }
        @media (min-width: 768px) {
            .chart-container {
                height: 350px;
            }
        }
        .stat-card {
            background: linear-gradient(135deg, #1768AC 0%, #0A2342 100%);
        }
        .nld-card {
            background: linear-gradient(135deg, #FF8C61 0%, #D84A1B 100%);
        }
    </style>
</head>
<body class="antialiased p-4 md:p-8">

    <div class="max-w-6xl mx-auto space-y-12">
        
        <header class="text-center py-12 border-b-4 border-[#25CED1]">
            <h1 class="text-4xl md:text-6xl font-extrabold tracking-tight text-[#0A2342] mb-4">Signal Processing Essentials</h1>
            <p class="text-xl md:text-2xl text-[#1768AC] font-light max-w-3xl mx-auto">An analysis of SNR Limitations, Receiver Noise Margins, and Non-Linear Devices based on physical system constraints.</p>
        </header>

        <section class="bg-white rounded-2xl shadow-lg p-6 md:p-10 border-l-8 border-[#1768AC]">
            <div class="mb-8">
                <h2 class="text-3xl font-bold text-[#0A2342] mb-4">1. The Quantization Bottleneck</h2>
                <p class="text-lg text-gray-700 leading-relaxed mb-6">
                    In Analog-to-Digital conversion, the Signal-to-Quantization Noise Ratio (SQNR) is fundamentally limited by the quantization error introduced by the system. The quantization error follows a uniform noise distribution. For a standard sine wave, mathematical derivation proves a core engineering rule of thumb: each additional bit of resolution provides roughly 6 dB of SQNR improvement.
                </p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="stat-card rounded-xl p-8 text-white shadow-md text-center">
                        <h3 class="text-xl text-[#25CED1] font-semibold mb-2">Core SQNR Equation (Sine Wave)</h3>
                        <div class="text-3xl md:text-4xl font-mono font-bold mt-4">
                            SQNR &approx; 6.02 n<sub>b</sub> + 1.76 dB
                        </div>
                        <p class="text-sm text-blue-200 mt-4">Where n<sub>b</sub> is the number of bits.</p>
                    </div>
                    <div class="bg-gray-50 rounded-xl p-6 border border-gray-200">
                        <h3 class="text-lg font-bold text-[#0A2342] mb-2">Key Variables</h3>
                        <ul class="space-y-3 text-gray-700">
                            <li><span class="font-bold text-[#FF8C61]">&#9654;</span> <strong>b<sub>q</sub></strong>: Number of quantization levels</li>
                            <li><span class="font-bold text-[#FF8C61]">&#9654;</span> <strong>A<sub>clip</sub></strong>: Maximum clipping amplitude</li>
                            <li><span class="font-bold text-[#FF8C61]">&#9654;</span> <strong>&Delta;q</strong>: Quantization Step Size &approx; 2A<sub>clip</sub> / 2<sup>n<sub>b</sub></sup></li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="mt-8">
                <h3 class="text-xl font-bold text-center mb-6 text-[#1768AC]">SQNR Growth vs. ADC Bit Resolution</h3>
                <div class="chart-container">
                    <canvas id="sqnrChart"></canvas>
                </div>
                <p class="text-center text-sm text-gray-500 mt-4">Demonstrating the linear 6.02 dB increase per bit.</p>
            </div>
        </section>

        <section class="bg-white rounded-2xl shadow-lg p-6 md:p-10 border-l-8 border-[#25CED1]">
            <div class="mb-8">
                <h2 class="text-3xl font-bold text-[#0A2342] mb-4">2. Receiver (Rx) Noise Margin & Degradation</h2>
                <p class="text-lg text-gray-700 leading-relaxed mb-6">
                    A receiver's total noise floor is cumulative. It combines the amplified thermal noise from the Analog Front End (AFE) and the quantization noise added by the ADC. The margin (&delta;) between these two noise sources dictates the overall degradation (D) in receiver sensitivity.
                </p>

                <div class="bg-gray-100 rounded-xl p-6 mb-8 text-center">
                    <h3 class="text-lg font-bold text-[#0A2342] mb-4">Receiver Signal Flow</h3>
                    <div class="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-8 font-mono text-sm md:text-base">
                        <div class="bg-white border-2 border-[#1768AC] p-4 rounded-lg shadow-sm w-full md:w-auto">
                            <strong>[ AFE ]</strong><br>
                            Gain: G, Noise Fig: F<br>
                            <span class="text-gray-500 text-xs">Thermal Noise: N<sub>o1</sub></span>
                        </div>
                        <div class="text-[#FF8C61] font-bold text-2xl rotate-90 md:rotate-0">&#10140;</div>
                        <div class="bg-white border-2 border-[#25CED1] p-4 rounded-lg shadow-sm w-full md:w-auto">
                            <strong>[ ADC ]</strong><br>
                            Quantization Noise<br>
                            <span class="text-gray-500 text-xs">Adds Noise: N<sub>2</sub></span>
                        </div>
                        <div class="text-[#FF8C61] font-bold text-2xl rotate-90 md:rotate-0">&#10140;</div>
                        <div class="bg-[#0A2342] text-white p-4 rounded-lg shadow-sm w-full md:w-auto">
                            <strong>Digital Data</strong><br>
                            Total Noise: N<sub>o2</sub>
                        </div>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 items-center">
                <div class="lg:col-span-1 space-y-6">
                    <div class="bg-blue-50 p-5 rounded-lg border border-blue-100">
                        <h4 class="font-bold text-[#1768AC] mb-2">Degradation Ratio</h4>
                        <p class="text-sm text-gray-700">N<sub>o2</sub> / N<sub>o1</sub> = 1 + (N<sub>2</sub> / N<sub>o1</sub>)</p>
                    </div>
                    <ul class="space-y-4 text-sm text-gray-700">
                        <li class="flex justify-between items-center p-3 bg-white shadow-sm rounded">
                            <span>Margin &delta; = -12 dB</span>
                            <strong class="text-[#25CED1]">D &approx; 0.26 dB</strong>
                        </li>
                        <li class="flex justify-between items-center p-3 bg-white shadow-sm rounded">
                            <span>Margin &delta; = -6 dB</span>
                            <strong class="text-[#FF8C61]">D &approx; 0.97 dB</strong>
                        </li>
                        <li class="flex justify-between items-center p-3 bg-white shadow-sm rounded">
                            <span>Margin &delta; = 0 dB</span>
                            <strong class="text-[#d9534f]">D = 3.00 dB</strong>
                        </li>
                    </ul>
                </div>
                <div class="lg:col-span-2">
                    <h3 class="text-xl font-bold text-center mb-4 text-[#1768AC]">Degradation (D) vs Noise Margin (&delta;)</h3>
                    <div class="chart-container">
                        <canvas id="degradationChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section class="bg-white rounded-2xl shadow-lg p-6 md:p-10 border-l-8 border-[#FF8C61]">
            <div class="mb-8">
                <h2 class="text-3xl font-bold text-[#0A2342] mb-4">3. Non-Linear Devices (NLD) & Compression</h2>
                <p class="text-lg text-gray-700 leading-relaxed mb-6">
                    Real-world amplifiers and devices are not perfectly linear. A Non-Linear Device (NLD) model includes higher-order terms (e.g., y(t) = &alpha;<sub>1</sub>x(t) + &alpha;<sub>2</sub>x<sup>2</sup>(t) - &alpha;<sub>3</sub>x<sup>3</sup>(t)). The <strong>1-dB Compression Point (P1dB)</strong> is the critical input amplitude where the actual fundamental output power droops exactly 1 dB below the ideal extrapolated linear power due to these higher-order negative terms.
                </p>
                <div class="nld-card rounded-xl p-6 text-white shadow-md text-center max-w-2xl mx-auto mb-8">
                    <h3 class="text-lg font-semibold mb-2">1-dB Compression Amplitude</h3>
                    <div class="text-2xl font-mono font-bold mt-2">
                        A<sub>1dB</sub> = &radic;( 0.145 &times; (&alpha;<sub>1</sub> / &alpha;<sub>3</sub>) )
                    </div>
                </div>
            </div>

            <div>
                <h3 class="text-xl font-bold text-center mb-6 text-[#1768AC]">Ideal Linear Extrapolation vs Actual Output</h3>
                <div class="chart-container">
                    <canvas id="compressionChart"></canvas>
                </div>
                <p class="text-center text-sm text-gray-500 mt-4">The point of 1 dB droop dictates the required backoff for linear operation.</p>
            </div>
        </section>

        <footer class="text-center py-8 text-gray-500 text-sm">
            <p>Data visualization synthesized from physical signal processing models.</p>
        </footer>

    </div>

    <script>
        const wrapLabel = (label) => {
            if (label.length <= 16) return label;
            const words = label.split(' ');
            let lines = [];
            let currentLine = '';
            words.forEach(word => {
                if ((currentLine + word).length > 16) {
                    lines.push(currentLine.trim());
                    currentLine = word + ' ';
                } else {
                    currentLine += word + ' ';
                }
            });
            lines.push(currentLine.trim());
            return lines;
        };

        const globalTooltipConfig = {
            tooltip: {
                callbacks: {
                    title: function(tooltipItems) {
                        const item = tooltipItems[0];
                        let label = item.chart.data.labels[item.dataIndex];
                        if (Array.isArray(label)) {
                            return label.join(' ');
                        } else {
                            return label;
                        }
                    }
                }
            }
        };

        const bitsData = [8, 10, 12, 14, 16, 24];
        const sqnrData = bitsData.map(nb => (6.02 * nb + 1.76).toFixed(1));
        const rawSqnrLabels = bitsData.map(nb => `${nb}-Bit Resolution System`);
        const sqnrLabels = rawSqnrLabels.map(wrapLabel);

        const ctxSqnr = document.getElementById('sqnrChart').getContext('2d');
        new Chart(ctxSqnr, {
            type: 'bar',
            data: {
                labels: sqnrLabels,
                datasets: [{
                    label: 'SQNR (dB)',
                    data: sqnrData,
                    backgroundColor: '#1768AC',
                    hoverBackgroundColor: '#25CED1',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    ...globalTooltipConfig,
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'SQNR (dB)', font: { weight: 'bold' } }
                    },
                    x: {
                        title: { display: true, text: 'ADC Resolution', font: { weight: 'bold' } }
                    }
                }
            }
        });

        const marginData = [-12, -9, -6, -3, 0];
        const rawDegLabels = marginData.map(m => `Margin ${m} dB vs Thermal`);
        const degLabels = rawDegLabels.map(wrapLabel);
        const degradationData = marginData.map(delta => (10 * Math.log10(1 + Math.pow(10, delta / 10))).toFixed(2));

        const ctxDeg = document.getElementById('degradationChart').getContext('2d');
        new Chart(ctxDeg, {
            type: 'line',
            data: {
                labels: degLabels,
                datasets: [{
                    label: 'Degradation D (dB)',
                    data: degradationData,
                    borderColor: '#FF8C61',
                    backgroundColor: 'rgba(255, 140, 97, 0.2)',
                    borderWidth: 3,
                    pointBackgroundColor: '#0A2342',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    ...globalTooltipConfig,
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Degradation D (dB)', font: { weight: 'bold' } }
                    },
                    x: {
                        title: { display: true, text: 'Noise Margin \u03B4 (dB)', font: { weight: 'bold' } }
                    }
                }
            }
        });

        const pIn = [0, 5, 10, 15, 20, 22, 24, 26, 28, 30];
        const rawCompLabels = pIn.map(p => `Input Power ${p} dBm`);
        const compLabels = rawCompLabels.map(wrapLabel);
        
        const gain = 10;
        const pOutIdeal = pIn.map(p => p + gain);
        
        const pOutActual = pIn.map(p => {
            const linearLin = Math.pow(10, (p + gain) / 10);
            const cubicTerm = 0.00005 * Math.pow(Math.pow(10, p / 10), 3); 
            const actualLin = linearLin - cubicTerm;
            return actualLin > 0 ? (10 * Math.log10(actualLin)).toFixed(2) : null;
        });

        const ctxComp = document.getElementById('compressionChart').getContext('2d');
        new Chart(ctxComp, {
            type: 'line',
            data: {
                labels: compLabels,
                datasets: [
                    {
                        label: 'Ideal Linear Output',
                        data: pOutIdeal,
                        borderColor: '#25CED1',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                        tension: 0
                    },
                    {
                        label: 'Actual NLD Output',
                        data: pOutActual,
                        borderColor: '#0A2342',
                        backgroundColor: '#0A2342',
                        borderWidth: 3,
                        pointRadius: 4,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    ...globalTooltipConfig,
                    legend: { position: 'top' }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Output Power (dBm)', font: { weight: 'bold' } }
                    },
                    x: {
                        title: { display: true, text: 'Input Power (dBm)', font: { weight: 'bold' } }
                    }
                }
            }
        });
    </script>
</body>
</html>
