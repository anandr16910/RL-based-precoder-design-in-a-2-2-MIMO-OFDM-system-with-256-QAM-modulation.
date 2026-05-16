# TSTP Scripts for Wi‑Fi 6 and WLAN Security Testing

TSTP-style scripts and procedures relevant to WLAN and Wi‑Fi 6 security validation. They demonstrate Linux scripting, packet capture automation, authentication validation, WPA2/WPA3 security testing, log analysis, and Wi‑Fi traffic inspection.

---

# 1. WPA3 Authentication Validation

## Objective
Verify successful WPA3-SAE authentication.

## Bash Script
```bash
#!/bin/bash

LOGFILE="/var/log/hostapd.log"

echo "Starting hostapd AP..."
sudo systemctl start hostapd

sleep 5

echo "Starting Wi‑Fi client connection..."
nmcli dev wifi connect WIFI6_TEST password "StrongPassword123"

sleep 10

grep "SAE authentication completed" $LOGFILE

if [ $? -eq 0 ]; then
    echo "[PASS] WPA3 Authentication Successful"
else
    echo "[FAIL] WPA3 Authentication Failed"
fi
```

---

# 2. WLAN Deauthentication Attack Detection

## Objective
Detect excessive deauthentication frames.

## Python Script
```python
from scapy.all import *

count = 0


def detect_deauth(pkt):
    global count
    if pkt.haslayer(Dot11Deauth):
        count += 1
        print(f"Deauthentication frame detected: {count}")

sniff(iface="wlan0mon", prn=detect_deauth)
```

## Expected Result
- Monitor mode captures deauthentication attacks.
- Excessive frames may indicate DoS attempts.

---

# 3. Wi‑Fi 6 OFDMA Traffic Validation

## Objective
Verify OFDMA-based traffic scheduling.

## tshark Script
```bash
#!/bin/bash

tshark -i wlan0 -Y "wlan.fc.type_subtype == 0x28" -a duration:30 > ofdma_capture.txt

grep "Trigger frame" ofdma_capture.txt

if [ $? -eq 0 ]; then
    echo "[PASS] OFDMA Trigger Frames Observed"
else
    echo "[FAIL] OFDMA Operation Not Observed"
fi
```

---

# 4. Rogue Access Point Detection

## Objective
Identify unauthorized APs broadcasting known SSIDs.

## Python Script
```python
from scapy.all import *

KNOWN_BSSID = "00:11:22:33:44:55"


def detect_rogue(pkt):
    if pkt.haslayer(Dot11Beacon):
        ssid = pkt.info.decode(errors="ignore")
        bssid = pkt.addr2

        if ssid == "CorporateWiFi" and bssid != KNOWN_BSSID:
            print(f"[ALERT] Rogue AP detected: {bssid}")

sniff(iface="wlan0mon", prn=detect_rogue)
```

---

# 5. WPA2/WPA3 Handshake Capture Validation

## Objective
Verify successful 4-way handshake exchange.

## Bash Script
```bash
#!/bin/bash

tcpdump -i wlan0mon -w handshake.pcap
```

## Wireshark Filter
```text
wlan_rsna_eapol
```

## Expected Result
- EAPOL handshake packets observed.
- Secure key negotiation validated.

---

# 6. WLAN Throughput Stress Test

## Objective
Validate AP stability under high traffic.

## iperf3 Script
```bash
#!/bin/bash

iperf3 -c 192.168.1.1 -t 60 -P 10
```

## Expected Result
- Stable throughput maintained.
- No AP crash or excessive packet loss.

---

# 7. MAC Spoofing Detection

## Objective
Detect duplicate MAC addresses.

## Python Script
```python
from collections import defaultdict
from scapy.all import *

mac_tracker = defaultdict(set)


def detect_spoof(pkt):
    if pkt.haslayer(Dot11):
        mac = pkt.addr2
        ssid = pkt.info.decode(errors="ignore") if pkt.haslayer(Dot11Beacon) else "UNKNOWN"

        mac_tracker[mac].add(ssid)

        if len(mac_tracker[mac]) > 1:
            print(f"[ALERT] Possible MAC spoofing: {mac}")

sniff(iface="wlan0mon", prn=detect_spoof)
```

---

# 8. Captive Portal Bypass Test

## Objective
Verify unauthorized internet access is blocked.

## Python Script
```python
import requests

url = "http://example.com"

response = requests.get(url)

if "login" in response.text.lower():
    print("[PASS] Captive portal enforced")
else:
    print("[FAIL] Captive portal bypass possible")
```

---

# 9. WLAN Channel Congestion Monitoring

## Objective
Measure channel utilization and interference.

## Bash Script
```bash
#!/bin/bash

sudo iw dev wlan0 survey dump > channel_stats.txt

cat channel_stats.txt
```

## Expected Result
- Channel busy time observed.
- Congestion statistics available.

---

# 10. 802.11 Management Frame Protection Validation

## Objective
Verify PMF (Protected Management Frames) enforcement.

## Bash Script
```bash
#!/bin/bash

grep "ieee80211w=2" /etc/hostapd/hostapd.conf

if [ $? -eq 0 ]; then
    echo "[PASS] PMF Mandatory Enabled"
else
    echo "[FAIL] PMF Not Enabled"
fi
```

---

# 11. DNS Spoofing Detection in WLAN

## Objective
Detect suspicious DNS responses.

## Python Script
```python
from scapy.all import *

TRUSTED_DNS = "8.8.8.8"


def dns_check(pkt):
    if pkt.haslayer(DNSRR):
        if pkt[IP].src != TRUSTED_DNS:
            print(f"[ALERT] Suspicious DNS response from {pkt[IP].src}")

sniff(filter="udp port 53", prn=dns_check)
```

---

# 12. Wi‑Fi 6 MU‑MIMO Validation

## Objective
Validate MU‑MIMO operation in Wi‑Fi 6 AP.

## tshark Script
```bash
#!/bin/bash

tshark -i wlan0 -Y "wlan_radio.phy == 11ax" -a duration:20 > mu_mimo.txt

cat mu_mimo.txt
```

## Expected Result
- 802.11ax PHY frames observed.
- Multi-user transmission activity verified.

---

# Common Wi‑Fi/WLAN Testing Tools

| Tool | Purpose |
|---|---|
| Wireshark | Packet analysis |
| tshark | CLI packet capture |
| tcpdump | Traffic capture |
| aircrack-ng | WLAN security auditing |
| Scapy | Packet crafting |
| hostapd | Access point daemon |
| iperf3 | Throughput testing |
| iw / iwconfig | Wireless configuration |
| nmcli | NetworkManager CLI |

---

# Typical WLAN Security Areas Tested

- WPA2/WPA3 authentication
- SAE validation
- PMF protection
- Deauthentication attack resilience
- Rogue AP detection
- Channel congestion
- DNS spoofing
- Captive portal enforcement
- OFDMA/MU‑MIMO validation
- Throughput stability

---

# Summary

“WLAN TSTP automation combines packet-level analysis, Linux scripting, authentication validation, traffic inspection, and Wi‑Fi protocol testing to ensure secure and standards-compliant wireless network operation.”

