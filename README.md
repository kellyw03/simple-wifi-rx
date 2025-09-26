# Simple Wi-Fi Receiver (PHY Layer)

## Overview
This project explores the **Wi-Fi physical layer (PHY)** by building a simplified **software-based receiver**.  

- A provided transmitter (`wifitransmitter.py`) generates Wi-Fi packets from input text.  
- I implemented the receiver (`wifireceiver.py`) to **reverse the process** and recover:
  - `begin_zero_padding` → Number of zeros added in Level 4
  - `message` → Original ASCII message
  - `length` → Length of the decoded message  

While much simpler than IEEE 802.11 (real Wi-Fi), this project simulates core PHY concepts:  
- Bit interleaving  
- Turbo coding & 4-QAM modulation  
- OFDM encoding  
- Noise addition and random zero-padding  

## wifitransmitter.py
- The Wi-Fi transmitter (`wifitransmitter.py`) was provided as starter code by course staff, and is **not** included in this repo.  
- Its purpose is to simulate the Wi-Fi physical layer (PHY) transmitter pipeline by taking an input text message and progressively encoding it into a wireless signal.
- Depending on the selected level, it applies several encoding stages. Higher stages include all lower stages beneath it.
    1. Level 1 – Interleaving & Length Header: Converts message to bitstream, interleaving, and encodes message length with repetition encoding
    2. Level 2 – Error Coding & Modulation: Applies predefined preamble, convolutional encoding, and 4-QAM
    3. Level 3 – OFDM: Applies Inverse FFT, converting into OFDM time-domain signals
    4. Level 4 – Channel Effects: Adds random zero-padding at beginning and end, adds gaussian noise at specified signal-to-noise-ratio (SNR). Returns number of leading zeros added.
