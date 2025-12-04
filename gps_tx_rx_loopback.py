#!/usr/bin/env python3
"""
gps_tx_rx_loopback.py

Transmit gpssim.bin on X310 TX and receive on X310 RX (hardware loopback).
Saves received IQ to rx_iq.bin (complex64). On Ctrl-C, it computes a
cross-correlation between a portion of the transmitted file and the captured RX
file to estimate lag and correlation strength.

CONFIGURE the USER settings below before running.
DO NOT radiate GPS signals — use a cable + attenuator (>=30 dB) or shielded box.
"""

import os
import sys
import time
import threading
import numpy as np
import uhd

# ------------------ USER SETTINGS (edit) ------------------
USR_ADDR = "addr=192.168.10.2"   # X310 IP
CHAN = 1                        # channel 0 or 1 (set to 1 for RF1)
FS = 2.5e6
FC = 1.57542e9                  # center frequency (Hz) - GPS L1
FILE_PATH = "gpssim.bin"        # input IQ file to transmit
FILE_FORMAT = "fc32"            # "fc32" (complex64) or "sc16" (interleaved int16)
LOOP_FILE = True                # loop the file while running
CHUNK_SAMPLES = 8192            # send/recv chunk size
ATTENUATOR_DB = 30              # informational only; ensure >=30 dB
TX_GAIN = 0.0                   # start low when debugging
RX_GAIN = 10.0                  # start low
OUT_RX_FILENAME = "rx_iq.bin"   # captured rx output (complex64)
CORR_SEARCH_SAMPLES = int(1e6)  # number of RX samples to use for correlation (cap)
TX_REF_SAMPLES = int(1e5)       # number of TX samples to correlate with (cap)
# ---------------------------------------------------------

def load_iq_file(path, fmt):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    if fmt == "fc32":
        data = np.fromfile(path, dtype=np.complex64)
        return data
    elif fmt == "sc16":
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size % 2 != 0:
            raise ValueError("sc16 file length not even")
        raw = raw.reshape(-1, 2)
        iq = (raw[:,0].astype(np.float32) + 1j * raw[:,1].astype(np.float32)) / 32768.0
        return iq.astype(np.complex64)
    else:
        raise ValueError("Unsupported FILE_FORMAT: use 'fc32' or 'sc16'")

def tx_thread_fn(tx_stream, iq_data, stop_event):
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False

    total = iq_data.shape[0]
    idx = 0
    first = True
    sent_chunks = 0
    try:
        while not stop_event.is_set():
            remaining = total - idx
            if remaining <= 0:
                if LOOP_FILE:
                    idx = 0
                    remaining = total
                else:
                    break
            chunk_size = min(CHUNK_SAMPLES, remaining)
            chunk = iq_data[idx: idx + chunk_size]
            if chunk.dtype != np.complex64:
                chunk = chunk.astype(np.complex64)
            tx_stream.send(chunk, md)
            if first:
                md.start_of_burst = False
                first = False
            idx += chunk_size
            sent_chunks += 1
            if (sent_chunks % 200) == 0:
                print(f"[TX] sent {sent_chunks} chunks, idx={idx}/{total}")
            # small yield - tune if underflow occurs
            time.sleep(0.0005)
    except Exception as e:
        print("TX thread exception:", repr(e))
    finally:
        md.end_of_burst = True
        try:
            tx_stream.send(np.zeros(0, dtype=np.complex64), md)
        except Exception:
            pass

def rx_thread_fn(rx_stream, stop_event, out_filename):
    md = uhd.types.RXMetadata()
    buf = np.zeros(CHUNK_SAMPLES, dtype=np.complex64)
    with open(out_filename, "wb") as fh:
        while not stop_event.is_set():
            try:
                num = rx_stream.recv(buf, md, timeout=2.0)
            except Exception as e:
                print("RX recv exception:", e)
                continue
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"[RX] metadata error: {md.error_code}")
            else:
                if num > 0:
                    fh.write(buf[:num].astype(np.complex64).tobytes())
                    rms = np.sqrt(np.mean(np.abs(buf[:num])**2))
                    print(f"[RX] {num} samples, rms={rms:.6f}")
            time.sleep(0.0005)

def fft_xcorr(a, b):
    na = a.size
    nb = b.size
    n = 1 << (int(np.ceil(np.log2(na + nb - 1))))
    A = np.fft.fft(a, n)
    B = np.fft.fft(b, n)
    corr = np.fft.ifft(B * np.conj(A))
    return corr[:(na + nb - 1)]

def analyze_correlation(tx_samples, rx_samples):
    tx = tx_samples[:min(tx_samples.size, TX_REF_SAMPLES)]
    rx = rx_samples[:min(rx_samples.size, CORR_SEARCH_SAMPLES)]

    print(f"Analyzing correlation: TX {tx.size} samples vs RX {rx.size} samples (caps applied)")

    corr = fft_xcorr(tx, rx)
    abs_corr = np.abs(corr)
    peak_idx = np.argmax(abs_corr)
    peak_val = abs_corr[peak_idx]
    lag = peak_idx - (tx.size - 1)
    E_tx = np.sum(np.abs(tx)**2)

    start = max(0, lag)
    end = start + tx.size
    if end <= rx.size:
        rx_segment = rx[start:end]
        E_rx_seg = np.sum(np.abs(rx_segment)**2)
    else:
        rx_segment = rx[start:rx.size]
        E_rx_seg = np.sum(np.abs(rx_segment)**2)

    norm_peak = peak_val / (np.sqrt(E_tx * (E_rx_seg + 1e-12)) + 1e-24)

    signal_power = np.sum(np.abs(rx_segment)**2) / max(1, rx_segment.size)
    residual = rx_segment - tx[:rx_segment.size]
    noise_power = np.sum(np.abs(residual)**2) / max(1, residual.size)
    snr_est = 10 * np.log10(max(1e-12, signal_power / (noise_power + 1e-12)))

    return {
        "peak_index": int(peak_idx),
        "lag_samples": int(lag),
        "peak_value": float(peak_val),
        "norm_peak": float(norm_peak),
        "snr_db_est": float(snr_est),
        "E_tx": float(E_tx),
    }

def main():
    print("Loading TX file:", FILE_PATH, "format:", FILE_FORMAT)
    try:
        iq_data = load_iq_file(FILE_PATH, FILE_FORMAT)
    except Exception as e:
        print("Failed to load TX file:", e)
        sys.exit(1)
    print("Loaded TX samples:", iq_data.size)

    try:
        usrp = uhd.usrp.MultiUSRP(USR_ADDR)
    except Exception as e:
        print("Failed to open USRP:", e)
        sys.exit(1)

    # configure device
    usrp.set_tx_rate(FS)
    usrp.set_rx_rate(FS)
    usrp.set_tx_freq(FC, CHAN)
    usrp.set_rx_freq(FC, CHAN)
    usrp.set_tx_gain(TX_GAIN, CHAN)
    usrp.set_rx_gain(RX_GAIN, CHAN)

    # Print available antenna names
    try:
        tx_antennas = usrp.get_tx_antennas(CHAN)
        rx_antennas = usrp.get_rx_antennas(CHAN)
        print("TX Antennas (chan {}): {}".format(CHAN, tx_antennas))
        print("RX Antennas (chan {}): {}".format(CHAN, rx_antennas))
    except Exception:
        tx_antennas = []
        rx_antennas = []
        print("Couldn't query antenna lists for channel", CHAN)

    # === HARD-SET CORRECT ANTENNA PORTS ===
    try:
        print(f"Setting TX antenna for chan {CHAN} -> TX/RX")
        usrp.set_tx_antenna("TX/RX", CHAN)
    except Exception as e:
        print("Could not set TX antenna:", e)

    try:
        print(f"Setting RX antenna for chan {CHAN} -> RX2")
        usrp.set_rx_antenna("RX2", CHAN)
    except Exception as e:
        print("Could not set RX antenna to RX2:", e)
        print("Available RX antennas:", rx_antennas)
    # ======================================

    # StreamArgs requires (cpu_format, wire_format)
    # Use 'fc32' on host and 'sc16' on the wire for X3xx family
    tx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    tx_stream_args.channels = [CHAN]
    rx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream_args.channels = [CHAN]

    tx_stream = usrp.get_tx_stream(tx_stream_args)
    rx_stream = usrp.get_rx_stream(rx_stream_args)

    # print stream channel mapping if available
    try:
        if hasattr(tx_stream, "get_channels"):
            print("TX stream channels:", tx_stream.get_channels())
    except Exception:
        pass
    try:
        if hasattr(rx_stream, "get_channels"):
            print("RX stream channels:", rx_stream.get_channels())
    except Exception:
        pass

    stop_event = threading.Event()
    tx_thread = threading.Thread(target=tx_thread_fn, args=(tx_stream, iq_data, stop_event), daemon=True)
    rx_thread = threading.Thread(target=rx_thread_fn, args=(rx_stream, stop_event, OUT_RX_FILENAME), daemon=True)

    print("Starting TX & RX. Ensure TX -> attenuator -> RX cable is connected (attenuator >= {} dB)".format(ATTENUATOR_DB))
    tx_thread.start()
    rx_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received — stopping threads...")
        stop_event.set()
        tx_thread.join(timeout=2.0)
        rx_thread.join(timeout=2.0)

    # After stopping, analyze captured file
    if not os.path.exists(OUT_RX_FILENAME):
        print("No RX file saved:", OUT_RX_FILENAME)
        return

    rx_data = np.fromfile(OUT_RX_FILENAME, dtype=np.complex64)
    print("Captured RX samples:", rx_data.size)
    if rx_data.size == 0:
        print("RX file empty — no signal captured.")
        return

    # select TX reference snippet for correlation (start of file)
    tx_ref = iq_data[:min(iq_data.size, TX_REF_SAMPLES)]
    # analyze correlation (tx_ref vs rx_data)
    print("Computing cross-correlation (this may need memory/time depending on sizes)...")
    result = analyze_correlation(tx_ref, rx_data)
    print("\n=== CORRELATION SUMMARY ===")
    print(f"Peak index (corr array): {result['peak_index']}")
    print(f"Lag (samples, RX = TX shifted by): {result['lag_samples']}")
    print(f"Normalized peak (0..1 scale, 1 strong match): {result['norm_peak']:.6f}")
    print(f"Estimated SNR (dB): {result['snr_db_est']:.2f}")
    print("===========================")
    print("Saved RX to:", OUT_RX_FILENAME)

if __name__ == "__main__":
    main()
