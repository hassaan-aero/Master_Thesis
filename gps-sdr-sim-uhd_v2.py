#!/usr/bin/env python3
"""
gps_sdr_sim_txrx.py

Single-process GNURadio script that transmits an interleaved int8 IQ GPS SIM file
(gps-sdr-sim format: I,Q,I,Q,... each as int8) and simultaneously captures RX
samples from the same USRP (X310). This avoids UHD device contention.

Requirements:
  - GNU Radio with UHD installed
  - gpssim.bin in interleaved int8 IQ format
  - Run when no other UHD application (e.g., GNSS-SDR) is holding the device.

Example:
  ./gps-sdr-sim-uhd_v2.py \
    --dev-addr "addr=192.168.10.2" \
    --samp-rate 2500000 \
    --freq 1575420000 \
    --tx-file gpssim.bin \
    --rx-out gps_rx.dat \
    --tx-ant "TX/RX" \
    --rx-ant "RX2" \
    --tx-gain 20 \
    --rx-gain 30 \
    --scale 0.0078125

Notes:
  - scale defaults to 1/128 (≈0.0078125) to map int8 [-128..127] -> ~[-1..1]
  - Adjust gains for your attenuator (you mentioned 30 dB).
"""

import sys
import time
import argparse
from gnuradio import gr, blocks, uhd

class TxRxTop(gr.top_block):
    def __init__(self, dev_addr, samp_rate, freq, tx_file, rx_out,
                 tx_ant, rx_ant, tx_gain, rx_gain, scale):
        gr.top_block.__init__(self, "gps_sdr_sim_txrx")

        # --- TX side: read interleaved int8 IQ -> deinterleave -> char_to_float -> scale -> float_to_complex -> usrp_sink
        # File source reading bytes (signed char). GNU Radio file_source reads raw bytes as unsigned char by default;
        # we use gr.sizeof_char (signed/unsigned behavior in char_to_float will handle numeric mapping).
        self.file_src = blocks.file_source(gr.sizeof_char, tx_file, repeat=True)

        # Deinterleave into two streams: even = I, odd = Q (both bytes)
        self.deint = blocks.deinterleave(gr.sizeof_char)  # creates two outputs

        # Convert bytes->float for I and Q separately.
        self.char_to_float_I = blocks.char_to_float(1, 1.0)
        self.char_to_float_Q = blocks.char_to_float(1, 1.0)

        # Apply scaling to map int8 range to approx [-1, 1]
        self.scale_I = blocks.multiply_const_ff(scale)
        self.scale_Q = blocks.multiply_const_ff(scale)

        # Build complex samples from two float streams
        self.float_to_complex = blocks.float_to_complex(1)

        # USRP sink (TX)
        self.usrp_sink = uhd.usrp_sink(
            device_addr=dev_addr,
            stream_args=uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_sink.set_center_freq(freq, 0)
        self.usrp_sink.set_gain(tx_gain, 0)
        self.usrp_sink.set_antenna(tx_ant, 0)
        # Optional: set bandwidth if needed: self.usrp_sink.set_bandwidth(bw,0)

        # --- RX side: usrp_source -> file_sink
        self.usrp_src = uhd.usrp_source(
            device_addr=dev_addr,
            stream_args=uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.usrp_src.set_samp_rate(samp_rate)
        self.usrp_src.set_center_freq(freq, 0)
        self.usrp_src.set_gain(rx_gain, 0)
        self.usrp_src.set_antenna(rx_ant, 0)

        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, rx_out)
        self.file_sink.set_unbuffered(False)

        # --- Connections
        # file_src -> deint (two outputs)
        self.connect(self.file_src, self.deint)

        # deint output 0 -> char_to_float_I -> scale_I -> float_to_complex LHS
        self.connect((self.deint, 0), self.char_to_float_I)
        self.connect(self.char_to_float_I, self.scale_I)
        self.connect(self.scale_I, (self.float_to_complex, 0))

        # deint output 1 -> char_to_float_Q -> scale_Q -> float_to_complex RHS
        self.connect((self.deint, 1), self.char_to_float_Q)
        self.connect(self.char_to_float_Q, self.scale_Q)
        self.connect(self.scale_Q, (self.float_to_complex, 1))

        # float_to_complex -> usrp_sink (TX)
        self.connect(self.float_to_complex, self.usrp_sink)

        # usrp_src (RX) -> file_sink
        self.connect(self.usrp_src, self.file_sink)


def parse_args():
    p = argparse.ArgumentParser(description="Single-process TX+RX of GPS-SIM file (int8 IQ).")
    p.add_argument("--dev-addr", default="addr=192.168.10.2", help='UHD device args (e.g. "addr=192.168.10.2")')
    p.add_argument("--samp-rate", type=float, default=2500000.0, help="Sample rate (Hz)")
    p.add_argument("--freq", type=float, default=1575420000.0, help="Center frequency (Hz)")
    p.add_argument("--tx-file", default="gpssim.bin", help="Input GPS-SIM file (interleaved int8 IQ)")
    p.add_argument("--rx-out", default="gps_rx.dat", help="Output file for received complex samples (fc32 interleaved)")
    p.add_argument("--tx-ant", default="TX/RX", help="TX antenna name (exact from uhd_usrp_probe)")
    p.add_argument("--rx-ant", default="RX2", help="RX antenna name (exact from uhd_usrp_probe)")
    p.add_argument("--tx-gain", type=float, default=20.0, help="TX gain")
    p.add_argument("--rx-gain", type=float, default=30.0, help="RX gain")
    p.add_argument("--scale", type=float, default=(1.0/128.0), help="Scale factor to map int8->float (default 1/128)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("=== gps_sdr_sim_txrx ===")
    print("Device args:", args.dev_addr)
    print("Sample rate:", args.samp_rate)
    print("Center freq:", args.freq)
    print("TX file:", args.tx_file)
    print("RX out:", args.rx_out)
    print("TX ant:", args.tx_ant, "RX ant:", args.rx_ant)
    print("TX gain:", args.tx_gain, "RX gain:", args.rx_gain)
    print("Scale (int8->float):", args.scale)
    print("Make sure no other UHD app (e.g., GNSS-SDR) is running. Ctrl-C to stop.")

    tb = TxRxTop(
        dev_addr=args.dev_addr,
        samp_rate=args.samp_rate,
        freq=args.freq,
        tx_file=args.tx_file,
        rx_out=args.rx_out,
        tx_ant=args.tx_ant,
        rx_ant=args.rx_ant,
        tx_gain=args.tx_gain,
        rx_gain=args.rx_gain,
        scale=args.scale
    )

    try:
        tb.start()
        # keep running until user stops (keyboard interrupt)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
        tb.stop()
        tb.wait()
        print("Stopped.")
    except Exception as e:
        print("Error:", e)
        tb.stop()
        tb.wait()
        sys.exit(1)
