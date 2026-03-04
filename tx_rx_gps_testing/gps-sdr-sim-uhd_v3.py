#!/usr/bin/env python3

import sys
import time
import argparse
import os
from gnuradio import gr, blocks, uhd

class TxRxTop(gr.top_block):

    def __init__(self, dev_addr, samp_rate, freq, tx_file, rx_out,
                 tx_ant, rx_ant, tx_gain, rx_gain, scale):

        gr.top_block.__init__(self, "gps_sdr_sim_txrx")

        # -----------------------------------
        # Determine number of samples
        # -----------------------------------
        file_size = os.path.getsize(tx_file)

        # int16 I + int16 Q = 4 bytes per complex sample
        num_complex_samples = file_size // 4

        print("TX file size:", file_size, "bytes")
        print("Complex samples in file:", num_complex_samples)

        # -----------------------------------
        # File Source (INT16)
        # -----------------------------------
        self.file_src = blocks.file_source(
            itemsize=gr.sizeof_short,
            filename=tx_file,
            repeat=False
        )

        # Separate I and Q
        self.deint = blocks.deinterleave(gr.sizeof_short)

        # -----------------------------------
        # Convert int16 -> float
        # -----------------------------------
        self.short_to_float_I = blocks.short_to_float(1, 1.0)
        self.short_to_float_Q = blocks.short_to_float(1, 1.0)

        # Scale to [-1,1]
        self.scale_I = blocks.multiply_const_ff(scale)
        self.scale_Q = blocks.multiply_const_ff(scale)

        self.float_to_complex = blocks.float_to_complex()

        # -----------------------------------
        # USRP Devices
        # -----------------------------------
        self.usrp_sink = uhd.usrp_sink(
            device_addr=dev_addr,
            stream_args=uhd.stream_args(
                cpu_format="fc32",
                channels=[0]
            ),
        )

        self.usrp_src = uhd.usrp_source(
            device_addr=dev_addr,
            stream_args=uhd.stream_args(
                cpu_format="fc32",
                channels=[0]
            ),
        )

        # Clock
        self.usrp_sink.set_clock_source("internal")
        self.usrp_src.set_clock_source("internal")

        # Sync time
        self.usrp_sink.set_time_now(uhd.time_spec(0.0))
        self.usrp_src.set_time_now(uhd.time_spec(0.0))

        # RF parameters
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_src.set_samp_rate(samp_rate)

        self.usrp_sink.set_center_freq(freq, 0)
        self.usrp_src.set_center_freq(freq, 0)

        self.usrp_sink.set_gain(tx_gain, 0)
        self.usrp_src.set_gain(rx_gain, 0)

        self.usrp_sink.set_antenna(tx_ant, 0)
        self.usrp_src.set_antenna(rx_ant, 0)

        # -----------------------------------
        # HEAD blocks (auto stop)
        # -----------------------------------
        self.tx_head = blocks.head(gr.sizeof_gr_complex, num_complex_samples)
        self.rx_head = blocks.head(gr.sizeof_gr_complex, num_complex_samples)

        # -----------------------------------
        # File sink
        # -----------------------------------
        self.file_sink = blocks.file_sink(
            itemsize=gr.sizeof_gr_complex,
            filename=rx_out
        )

        # -----------------------------------
        # Connections
        # -----------------------------------

        self.connect(self.file_src, self.deint)

        # I branch
        self.connect((self.deint,0), self.short_to_float_I)
        self.connect(self.short_to_float_I, self.scale_I)
        self.connect(self.scale_I, (self.float_to_complex,0))

        # Q branch
        self.connect((self.deint,1), self.short_to_float_Q)
        self.connect(self.short_to_float_Q, self.scale_Q)
        self.connect(self.scale_Q, (self.float_to_complex,1))

        # TX chain
        self.connect(self.float_to_complex, self.tx_head)
        self.connect(self.tx_head, self.usrp_sink)

        # RX chain
        self.connect(self.usrp_src, self.rx_head)
        self.connect(self.rx_head, self.file_sink)

        print("TX sample rate:", self.usrp_sink.get_samp_rate())
        print("RX sample rate:", self.usrp_src.get_samp_rate())


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dev-addr",
        default="addr=192.168.10.2")

    parser.add_argument("--samp-rate",
        type=float,
        default=2500000)

    parser.add_argument("--freq",
        type=float,
        default=1575420000)

    parser.add_argument("--tx-file",
        default="gpssim_twopointfive.bin")

    parser.add_argument("--rx-out",
        default="gps_rx.dat")

    parser.add_argument("--tx-ant",
        default="TX/RX")

    parser.add_argument("--rx-ant",
        default="RX2")

    parser.add_argument("--tx-gain",
        type=float,
        default=20)

    parser.add_argument("--rx-gain",
        type=float,
        default=20)

    # correct scaling for int16
    parser.add_argument("--scale",
        type=float,
        default=0.000030517578125)   # 1 / 32768

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("\n=== gps_sdr_sim_txrx ===")
    print("Device:", args.dev_addr)
    print("Sample rate:", args.samp_rate)
    print("Frequency:", args.freq)
    print("TX file:", args.tx_file)
    print("RX output:", args.rx_out)
    print("TX gain:", args.tx_gain)
    print("RX gain:", args.rx_gain)

    tb = TxRxTop(
        args.dev_addr,
        args.samp_rate,
        args.freq,
        args.tx_file,
        args.rx_out,
        args.tx_ant,
        args.rx_ant,
        args.tx_gain,
        args.rx_gain,
        args.scale
    )

    try:

        tb.start()
        tb.wait()

        print("\nFlowgraph finished successfully")

    except KeyboardInterrupt:

        print("\nStopping...")
        tb.stop()
        tb.wait()

    except Exception as e:

        print("Error:", e)
        tb.stop()
        tb.wait()
        sys.exit(1)