#!/usr/bin/env python3
import time, sys, numpy as np, uhd

USR = "addr=192.168.10.2"
CHAN = 0
FS = 2.5e6
FC = 1.57542e9
RX_GAIN = 30.0
BUF = 8192
TEST_SECONDS = 6

try:
    usrp = uhd.usrp.MultiUSRP(USR)
except Exception as e:
    print("USRP open error:", e); sys.exit(1)

usrp.set_rx_rate(FS)
usrp.set_rx_freq(FC, CHAN)
usrp.set_rx_gain(RX_GAIN, CHAN)

print("RX antennas (chan):", usrp.get_rx_antennas(CHAN))
try:
    usrp.set_rx_antenna("RX2", CHAN)
    print("Set RX antenna -> RX2")
except Exception as e:
    print("Could not set RX2:", e)

rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = [CHAN]
rx_stream = usrp.get_rx_stream(rx_args)

buf = np.zeros(BUF, dtype=np.complex64)
md = uhd.types.RXMetadata()

print(f"Starting RX-only test for {TEST_SECONDS}s on CHAN={CHAN}, ANT=RX2 ...")
t0 = time.time()
while time.time() - t0 < TEST_SECONDS:
    try:
        num = rx_stream.recv(buf, md, timeout=1.0)
    except Exception as e:
        print("recv exception:", e)
        break
    print("md:", md.error_code, "num:", num)
    if md.error_code == uhd.types.RXMetadataErrorCode.none and num > 0:
        rms = np.sqrt(np.mean(np.abs(buf[:num])**2))
        print(f"Got {num} samples   rms={rms:.6e}")
    time.sleep(0.1)

print("RX-only CHAN0 test complete.")
