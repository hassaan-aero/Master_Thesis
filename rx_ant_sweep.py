#!/usr/bin/env python3
import time, sys, numpy as np, uhd

# EDIT HERE: set channel and antenna to try
CHAN = 1           # 0 or 1
ANT  = "TX/RX"       # "RX2" or "TX/RX" or "CAL"
USR  = "addr=192.168.10.2"
FS   = 2.5e6
FC   = 1.57542e9
RX_GAIN = 20.0
BUF = 8192
TIME = 6

print("TEST START: CHAN", CHAN, "ANT", ANT)
try:
    usrp = uhd.usrp.MultiUSRP(USR)
except Exception as e:
    print("USRP open error:", e); sys.exit(1)

usrp.set_rx_rate(FS)
usrp.set_rx_freq(FC, CHAN)
usrp.set_rx_gain(RX_GAIN, CHAN)

print("Available RX antennas:", usrp.get_rx_antennas(CHAN))
try:
    usrp.set_rx_antenna(ANT, CHAN)
    print("Set antenna ->", ANT)
except Exception as e:
    print("set_rx_antenna failed:", e); sys.exit(1)

rx_args = uhd.usrp.StreamArgs("fc32","sc16")
rx_args.channels = [CHAN]
rx_stream = usrp.get_rx_stream(rx_args)
buf = np.zeros(BUF, dtype=np.complex64)
md = uhd.types.RXMetadata()

t0 = time.time()
while time.time() - t0 < TIME:
    try:
        num = rx_stream.recv(buf, md, timeout=1.0)
    except Exception as e:
        print("recv exception:", e)
        break
    print("md:", md.error_code, "num:", num)
    if md.error_code == uhd.types.RXMetadataErrorCode.none and num>0:
        print("Got samples rms=", np.sqrt(np.mean(np.abs(buf[:num])**2)))
    time.sleep(0.1)

print("TEST DONE")
