"""
import numpy as np

data = np.fromfile("gps_rx.dat", dtype=np.complex64)
mag = np.abs(data)

print("Samples:", len(data))
print("Min:", mag.min())
print("Max:", mag.max())

print("Mean:", mag.mean())
print("Std:", mag.std())
"""

"""
import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile("gps_rx.dat", dtype=np.complex64)
mag = np.abs(data)

# Downsample for easier plotting (otherwise too dense)
ds = 500
mag_ds = mag[::ds]

plt.figure(figsize=(12,4))
plt.plot(mag_ds)
plt.title("Signal Magnitude Over Time (Downsampled)")
plt.xlabel("Sample Index (downsampled)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
"""
"""
import numpy as np

# Read raw int16 interleaved IQ
raw = np.fromfile("gpssim.bin", dtype=np.int16)

# Separate I and Q
I = raw[0::2].astype(np.float32)
Q = raw[1::2].astype(np.float32)

# Normalize (important!)
I /= 32768.0
Q /= 32768.0

# Combine into complex
complex_samples = I + 1j * Q

# Save as gr_complex (complex64)
complex_samples.astype(np.complex64).tofile("gpssim_complex.dat")

print("Conversion done.")
"""

import numpy as np

data = np.fromfile("gpssim_complex.dat", dtype=np.complex64)

# Take small chunk
chunk = data[:200000]

# Estimate frequency offset
phase = np.angle(chunk[1:] * np.conj(chunk[:-1]))
freq_est = np.mean(phase) * 2500000 / (2*np.pi)

print("Estimated frequency offset (Hz):", freq_est)