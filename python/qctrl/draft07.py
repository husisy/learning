# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-integrate-boulder-opal-with-qua-from-quantum-machines
import dotenv
import numpy as np
import matplotlib.pyplot as plt

import qctrl
import qctrlvisualizer
# from qctrl import Qctrl
import qctrlqua

# from qctrlvisualizer import get_qctrl_style

plt.style.use(qctrlvisualizer.get_qctrl_style())
QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)

# Create Gaussian pulse.
total_rotation = np.pi  # π pulse
max_rabi_rate = 2 * np.pi * 50e6  # rad/s
duration = total_rotation * 10 / max_rabi_rate / np.sqrt(2 * np.pi)

control_pulse = QCTRL_HANDLE.signals.gaussian_pulse(
    amplitude=max_rabi_rate, drag=0.1 * duration, duration=duration
)

# Resample the pulse to match the hardware time resolution.
dt = 1e-9  # s
resampled = control_pulse.export_with_time_step(dt)

# Map the optimal pulse into input amplitudes for the hardware.
rabi_rate = 2 * np.pi * 300e6  # rad/s
qctrl_i = np.real(resampled) / rabi_rate
qctrl_q = np.imag(resampled) / rabi_rate

# Plot resulting arrays.
fig,ax = plt.subplots()
ax.plot(qctrl_i, label="$I$")
ax.plot(qctrl_q, label="$Q$")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Voltage (V)")
ax.legend()

qm_config = {}
# Add the Q-CTRL robust pulse to the QUA configuration.
pulse_name = "Q-CTRL"
channel_name = "drive_channel"  # drive channel name in qm_config
qm_config = qctrlqua.add_pulse_to_config(
    pulse_name, channel_name, qctrl_i, qctrl_q, qm_config
)

# Use the updated qm_config dictionary to run QUA on the hardware.
