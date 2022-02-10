import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
plt.ion()


t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2*np.pi*7*t) + scipy.signal.gausspulse(t-0.4, fc=2)
cwtmatr = scipy.signal.cwt(sig, scipy.signal.ricker, widths=np.arange(1,31))
fig,ax = plt.subplots()
ax.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

fig,ax = plt.subplots()
ax.plot(t, sig, label='fig')
for i in range(5):
    ax.plot(t, cwtmatr[i], ':', label=str(i))
ax.legend()
