import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets

plt.ion()


def demo_RadioButtons():
    fig, ax0 = plt.subplots()
    xdata = np.linspace(0, 2*np.pi, 100)
    ydata = np.sin(xdata)
    line, = ax0.plot(xdata, ydata)
    ax1 = fig.add_axes([0.7, 0.7, 0.15, 0.15])
    button = matplotlib.widgets.RadioButtons(ax1, ('red', 'blue', 'green'), active=0)
    def hf_select_button(label):
        line.set_color(label)
        fig.canvas.draw()
    button.on_clicked(hf_select_button)


def demo_Slider():
    fig, ax0 = plt.subplots()
    xdata = np.linspace(0, 2*np.pi, 100)
    ydata = np.sin(xdata)
    line, = ax0.plot(xdata, ydata)
    ax0.set_ylim(-1.3, 1.3)
    ax1 = fig.add_axes([0.3, 0.15, 0.55, 0.03])
    slider_frequency = matplotlib.widgets.Slider(ax1, 'frequency', 0.1, 10, valinit=1)
    def hf_update_frequency(freq):
        line.set_ydata(np.sin(freq*xdata))
        fig.canvas.draw()
    slider_frequency.on_changed(hf_update_frequency)
