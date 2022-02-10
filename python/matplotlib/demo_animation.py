import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
plt.ion()


def demo_animation_pause():
    fig,ax = plt.subplots(figsize=(8,6), dpi=80)
    x = np.linspace(0, 2*np.pi, 100)
    for delta in np.linspace(0, 4*np.pi, 100):
        y_cos = np.cos(x + delta)
        y_sin = np.sin(x + delta)

        ax.cla()
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        ax.plot(x, y_cos, 'b--', linewidth=2.0, label='cos(x)')
        ax.plot(x, y_sin, 'g-', linewidth=2.0, label='sin(x)')
        ax.legend(loc='upper left')
        plt.pause(0.1)


def demo_animation_legend():
    fig, ax = plt.subplots()
    N0 = 20
    xdata = np.linspace(0, 2*np.pi, 100)
    omega = np.linspace(1, 4, N0)
    ydata = np.sin(omega[:,np.newaxis]*xdata)
    hline0, = ax.plot(xdata, ydata[0], label=f'omega={omega[0]:.3f}')
    hlegend = ax.legend(loc='upper right')
    def hf_frame(ind0):
        hline0.set_data(xdata, ydata[ind0])
        label_i = f'{omega[ind0]:.3f}'
        # hline0.set_label(label_i) #doesn't help
        htext = hlegend.get_texts()[0]
        htext.set_text(label_i)
        return hline0,htext
    ani = matplotlib.animation.FuncAnimation(fig, hf_frame, frames=N0, interval=200)
    return ani


def demo_FuncAnimation():
    fig, ax = plt.subplots()
    hline_cos = ax.plot([], [], lw=2, label='cos')[0]
    hline_sin = ax.plot([], [], lw=2, label='sin')[0]
    ax.grid()

    np_delta = np.linspace(0, 4*np.pi, 100)

    def hf_frame(ind0):
        xdata = np.linspace(0, 2*np.pi, 100)
        ydata0 = np.cos(xdata + np_delta[ind0])
        ydata1 = np.sin(xdata + np_delta[ind0])
        hline_cos.set_data(xdata, ydata0)
        hline_sin.set_data(xdata, ydata1)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 1)
        return hline_cos,hline_sin

    ani = matplotlib.animation.FuncAnimation(fig, hf_frame, frames=len(np_delta), interval=100)
    # ani._start()
    # ani.pause()
    # ani.resume()
    # see https://stackoverflow.com/a/43297660
    return ani #animation will stop if the variable ani is garbage-collected


def demo_double_pendulum():
    #https://matplotlib.org/stable/gallery/animation/double_pendulum.html
    gravity = 9.8 #m/s^2
    L1 = 1.0  # length of pendulum 1 in m
    L2 = 1.0  # length of pendulum 2 in m
    L = L1 + L2  # maximal length of the combined pendulum
    M1 = 1.0  # mass of pendulum 1 in kg
    M2 = 1.0  # mass of pendulum 2 in kg
    t_stop = 10  # how many seconds to simulate
    num_t_grid = 500

    def derivs(state, t):
        dydx = np.zeros_like(state)
        dydx[0] = state[1]
        delta = state[2] - state[0]
        den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
        dydx[1] = (M2*L1*state[1]*state[1]*np.sin(delta)*np.cos(delta) + M2*gravity*np.sin(state[2])*np.cos(delta)
                    + M2*L2*state[3]*state[3]*np.sin(delta) - (M1+M2)*gravity*np.sin(state[0])) / den1
        dydx[2] = state[3]
        den2 = (L2/L1) * den1
        dydx[3] = (-M2*L2*state[3]*state[3]*np.sin(delta)*np.cos(delta) + (M1+M2)*gravity*np.sin(state[0])*np.cos(delta)
                    - (M1+M2)*L1*state[1]*state[1]*np.sin(delta) - (M1+M2)*gravity*np.sin(state[2])) / den2
        return dydx

    tspan = np.linspace(0, t_stop, num_t_grid)
    state = np.radians([120, 0, -10, 0]) #theta0 (degree), angular_velocity0 (degree/second), theta1, angular_velocity1
    y = scipy.integrate.odeint(derivs, state, tspan)
    x1 = L1*np.sin(y[:, 0])
    y1 = -L1*np.cos(y[:, 0])
    x2 = L2*np.sin(y[:, 2]) + x1
    y2 = -L2*np.cos(y[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    hline0 = ax.plot([], [], 'o-', lw=2)[0]
    hline1 = ax.plot([], [], ',-', lw=1)[0]
    htext = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_x = []
    history_y = []

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        if i == 0:
            history_x.clear()
            history_y.clear()
        history_x.append(thisx[2])
        history_y.append(thisy[2])
        hline0.set_data(thisx, thisy)
        hline1.set_data(history_x, history_y)
        htext.set_text(f'time = {tspan[i]:.1f}s')
        return hline0, hline1, htext


    tmp0 = t_stop*1000/num_t_grid
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_t_grid, interval=tmp0, blit=True)
    return ani


def demo_jupyterlab_animation():
    from IPython.display import HTML
    fig, ax = plt.subplots()
    xdata = np.linspace(0, 2*np.pi, 100)
    omega = np.linspace(1, 4, 20)
    ydata = np.sin(omega[:,np.newaxis]*xdata)
    hline0, = ax.plot(xdata, ydata[0], label=f'omega={omega[0]:.3f}')
    hlegend = ax.legend(loc='upper right')
    def hf_frame(ind0):
        hline0.set_data(xdata, ydata[ind0])
        label_i = f'{omega[ind0]:.3f}'
        # hline0.set_label(label_i) #doesn't help
        htext = hlegend.get_texts()[0]
        htext.set_text(label_i)
        return hline0,htext
    ani = matplotlib.animation.FuncAnimation(fig, hf_frame, frames=len(omega), interval=200)
    plt.close(fig) #fig is not necessary anymore
    ret = HTML(ani.to_jshtml())
    return ret
