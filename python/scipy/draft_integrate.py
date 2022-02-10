import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
# plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


## example00 ivp
hf_pde = lambda t,y: -0.5*y
hf_solution = lambda t,y0: y0*np.exp(-0.5*t)

tspan = [0,5]
y0 = np.array([2,4,8])
z0 = scipy.integrate.solve_ivp(hf_pde, tspan, y0)
assert hfe(hf_solution(z0.t, y0[:,np.newaxis]), z0.y) < 1e-2

tspan = [0,5]
t_eval = np.array([0,1,2,3,4,5])
y0 = np.array([2,4,8])
z0 = scipy.integrate.solve_ivp(hf_pde, tspan, y0, t_eval=t_eval)
assert hfe(t_eval, z0.t) < 1e-5
assert hfe(hf_solution(z0.t, y0[:,np.newaxis]), z0.y) < 1e-2


## example01 ivp Lotka-Volterra equation https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
def hf_lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]
tspan = [0,15]
y0 = np.array([10,5])
args_abcd = (1.5, 1, 3, 1)
z0 = scipy.integrate.solve_ivp(hf_lotkavolterra, tspan, y0, args=args_abcd, dense_output=True)

t = np.linspace(0, 15, 300)
y = z0.sol(t)
fig,ax = plt.subplots()
ax.plot(t, y[0], label='x')
ax.plot(t, y[1], label='y')
ax.legend()


## example02 bvp Bratu's problem
hf_pde = lambda x,y: np.stack([y[1], -np.exp(y[0])])
hf_boundary = lambda ya,yb: np.array([ya[0], yb[0]])
x = np.linspace(0, 1, 5)
y_a = np.zeros((2, x.size))
y_b = np.zeros((2, x.size))
y_b[0] = 3
res_a = scipy.integrate.solve_bvp(hf_pde, hf_boundary, x, y_a)
res_b = scipy.integrate.solve_bvp(hf_pde, hf_boundary, x, y_b)

x_plot = np.linspace(0, 1, 100)
y_plot_a = res_a.sol(x_plot)[0]
y_plot_b = res_b.sol(x_plot)[0]
fig,ax = plt.subplots()
ax.plot(x_plot, y_plot_a, label='y_a')
ax.plot(x_plot, y_plot_b, label='y_b')
ax.legend()


## example03 bvp Sturm-Liouville problem
hf_pde = lambda x,y,p: np.stack([y[1], -p[0]**2 * y[0]])
hf_boundary = lambda ya,yb,p: np.array([ya[0], yb[0], ya[1] - p[0]])
x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
y[0, 1] = 1
y[0, 3] = -1
sol = scipy.integrate.solve_bvp(hf_pde, hf_boundary, x, y, p=[6])
sol.p[0] #approximate 2pi

x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]
fig,ax = plt.subplots()
ax.plot(x_plot, y_plot)
