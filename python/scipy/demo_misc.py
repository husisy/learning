import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import sympy
# plt.ion()


def demo_pade_approximation():
    hf0 = lambda x: np.sin(6*x) / x
    s_x = sympy.symbols('x')
    expr = sympy.sin(6*s_x) / s_x
    taylor_order = 8
    z0 = sympy.series(expr, s_x, x0=0, n=taylor_order+1).removeO().as_coefficients_dict()
    taylor_coefficient = np.array([float(z0[s_x**i]) for i in range(taylor_order+1)][::-1])
    data_x = np.linspace(-1,1,100)

    fig,ax = plt.subplots()
    ax.plot(data_x, hf0(data_x), ':', linewidth=3, label='sin(6*x)/x')
    for q_order in [8,6,4,2,0]:
        hf_p, hf_q = scipy.interpolate.pade(taylor_coefficient[::-1], q_order)
        tmp0 = '$P_{}/Q_{}$'.format(taylor_order-q_order, q_order)
        ax.plot(data_x, hf_p(data_x) / hf_q(data_x), label=tmp0)
    ax.legend()
    ax.set_title('Pade approximation')


def demo_type_conversion():
    # https://numpy.org/doc/stable/reference/ufuncs.html#casting-rules
    mark = {False: ' -', True: ' Y'}
    ntypes = np.typecodes['All']
    print('X ' + ' '.join(ntypes))
    for row in ntypes:
        print(row, end='')
        for col in ntypes:
            print(mark[np.can_cast(row, col)], end='')
        print()
