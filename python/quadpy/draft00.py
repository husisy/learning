import numpy as np
import quadpy
import scipy.integrate

# import matplotlib.pyplot as plt
# plt.ion()


scheme = quadpy.line_segment.chebyshev_gauss_1(5)
scheme.show()
scheme.weights
scheme.points
