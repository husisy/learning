import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial

import picos


def demo_projection_onto_convex_hull():
    # https://picos-api.gitlab.io/picos/quick.html
    np_rng = np.random.default_rng(235)
    n = 20
    A = np_rng.normal(size=(2,n))
    b = np.array([2,-1])
    x = picos.RealVariable("x", n)
    prob = picos.Problem()
    prob.minimize = abs(A*x - b)
    prob += [picos.sum(x) == 1, x >= 0]
    prob.solve(solver="cvxopt")
    p = (A*x).np

    V = scipy.spatial.ConvexHull(A.T).vertices
    fig,ax = plt.subplots()
    ax.fill(A.T[V, 0], A.T[V, 1], "lightgray")
    ax.plot(A.T[:, 0], A.T[:, 1], "k.")
    ax.plot(*zip(b, p), "k.--")
    ax.annotate("$\mathrm{conv} \{a_1, \ldots, a_n\}$", [0.25, 0.5])
    ax.annotate("$b$", b + 1/100)
    ax.annotate("$Ax$", p + 1/100)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_worst_projection_onto_convex_hull():
    # https://picos-api.gitlab.io/picos/quick.html
    np_rng = np.random.default_rng(233)
    n = 20
    A = np_rng.uniform(size=(2, n))
    b = np.array([0.05, 0.2])
    # Define an ellipsoidal uncertainty set Θ and a perturbation parameter θ.
    # The perturbation is later added to the data, rendering it uncertain. ‖Lθ‖ ≤ 1.
    Theta = picos.uncertain.ConicPerturbationSet("θ", 2)
    Theta.bound(abs([[5,0], [0,10]] * Theta.element) <= 1)
    theta = Theta.compile()
    x = picos.RealVariable("x", n)
    prob = picos.Problem()
    prob.minimize = abs(A*x - (b + theta))
    prob += [picos.sum(x) == 1, x >= 0]
    prob.solve(solver="cvxopt")
    p = (A*x).np

    V = scipy.spatial.ConvexHull(A.T).vertices
    E = matplotlib.patches.Ellipse(b, 0.4, 0.2, color="lightgray", ec="k", ls="--")
    fig,ax = plt.subplots()
    ax.add_artist(E)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, None)
    ax.axis("off")
    ax.fill(A.T[V, 0], A.T[V, 1], "lightgray")
    ax.plot(A.T[:, 0], A.T[:, 1], "k.")
    ax.plot(*zip(b, p), "k.")
    ax.annotate("$\mathrm{conv} \{a_1, \ldots, a_n\}$", [0.25, 0.5])
    ax.annotate("$b$", b + 1/200)
    ax.annotate("$Ax$", p + 1/200)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
