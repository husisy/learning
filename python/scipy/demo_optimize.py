import numpy as np
import scipy.optimize

def rosen(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
def rosen_grad(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    ret = np.zeros_like(x)
    ret[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    ret[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    ret[-1] = 200*(x[-1]-x[-2]**2)
    return ret
def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] - 400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp


def demo_minimize(N0=10):
    x0 = np.random.rand(N0)
    # z0.x, z0.fun
    z0 = scipy.optimize.minimize(rosen, x0, method='nelder-mead', options={'xtol':1e-8,'disp':False}) #Nelder-Mead Simplex algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='BFGS', jac=rosen_grad, options={'disp':False}) #Broyden-Fletcher-Goldfarb-Shanno
    z0 = scipy.optimize.minimize(rosen, x0, method='Newton-CG', jac=rosen_grad, hess=rosen_hess,
                options={'xtol':1e-8,'disp':False}) #Newton-Conjugate-Gradient algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='Newton-CG', jac=rosen_grad, hessp=rosen_hess_p,
                options={'xtol':1e-8,'disp':False}) #Newton-Conjugate-Gradient algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='trust-ncg', jac=rosen_grad, hess=rosen_hess,
                options={'gtol':1e-8,'disp':False}) #trust-region Newton-Conjugate-Gradient Algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='trust-ncg', jac=rosen_grad, hessp=rosen_hess_p,
                options={'gtol':1e-8,'disp':False}) #trust-region Newton-Conjugate-Gradient Algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='trust-krylov', jac=rosen_grad, hess=rosen_hess,
                options={'gtol':1e-8,'disp':False}) #trust-region truncated generalized Lanczos / Conjugate-Gradient algorithm
    z0 = scipy.optimize.minimize(rosen, x0, method='trust-krylov', jac=rosen_grad, hessp=rosen_hess_p,
                options={'gtol':1e-8,'disp':False}) #trust-region truncated generalized Lanczos / Conjugate-Gradient algorithm


def demo_constraint_minimize():
    bounds = scipy.optimize.Bounds([0,-0.5], [1,2])
    linear_constraint = scipy.optimize.LinearConstraint([[1,2],[2,1]], [-np.inf,1], [1,1])
    cons_f = lambda x: [x[0]**2 + x[1], x[0]**2-x[1]]
    cons_J = lambda x: [[2*x[0],1], [2*x[0],-1]]
    nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J)

    x0 = np.array([0.5, 0])
    res = scipy.optimize.minimize(rosen, x0, method='trust-constr', jac=rosen_grad, hess=rosen_hess,
            constraints=[linear_constraint, nonlinear_constraint],
            options={'verbose':1}, bounds=bounds)
    hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
    x_ = np.array([0.4149,0.1701])
    print('res.x: {:.4f}, {:.4f}; rel err: {:.2f}'.format(res.x[0],res.x[1],hfe(x_,res.x)))


def _root_hf0(x):
    tmp0 = x[0] + 0.5*(x[0]-x[1])**3 - 1
    tmp1 = 0.5 * (x[1] - x[0])**3 + x[1]
    ret = np.array([tmp0,tmp1])
    return ret

def _root_hf0_grad(x):
    tmp0 = 1 + 1.5 * (x[0] - x[1])**2
    tmp1 = -1.5 * (x[0] - x[1])**2
    tmp2 = -1.5 * (x[1] - x[0])**2
    tmp3 = 1 + 1.5 * (x[1] - x[0])**2
    ret = np.array([[tmp0,tmp1], [tmp2,tmp3]])
    return ret

def demo_root():
    x0 = np.array([0,0])
    ret0 = scipy.optimize.root(_root_hf0, x0, method='broyden1')
    assert np.all(np.abs(_root_hf0(ret0.x)) < 1e-5)
    ret1 = scipy.optimize.root(_root_hf0, x0, jac=_root_hf0_grad, method='hybr') #default hybr
    assert np.all(np.abs(_root_hf0(ret1.x)) < 1e-5)
