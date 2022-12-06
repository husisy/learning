import cvxpy
import numpy as np
import scipy.optimize

import torch


class DummyModel(torch.nn.Module):
    def __init__(self, vecX, vecU, beta=1e-4):
        super().__init__()
        N0, N1 = vecX.shape
        self.vecX = torch.tensor(vecX, dtype=torch.float64)
        self.vecU = torch.tensor(vecU, dtype=torch.float64)
        self.vecU_norm = self.vecU / torch.linalg.norm(self.vecU)
        np_rng = np.random.default_rng()
        self.beta = beta
        self.lambda_ = torch.nn.Parameter(torch.tensor(np_rng.uniform(0, 1, size=N0), dtype=torch.float64))
        self.alpha = torch.nn.Parameter(torch.tensor([0], dtype=torch.float64))

    def forward(self):
        tmp0 = torch.nn.functional.relu(self.lambda_)
        tmp1 = (tmp0 / torch.sum(tmp0)) @ self.vecX
        tmp2 = torch.dot(tmp1, self.vecU_norm) / torch.linalg.norm(tmp1)
        loss_angle = torch.sqrt(1-tmp2) #remove nan in gradident
        # loss_angle = torch.nn.functional.smooth_l1_loss(torch.arccos(tmp2), torch.tensor(0.0), beta=self.beta)
        # loss_L1 = torch.abs(self.alpha[0]*self.vecU - tmp1).sum()
        # loss_L1 = torch.nn.functional.smooth_l1_loss(self.alpha[0]*self.vecU, tmp1, beta=self.beta)
        loss_L1 = torch.abs(self.alpha[0]*self.vecU - tmp1).sum()
        loss_alpha = - self.alpha[0]
        return loss_angle, loss_L1, loss_alpha

    def forward_theta(self, theta):
        for x in [self.lambda_,self.alpha]:
            if x.grad is not None:
                x.grad.zero_()
        self.lambda_.data[:] = torch.tensor(theta[:-1], dtype=torch.float64)
        self.alpha.data[:] = torch.tensor(theta[-1:], dtype=torch.float64)
        ret = self()
        return ret

    def minimize_angle(self, lambda_=None):
        # TODO arccos is diverge at theta=0, replace with minimize L1-norm at this step
        def hf0(theta):
            assert np.all(np.logical_not(np.isnan(theta)))
            theta = np.concatenate([theta, self.alpha.detach().numpy()])
            loss_angle, loss_L1, loss_alpha = self.forward_theta(theta)
            loss_angle.backward()
            grad = np.nan_to_num(np.concatenate([self.lambda_.grad.detach().numpy()]), nan=0)
            return loss_angle.item(),grad
        if lambda_ is None:
            theta0 = np.concatenate([self.lambda_.detach().numpy()])
        else:
            theta0 = lambda_
        bounds = [(0,None) for _ in range(len(self.lambda_))]
        theta_optim = scipy.optimize.minimize(hf0, theta0, method='L-BFGS-B', jac=True, tol=1e-10, bounds=bounds)
        return theta_optim

    def maximize_alpha(self, lambda_=None, alpha=None, factor=1):
        def hf0(theta):
            assert np.all(np.logical_not(np.isnan(theta)))
            loss_angle, loss_L1, loss_alpha = self.forward_theta(theta)
            loss = loss_alpha + factor*loss_L1
            loss.backward()
            grad = np.nan_to_num(np.concatenate([self.lambda_.grad.detach().numpy(), self.alpha.grad.detach().numpy()]), nan=0)
            return loss.item(),grad
        if lambda_ is None:
            lambda_ = self.lambda_.detach().numpy()
        else:
            lambda_ = np.asarray(lambda_).reshape(-1)
        if alpha is None:
            alpha = self.alpha.detach().numpy()
        else:
            alpha = np.asarray(alpha).reshape(-1)
        theta0 = np.concatenate([lambda_, alpha], dtype=np.float64)
        bounds = [(0,None) for _ in range(len(self.lambda_)+1)]
        theta_optim = scipy.optimize.minimize(hf0, theta0, method='L-BFGS-B', jac=True, tol=1e-10, bounds=bounds)
        return theta_optim


np_rng = np.random.default_rng()

N0 = 20
N1 = 6
vecX = np_rng.normal(size=(N0,N1))
vecU = np_rng.normal(size=(N1))
# vecX = from_pickle('vecX')
# vecU = from_pickle('vecU')

cvxLambda = cvxpy.Variable(N0)
cvxAlpha = cvxpy.Variable()
objective = cvxpy.Maximize(cvxAlpha)
constraints = [
    vecU*cvxAlpha==(cvxLambda@vecX),
    cvxLambda>=0,
    cvxpy.sum(cvxLambda)==1,
    cvxAlpha>=0,
]
prob = cvxpy.Problem(objective, constraints)
prob.solve(verbose=True)
print('alpha:', cvxAlpha.value)
print('lambda:', cvxLambda.value)

# FAIL
for _ in range(5):
    model = DummyModel(vecX, vecU, beta=1e-3)
    lambda_i = None
    alpha_i = None
    for _ in range(4):
        z0 = model.minimize_angle(lambda_=lambda_i)
        lambda_i = z0.x
        z1 = model.maximize_alpha(factor=10, lambda_=lambda_i, alpha=alpha_i)
        lambda_i = z1.x[:-1]
        alpha_i = z1.x[-1:]
    print(cvxAlpha.value, model.alpha.item(), [x.detach().numpy().item() for x in model()])
