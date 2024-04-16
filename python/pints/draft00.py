import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import scipy.optimize

import pints
import pints.plot

plt.ion()
np_rng = np.random.default_rng()

k = 1.5
y0 = 1
times = np.linspace(0,10,50)

# A one-compartment PK model
def onecomp(t, y, k):
    dydt = -k * y
    return dydt


def simulate(func, parameters, y0, times):
    tmp0 = np.array(y0).reshape(-1)
    ret = scipy.integrate.solve_ivp(func, t_span=(times[0], times[-1]), y0=tmp0, t_eval=times, args=(parameters,)).y.reshape(-1)
    return ret

actual_values = simulate(onecomp, k, y0, times)


## Bayesian optimisation
# Make noisy data that we're inferring from. noisy_data is known to us.
noise = np_rng.normal(0, 0.03, len(actual_values))
noisy_data = actual_values + noise


def scalar_to_minimise(parameters):
    y_model = simulate(onecomp, parameters, 1, times)
    ret = np.mean((y_model - noisy_data)**2)
    return ret

result = scipy.optimize.minimize_scalar(scalar_to_minimise)
recon_model = simulate(onecomp, result.x, 1, times)


class PintsOneComp(pints.ForwardModel):
    def n_parameters(self):
        return 1
    def simulate(self, parameter, times):
        hf0 = lambda y,t,k: -k*y
        y0 = 1
        ret = scipy.integrate.odeint(hf0, y0, times, (parameter,)).reshape(-1)
        return ret

problem = pints.SingleOutputProblem(PintsOneComp(), times, noisy_data) # Create a model instance with measured data
error_measure = pints.SumOfSquaresError(problem) # Define the error measure to be used
optimisation = pints.OptimisationController(error_measure, [1], method=pints.XNES) # Define a statistical problem
optimisation.set_log_to_screen(False) # Suppress log output
parameters, error = optimisation.run() # Run the statistical model

fig,ax = plt.subplots()
ax.plot(times, noisy_data, '.', label='Measured values')
ax.plot(times, recon_model, '--', label='Custom inferred values')
ax.plot(times, PintsOneComp().simulate(parameters, times), '--', lw=2, label='Pints inferred values')
ax.legend()


## Bayesian sampling

class OdeModel:
    def __init__(self, thetas, covariates, prior, likelihood, modeltype):
        self.thetas = thetas
        self.covariates = covariates
        self.modeltype = modeltype
        self.prior = prior
        self.likelihood = likelihood

def uniform_prior(theta):
    """Returns 0.1 if entire input list is between 0 & 10, else 0"""
    tmp0 = [(0.1 if ((0<v) and (v<10)) else 0) for v in theta.values()]
    ret = min(tmp0)
    return ret

def likelihood_k(theta, y_data):
    """Returns the likelihood, P(theta|y)"""
    k = theta['k']
    sigma = 0.03
    pdf = []
    y_model = simulate(onecomp, k, 1, times)
    other_bit = 1/(2*np.pi*sigma**2)
    for t in range(len(y_data)): # this loop gives a normally distributed pdf
        square_error = (y_data[t] - y_model[t])**2
        exponential = np.exp(-square_error/(2*sigma**2))
        pdf.append(exponential*other_bit)
    return np.prod(pdf)


def propose_new_theta(model, y_data, theta):
    """Randomly proposes a new theta and decides whether to accept or not
    In
    model: instance of OdeModel class
    y_data: list with experimental data
    theta: parameters, in a list

    Out: new parameters, either the same (if proposed not accepted) or different
    """

    numerator = model.prior(theta) * model.likelihood(theta, y_data)

    # randomly get a proposed theta & calculate its numerator
    proposed_theta = {}
    for key, value in theta.items():
        proposed_k = np.random.normal(value, model.covariates[key])
        proposed_theta[key] = proposed_k
    proposed_numerator = model.prior(proposed_theta) * model.likelihood(proposed_theta, y_data)

    # if the new numerator should be accepted (metropolis hastings criteria), replace theta
    if proposed_numerator == 0:
        pass
    elif proposed_numerator > numerator:
        theta = proposed_theta
        numerator = proposed_numerator
    elif np.random.rand() < proposed_numerator/numerator:
        theta = proposed_theta
        numerator = proposed_numerator
    return theta

def metropolis_singlethread(model, y_data, threadnum, max_iters):
    for _ in range(max_iters):
        theta = propose_new_theta(model, y_data, model.thetas[threadnum][-1])
        model.thetas[threadnum].append(theta)

def metropolishastings(model, y_data, blocksize, number_of_blocks):
    for _ in range(number_of_blocks):
        for threadnum, thetas_onelot in enumerate(model.thetas):
            metropolis_singlethread(model, y_data, threadnum, blocksize)


ks = np.linspace(0,10,100)
likelihoods = [likelihood_k({'k':n}, noisy_data) for n in ks]

fig,ax = plt.subplots()
ax.plot(ks, likelihoods)
ax.set_xlabel('input parameter, k')
ax.set_ylabel('likelihood')
ax.axvline(1.5, color='k', label='True value of k')




thetas_k = [[{'k':5}], [{'k':3}], [{'k':1}]] # Three initial guesses for k
covariates_k = {'k':0.05} # Step size (SD of normal distribution for choosing next proposed theta)
model = OdeModel(thetas_k, covariates_k, uniform_prior, likelihood_k, onecomp)
metropolishastings(model, noisy_data, 10, 100)

fig,ax = plt.subplots()
for n in range(len(model.thetas)):
    ks_list= [theta['k'] for theta in model.thetas[n]]
    ax.plot(ks_list[:500]) # only first 500
ax.set_xlabel('iteration #')
ax.set_ylabel('k')

all_ks = [[x['k'] for x in theta] for theta in model.thetas]
fig,ax = plt.subplots()
ax.hist(all_ks, bins=100, stacked=True)
ax.set_xlabel('k')
ax.set_ylabel('occurrence')


log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=0.05) # Define & wrap a physical model
startpoints = [[1],[3],[5]] # Start 3 Markov chains from arbitrary points
mcmc = pints.MCMCController(log_likelihood, 3, startpoints, method=pints.HaarioBardenetACMC) # Define a statistical problem
mcmc.set_max_iterations(2000) # Set number of iterations to attempt
mcmc.set_log_to_screen(False) # Suppress log output
samples = mcmc.run() # Run the statistical model

pints.plot.trace(samples)

pints.plot.series(np.vstack(samples[:,1000:]), problem)
