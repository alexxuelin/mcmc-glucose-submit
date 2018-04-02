# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# scipy for algorithms
import scipy
from scipy import stats

# pymc3 for Bayesian Inference, pymc built on t
import pymc3 as pm
import theano.tensor as tt
import scipy
from scipy import optimize

# matplotlib for plotting
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import matplotlib

# beta  = rate param in exponential
def exponential(x, beta, alpha=0):
    return 1 - np.exp(np.dot(-beta, x) + alpha)

N_SAMPLES = 1000

diabetes_data = pd.read_csv('lunches.csv')

figsize(16, 6)

# Diabetes data
plt.scatter(diabetes_data['Offset'], diabetes_data['Indicator'],
            s= 60, alpha=0.01, facecolor = 'b', edgecolors='b')

plt.savefig("diabetes_indicator_dinner.png")
plt.clf()
plt.cla()
plt.close()


# Sort the values by time offset
diabetes_data.sort_values('Offset', inplace=True)

# Time is the time offset
time = np.array(diabetes_data.loc[:, 'Offset'])

# Observations are the indicator
diabetes_obs = np.array(diabetes_data.loc[:, 'Indicator'])

with pm.Model() as diabetes_model:
    # Create the alpha and beta parameters
    alpha = pm.Normal('alpha', mu=0.2, tau=0.01, testval=0.0)
    beta = pm.Normal('beta', mu=0.2, tau=0.01, testval=0.0)

    # Create the probability from the exponential function
    p = pm.Deterministic('p', 1 -  tt.exp(-beta * time + alpha))

    # Create the bernoulli parameter which uses the observed dat
    observed = pm.Bernoulli('obs', p, observed=diabetes_obs)

    # Starting values are found through Maximum A Posterior estimation
    # start = pm.find_MAP()

    # Using Metropolis Hastings Sampling
    step = pm.Metropolis()

    # Sample from the posterior using the sampling method
    diabetes_trace = pm.sample(N_SAMPLES, step=step, njobs=2);


	# Extract the alpha and beta samples
alpha_samples = diabetes_trace["alpha"][1000:, None]
beta_samples = diabetes_trace["beta"][1000:, None]

figsize(16, 10)

plt.subplot(211)
plt.title(r"""Distribution of $\alpha$ with %d samples""" % N_SAMPLES)

plt.hist(alpha_samples, histtype='stepfilled',
         color = 'darkred', bins=30, alpha=0.8, density=True);
plt.ylabel('Probability Density')

plt.savefig("alpha_probability_density_dinner.png")
plt.clf()
plt.cla()
plt.close()


plt.subplot(212)
plt.title(r"""Distribution of $\beta$ with %d samples""" % N_SAMPLES)
plt.hist(beta_samples, histtype='stepfilled',
         color = 'darkblue', bins=30, alpha=0.8, density=True)
plt.ylabel('Probability Density');

plt.savefig("beta_probability_density_dinner.png")
plt.clf()
plt.cla()
plt.close()

# Time values for probability prediction
time_est = np.linspace(time.min()- 15, time.max() + 15, 1e3)[:, None]

# Take most likely parameters to be mean values
alpha_est = alpha_samples.mean()
beta_est = beta_samples.mean()

# Probability at each time using mean values of alpha and beta
diabetes_est = exponential(time_est, beta=beta_est, alpha=alpha_est)

figsize(16, 6)
plt.plot(time_est, diabetes_est, color = 'navy',
         lw=3, label="Most Likely exponential Model")
plt.scatter(time, diabetes_obs, edgecolor = 'slateblue',
            s=50, alpha=0.2, label='obs')

plt.savefig("diabetes_est_dinner.png")
plt.clf()
plt.cla()
plt.close()


colors = ["#348ABD", "#A60628", "#7A68A6"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("BMH", colors)
figsize(12, 6)
probs = diabetes_trace['p']

plt.scatter(time, probs.mean(axis=0), cmap = cmap,
            c = probs.mean(axis=0), s = 50);

plt.savefig("diabetes_trace_dinner.png")
plt.clf()
plt.cla()
plt.close()

diabetes_all_est = exponential(time_est.T, beta_samples, alpha_samples)
quantiles = stats.mstats.mquantiles(diabetes_all_est, [0.025, 0.975], axis=0)

plt.fill_between(time_est[:, 0], *quantiles, alpha=0.6,
                 color='slateblue', label = '95% CI')
plt.plot(time_est, diabetes_est, lw=2, ls='--',
         color='black', label="average posterior \nprobability of diabetes")

plt.savefig("stuff_dinner.png")
plt.clf()
plt.cla()
plt.close()

def diabetes_posterior(time_offset):
    figsize(16, 8)
    prob = exponential(time_offset, beta_samples, alpha_samples)
    plt.hist(prob, bins=100, histtype='step', lw=4)
    plt.title('Probability Distribution for high blood sugar at offset %s' % time_offset)
    plt.xlabel('Probability of High Blood Sugar'); plt.ylabel('Samples')
    plt.show();

for i in range(20,27):
    diabetes_posterior(i*10)

figsize(20, 12)
pm.traceplot(diabetes_trace, ['alpha', 'beta']);
plt.show()

pm.autocorrplot(diabetes_trace, ['alpha', 'beta']);
plt.show()
