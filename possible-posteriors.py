from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import logistic

import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

# gamma
a = 1.99
mean, var, skew, kurt = gamma.stats(a, moments = 'mvsk')
x = np.linspace(gamma.ppf(0.01, a),
                 gamma.ppf(0.99, a), 100)
ax1.plot(x, gamma.pdf(x, a),
       'r-', lw=5, alpha=0.6, label='gamma pdf')
ax1.set_title('gamma pdf')
ax2.plot(x, gamma.cdf(x, a),
       'r-', lw=5, alpha=0.6, label='gamma cdf')
ax2.set_title('gamma cdf')

# logistic
b = 0.5
mean, var, skew, kurt = logistic.stats(b, moments = 'mvsk')
x = np.linspace(logistic.ppf(0.01, b),
                 logistic.ppf(0.99, b), 100)
ax3.plot(x, logistic.pdf(x, b),
       'g-', lw=5, alpha=0.6, label='gamma pdf')
ax3.set_title('logistic pdf')
ax4.plot(x, logistic.cdf(x, b),
       'g-', lw=5, alpha=0.6, label='gamma cdf')
ax4.set_title('logistic cdf')


# exponential
a = 1.99
mean, var, skew, kurt = expon.stats(a, moments = 'mvsk')
x = np.linspace(expon.ppf(0.01, a),
                 expon.ppf(0.99, a), 100)
ax5.plot(x, expon.pdf(x, a),
       'b-', lw=5, alpha=0.6, label='gamma pdf')
ax5.set_title('exponential pdf')
ax6.plot(x, expon.cdf(x, a),
       'b-', lw=5, alpha=0.6, label='gamma cdf')
ax6.set_title('exponential cdf')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
