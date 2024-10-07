import numpy as np
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(1234)

# define a prior estimate of the mean value of x:
xp = 3

# define a variance for the prior distribution of x:
sigma_x = 1

# we take a measurement of some variable x with gaussian error:
sigma_z = 1
z = np.random.normal(xp, sigma_z)

# calculate the mean and variance of the posterior distribution:
x_bar = ((sigma_x)/(sigma_x + sigma_z)) * z + ((sigma_z)/(sigma_x + sigma_z)) * xp
sigma = ((1/(sigma_z)) + (1/(sigma_x)))**(-1)

# calculate the prior, sensor, and posterior distributions
x = np.linspace(xp-3.5*sigma_x, xp+3.5*sigma_x, 1000)
prior = np.exp(-np.square(x-xp)/2*(sigma_x))/(np.sqrt(2*np.pi*(sigma_x)))
sensor = np.exp(-np.square(x-z)/2*(sigma_z))/(np.sqrt(2*np.pi*(sigma_z)))
posterior = np.exp(-np.square(x-x_bar)/2*(sigma))/(np.sqrt(2*np.pi*(sigma)))

# plot
plt.plot(x, prior, color='red')
plt.plot(x, sensor, color='blue')
plt.plot(x, posterior, color='purple')
plt.plot(z, 0, 'x')
plt.legend(["prior distribution", "sensor distribution", "posterior distribution", "measurement"], loc="upper left")
plt.savefig('../figs/ch1ex1.png')

plt.show()
