import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# set seed for reproducibility
np.random.seed(1234)

# set sensor locations
sensor_one_location = [15, 0]
sensor_two_location = [50, 20]

# set a meshgrid for working with according to fig in chapter. we also set some
# sort of a position array - this is all to do plotting magic later on
x = np.linspace(0, 50, 1000)
y = np.linspace(0, 50, 1000)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# set a prior. I'm just making it a regular gaussian around the seeming
# actual location of the target, which is different from the book but I
# think it's fairly close
mu_prior = [15, 25]
sigma_prior = [[20, 0],
               [0, 20]]
prior = multivariate_normal(mu_prior, sigma_prior)

# get a pdf object for plotting
prior_density = prior.pdf(pos)

# set likelihood function for sensor 1. this sensor is very good at determining
# x positions and very bad at determining y positions
sigma_sensor_one = [[.1, 0],
                    [0, 60]]
sensor_one_model = multivariate_normal(mu_prior, sigma_sensor_one)

# get a pdf object for plotting
sensor_one_density = sensor_one_model.pdf(pos)

# set likelihood function for sensor 2. this sensor is very bad at determining
# x positions and very good at determining y positions
sigma_sensor_two = [[60, 0],
                    [0, .1]]
sensor_two_model = multivariate_normal(mu_prior, sigma_sensor_two)

# get a pdf object for plotting
sensor_two_density = sensor_two_model.pdf(pos)

# take a measurement from sensor 1 and calculate a posterior. note we can only
# do it this way (element wise as in ex1) because our covariance matrices are
# diagonal; I should as an exercise write all of this out in matrix math.
z1 = sensor_one_model.rvs(random_state=1234)
mu_posterior = [sigma_prior[0][0]/(sigma_prior[0][0] + sigma_sensor_one[0][0])*mu_prior[0] +
                sigma_sensor_one[0][0]/(sigma_prior[0][0] + sigma_sensor_one[0][0]) * z1[0],
                sigma_prior[1][1]/(sigma_prior[1][1] + sigma_sensor_one[1][1])*mu_prior[1] +
                sigma_sensor_one[1][1]/(sigma_prior[1][1] + sigma_sensor_one[1][1]) * z1[1]]
sigma_posterior = [[(((sigma_prior[0][0])*(sigma_sensor_one[0][0]))/((sigma_prior[0][0])+(sigma_sensor_one[0][0]))), 0],
                   [0, (((sigma_prior[1][1])*(sigma_sensor_one[1][1]))/((sigma_prior[1][1])+(sigma_sensor_one[1][1])))]]
posterior = multivariate_normal(mu_posterior, sigma_posterior)
posterior_density_one_obs = posterior.pdf(pos)
print(f"first observation from sensor one: {z1}")
print(f"sigma_posterior after one obs: {sigma_posterior}")
print(f"mu_posterior after one obs: {mu_posterior}")

# take a second measurement from sensor 1 and plot again
z2 = sensor_one_model.rvs(random_state=2345)
mu_posterior = [sigma_posterior[0][0]/(sigma_posterior[0][0] + sigma_sensor_one[0][0])*mu_posterior[0] +
                sigma_sensor_one[0][0]/(sigma_posterior[0][0] + sigma_sensor_one[0][0]) * z2[0],
                sigma_posterior[1][1]/(sigma_posterior[1][1] + sigma_sensor_one[1][1])*mu_posterior[1] +
                sigma_sensor_one[1][1]/(sigma_posterior[1][1] + sigma_sensor_one[1][1]) * z2[1]]
sigma_posterior = [[(((sigma_posterior[0][0])*(sigma_sensor_one[0][0]))/((sigma_posterior[0][0])+(sigma_sensor_one[0][0]))), 0],
                   [0, (((sigma_posterior[1][1])*(sigma_sensor_one[1][1]))/((sigma_posterior[1][1])+(sigma_sensor_one[1][1])))]]
posterior = multivariate_normal(mu_posterior, sigma_posterior)
posterior_density_two_obs = posterior.pdf(pos)
print(f"second observation from sensor one: {z2}")
print(f"sigma_posterior after two obs: {sigma_posterior}")
print(f"mu_posterior after two obs: {mu_posterior}")

# now take an observation from sensor 2 and plot
z3 = sensor_two_model.rvs(random_state=3456)
mu_posterior = [sigma_posterior[0][0]/(sigma_posterior[0][0] + sigma_sensor_two[0][0])*mu_posterior[0] +
                sigma_sensor_two[0][0]/(sigma_posterior[0][0] + sigma_sensor_two[0][0]) * z3[0],
                sigma_posterior[1][1]/(sigma_posterior[1][1] + sigma_sensor_two[1][1])*mu_posterior[1] +
                sigma_sensor_two[1][1]/(sigma_posterior[1][1] + sigma_sensor_two[1][1]) * z3[1]]
sigma_posterior = [[(((sigma_posterior[0][0])*(sigma_sensor_two[0][0]))/((sigma_posterior[0][0])+(sigma_sensor_two[0][0]))), 0],
                   [0, (((sigma_posterior[1][1])*(sigma_sensor_two[1][1]))/((sigma_posterior[1][1])+(sigma_sensor_two[1][1])))]]
posterior = multivariate_normal(mu_posterior, sigma_posterior)
posterior_density_three_obs = posterior.pdf(pos)
print(f"observation from sensor two: {z3}")
print(f"sigma_posterior after three obs: {sigma_posterior}")
print(f"mu_posterior after three obs: {mu_posterior}")


# plot prior
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, prior_density, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y)')
plt.title("prior density")
plt.savefig("../figs/ch1ex5_prior.png")
plt.show()

# plot likelihood function of sensor 1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, sensor_one_density, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y|z1)')
plt.title("likelihood function from sensor one")
plt.savefig("../figs/ch1ex5_sensor_one_likelihood.png")
plt.show()

# plot posterior after one measurement from sensor 1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, posterior_density_one_obs, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y|z1)')
plt.title("posterior after one obs from sensor one")
plt.savefig("../figs/ch1ex5_posterior1.png")
plt.show()

# plot posterior after two measurements from sensor 1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, posterior_density_two_obs, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y|z2)')
plt.title("posterior after two obs from sensor one")
plt.savefig("../figs/ch1ex5_posterior2.png")
plt.show()

# plot likelihood function for sensor 2
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, sensor_two_density, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y|z1)')
plt.title("likelihood function from sensor two")
plt.savefig("../figs/ch1ex5_sensor_two_likelihood.png")
plt.show()

# plot posterior after two measurements from sensor 1 and one measurement from
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, posterior_density_three_obs, cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P(x,y|z2)')
plt.title("posterior after two obs from sensor one and one from sensor two")
plt.savefig("../figs/ch1ex5_posterior3.png")
plt.show()
