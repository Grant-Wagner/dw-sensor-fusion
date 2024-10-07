import numpy as np

# set seed for reproducibility
np.random.seed(1234)

# define a prior estimate of the mean value of x:
xp = 3

# define a variance for the prior distribution of x:
sigma_x = 8

# define a sensor model, and take an initial measurement
sigma_z = 8
z = np.random.normal(xp, sigma_z)

# calculate the mean and variance of the posterior distribution:
x_bar = ((sigma_x)/(sigma_x + sigma_z)) * z + ((sigma_z)/(sigma_x + sigma_z)) * xp
sigma = ((1/(sigma_z)) + (1/(sigma_x)))**(-1)
print(f"initial measurement: {z}")
print(f"initial estimate of the position based on prior and measurement: {x_bar}")
print(f"true value of position: {xp}")
print(f"new sigma: {sigma}")
print("\n")

# recursively update our estimate:
num_iterations = 100
for i in range(num_iterations):
    z = np.random.normal(xp, sigma_z)

    # calculate the mean and variance of the posterior distribution:
    x_bar = ((sigma)/(sigma + sigma_z)) * z + ((sigma_z)/(sigma + sigma_z)) * x_bar
    sigma = ((1/(sigma_z)) + (1/(sigma)))**(-1)
    
    print(f"measurement {i+1}: {z}")
    print(f"estimate of the position based on measurement: {x_bar}")
    print(f"true value of position: {xp}")
    print(f"new sigma: {sigma}")
    print("\n")
