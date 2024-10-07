import numpy as np

# set some true state; in this case x has target type 1.
possible_states = [0, 1, 2]
x_true = 0

# set a sensor model. this models a sensor that is good at determining
# if a target is present, but not good at identifying what kind of target
# it is.
sensor_model = np.array([[.45, .45, .1],
                         [.45, .45, .1],
                         [.1, .1, .8]])

# define a prior. in this case it is uniform, meaning we guess with equal
# probability that there is a target of type 1, type 2, or no target.
prior = np.array([1/3, 1/3, 1/3])

# take a measurement
measurement_prob = sensor_model[x_true]
measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]

# get a posterior based on our measurement, in the form 
# p(x|z) = a * p(z|x) * p(x)
posterior = (1 / sum(sensor_model[measurement] * prior)) * (sensor_model[measurement] * prior)
print(f"posterior after single measurement: {posterior}")

# take a second measurement
measurement_prob = sensor_model[x_true]
measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]

# set the new prior to be the posterior gained by the previous measurement
prior = posterior

# get new posterior
posterior = (1 / sum(sensor_model[measurement] * prior)) * (sensor_model[measurement] * prior)
print(f"posterior after single measurement: {posterior}")

# an extension of my own devising. we make the sensor _slightly_ better at
# distinguishing between targets of type 1 and 2. then recursively update with
# successive measurements. this results in a higher and higher confidence over
# time, since we hold the true state constant.
num_iterations = 10
print(f"now running the better sensor model for {num_iterations} measurements")
better_sensor_model = np.array([[.5, .4, .1],
                                [.4, .5, .1],
                                [.1, .1, .8]])
prior = np.array([1/3, 1/3, 1/3])
for i in range(num_iterations):
    # take a measurement
    measurement_prob = better_sensor_model[x_true]
    measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]
    
    # get a posterior based on our measurement, in the form 
    # p(x|z) = a * p(z|x) * p(x)
    posterior = (1 / sum(better_sensor_model[measurement] * prior)) * (better_sensor_model[measurement] * prior)
    print(f"posterior after {i} measurements: {posterior}")
    prior = posterior

