import numpy as np

# set random seed. this seed happens to give us the situation described in the
# text where we observe target type 1 from both sensors
np.random.seed(2)

# set some true state; in this case x has target type 1.
possible_states = [0, 1, 2]
x_true = 0

# set a sensor model. this models a sensor that is good at determining
# if a target is present, but not good at identifying what kind of target
# it is.
target_detector_model = np.array([[.45, .45, .1],
                                  [.45, .45, .1],
                                  [.1, .1, .8]])

# set a second sensor model. this models a sensor that is good at determining
# the type of target present, but not good at determining if an object is 
# present.
target_discriminator_model = np.array([[.45, .1, .45],
                                       [.1, .45, .45],
                                       [.45, .45, .1]])

# define a prior. in this case it is uniform, meaning we guess with equal
# probability that there is a target of type 1, type 2, or no target.
prior = np.array([1/3, 1/3, 1/3])

# take a measurement with target detector
measurement_prob = target_detector_model[x_true]
measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]

# get a posterior based on our measurement, in the form
# p(x|z) = a * p(z|x) * p(x)
posterior = (1 / sum(target_detector_model[measurement] * prior)) * (target_detector_model[measurement] * prior)
print(f"posterior after measurement from target detector: {posterior}")

# take a measurement with target discriminator
measurement_prob = target_discriminator_model[x_true]
measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]

# set the new prior to be the posterior gained by the previous measurement
prior = posterior

# get new posterior
posterior = (1 / sum(target_discriminator_model[measurement] * prior)) * (target_discriminator_model[measurement] * prior)
print(f"posterior after measurement from target discriminator: {posterior}")

# again, extending based on my own devising. we recurrently update the gained
# posterior distribution. pretty neat how confidence increases
num_iterations = 10
print(f"now running the model for {num_iterations} measurements")
prior = np.array([1/3, 1/3, 1/3])
for i in range(num_iterations):
    # take a measurement
    measurement_prob = target_detector_model[x_true]
    measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]
    
    # get a posterior based on our measurement, in the form 
    # p(x|z) = a * p(z|x) * p(x)
    posterior = (1 / sum(target_detector_model[measurement] * prior)) * (target_detector_model[measurement] * prior)
    print(f"posterior after {i} detection measurements: {posterior}")
    prior = posterior
    
    # take a measurement
    measurement_prob = target_discriminator_model[x_true]
    measurement = np.random.choice(possible_states, 1, p=list(measurement_prob))[0]
    
    # get a posterior based on our measurement, in the form 
    # p(x|z) = a * p(z|x) * p(x)
    posterior = (1 / sum(target_discriminator_model[measurement] * prior)) * (target_discriminator_model[measurement] * prior)
    print(f"posterior after {i} discrimination measurements: {posterior}")
    prior = posterior
