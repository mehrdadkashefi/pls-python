# Particle Swarm Optimization (PSO)
# Programmer: Mehrdad Kashefi

import numpy as np
import matplotlib.pyplot as plt
from pls import one_neuron_cost
# Number of time that cost function is called

NFE = 0


def cost_function(input):
    global NFE
    NFE = NFE + 1
    [cost, r2, r] = one_neuron_cost(input)

    return cost


# Number of optimization variables
num_var = 10
# Variable min and max values
var_min = 0
var_max = 10000
# Maximum number of iterations
max_iter = 5
# Number of particles
num_particle = 10

"""
# Classical Optimization parameters
w=1            # Inertia Weight
w_damp=0.99     # Inertia Weight Damping Ratio
c1=2           # Personal Learning Coefficient
c2=2           # Global Learning Coefficient
"""

# Constriction Coefficients  In this case :Velocity limit is not needed
phi1 = 2.05
phi2 = 2.05
phi = phi1+phi2
chi = 2/(phi-2+np.sqrt(phi**2-4*phi))
w = chi          # Inertia Weight
w_damp = 1        # Inertia Weight Damping Ratio
c1 = chi*phi1    # Personal Learning Coefficient
c2 = chi*phi2    # Global Learning Coefficient

# Velocity Constraint
VelMax = 0.1*(var_max - var_min)
VelMin = -VelMax


class Empty:
    pass


particle = [Empty() for i in range(num_particle)]

global_best = {'cost': np.inf,
               'position': np.random.uniform(var_min, var_max, (1, num_particle))}

for part in range(num_particle):
    # Initialize Random Position
    particle[part].position = np.random.uniform(var_min, var_max, (1, num_var))

    # Initializing Velocities
    particle[part].velocity = np.zeros((1, num_var))

    # Random Particle Evaluation

    particle[part].cost = cost_function(particle[part].position)

    # Update Personal Best

    particle[part].best_position = particle[part].position
    particle[part].best_cost = particle[part].cost

    if particle[part].best_cost < global_best['cost']:
        global_best['cost'] = particle[part].best_cost
        global_best['position'] = particle[part].best_position


cost_buffer = np.zeros((1, max_iter))
nfe = np.zeros((1, max_iter))

# PSO Loop

for iter in range(max_iter):
    for part in range(num_particle):
        # Update Velocity
        particle[part].velocity =( w * particle[part].velocity) \
                                 +(c1 * np.random.rand(1, num_var) * (particle[part].best_position - particle[part].position))\
                                 +(c2 * np.random.rand(1, num_var) * (global_best['position'] - particle[part].position))


        #Velocity Limit effect

        particle[part].velocity[particle[part].velocity > var_max] = var_max
        particle[part].velocity[particle[part].velocity < var_min] = var_min

        # Updating Particle Positions

        particle[part].position = particle[part].position + particle[part].velocity

        # Velocity Mirror Effect
        outside_sample = ((particle[part].position < var_min) | (particle[part].position > var_max))
        particle[part].velocity[outside_sample] = -particle[part].velocity[outside_sample]

        # Cost Evaluation
        particle[part].cost = cost_function(particle[part].position)

        # Update personal Best

        if particle[part].cost < particle[part].best_cost:

            particle[part].best_position = particle[part].position
            particle[part].best_cost = particle[part].cost

            # Update Global best

            if particle[part].best_cost < global_best['cost']:
                global_best['position'] = particle[part].position
                global_best['cost'] = particle[part].cost

    w = w * w_damp
    cost_buffer[0, iter] = global_best['cost']
    nfe[0, iter] = NFE

    print('In Iteration: ', iter, 'NFE: ', nfe[0, iter], 'Cost: ', cost_buffer[0, iter])


[cost, r2, r] = one_neuron_cost(global_best['position'])
print("========================")
print("Global Best:")
print("========================")
print('Best Cost: ', global_best['cost'])
print('Best Variables: ', global_best['position'])
print('Best r2: ', r2)
print('Best r: ', r)

plt.figure()
plt.semilogy(nfe[0, :], cost_buffer[0, :])
plt.xlabel("NFE")
plt.ylabel("Cost")
plt.show()
