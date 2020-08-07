from scipy.integrate import odeint
import numpy as np
from SALib.analyze.fast import analyze as fast_analyzer
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
import matplotlib

matplotlib.rc('figure', figsize=(6, 6))


def deriv(y, t, N, R_0, gamma, delta, fm, sd):
    S, E, I, R = y
    force_of_infection = ((1 - sd) * R_0 * gamma * (1 - (0.33 * fm)) * I) / N
    dSdt = -force_of_infection * S
    dEdt = force_of_infection * S - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


N = 100820000  # Total population
D = 14  # infections lasts 14 days

# Default values
gamma = 1.0 / D
delta = 1.0 / 5.2  # incubation period of five days
R_0 = 2.6
fm = 0.0
sd = 0.0
S0, E0, I0, R0 = 0.75 * N, 0.001 * N, 1, 0  # initial conditions: 0.1% exposed
y0 = S0, E0, I0, R0  # Initial conditions vector
t = np.linspace(0, 349, 350)  # Grid of time points (in days)

# Single calculation to check the model is working
y = odeint(deriv, y0, t, args=(N, R_0, gamma, delta, fm, sd))

# number of variables to be checked is 5, bounds define the min, max bound
problem = {
    'num_vars': 5,
    'names': ['R0', 'gamma', 'delta', 'FM compliance', 'β(social distancing)'],
    'bounds': [[1.0, 5.0],
               [1 / 21, 1 / 7],
               [0.07142857, 0.5],
               [0.0, 0.9],
               [0.0, 0.5]]
}

# Generate 100000 input random samples. this algorithm run on 36GB RAM computer
vals = saltelli.sample(problem, 100000)

# initializing matrix to store output
Y = np.zeros([len(vals), 1])

Y = np.zeros([len(vals), 4])
for i in range(len(vals)):
    Y[i][:] = odeint(deriv, y0, t, args=(N, vals[i][0], vals[i][1], vals[i][2], vals[i][3], vals[i][4]))[len(y) - 1]
# Write the sample input onto .txt file
np.savetxt("param_values.txt", vals)
print(Y[:, 2])
# completing sobol analysis for each S, E, I and R
print('\n\n====S Sobol output====\n\n')
Si_X1 = sobol.analyze(problem, Y[:, 0], print_to_console=True)
print('\n\n====E Sobol output====\n\n')
Si_X2 = sobol.analyze(problem, Y[:, 1], print_to_console=True)
print('\n\n====I Sobol output====\n\n')
Si_X3 = sobol.analyze(problem, Y[:, 2], print_to_console=True)
print('\n\n====R Sobol output====\n\n')
Si_X4 = sobol.analyze(problem, Y[:, 3], print_to_console=True)

Saltelli_T, Saltelli_1st, io = Si_X3.to_df()


# plotting indices ranking
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 7))

ax1 = barplot(Saltelli_T, ax=ax1)
ax2 = barplot(Saltelli_1st, ax=ax2)
ax3 = barplot(io, ax=ax3)

plt.show()
