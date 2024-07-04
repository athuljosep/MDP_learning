import numpy as np
import matplotlib.pyplot as plt
import MDP_Module as mdp

# States
states = ['S1', 'S2']
          
# Actions
actions = ['A1', 'A2']
           
# Transition Probabilities
P = {
    ('S1', 'A1'): {'S1': 0.8, 'S2': 0.2},
    ('S1', 'A2'): {'S1': 0.5, 'S2': 0.5},
    ('S2', 'A1'): {'S1': 0.4, 'S2': 0.6},
    ('S2', 'A2'): {'S1': 0.7, 'S2': 0.3}
}

# Rewards
R = {
    ('S1', 'A1'): {'S1': 0.0, 'S2': 1.0},
    ('S1', 'A2'): {'S1': 0.0, 'S2': 1.0},
    ('S2', 'A1'): {'S1': 0.0, 'S2': 1.0},
    ('S2', 'A2'): {'S1': 0.0, 'S2': 1.0}
}

# Discount factor
gamma = 0.9

# Convergence tolerance
theta = 1e-6

# Number of iterations
n_iter = 50000

# Value iteration
V, Ans = mdp.value_iteration(states, actions, P, R, gamma, theta, n_iter)
print(Ans)

# Extracting S1 and S2 values
s1_values = [entry[0][1] for entry in Ans]
s2_values = [entry[1][1] for entry in Ans]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(s1_values, label='S1', marker='x')
plt.plot(s2_values, label='S2', marker='x')
plt.xlabel('Number of Iteration')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

##############################################
# Extracting Policy
##############################################

pi = {s: '' for s in states}

def compute_expression(a):
    return sum(P[(s, a)][s_prime] * (R[(s, a)][s_prime] + gamma * V[s_prime]) for s_prime in states)

for s in states:
    pi[s] = max(actions, key=compute_expression)

print("Policy is ", pi)