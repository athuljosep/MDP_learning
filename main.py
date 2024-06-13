import numpy as np

# what action to take?
def phi_x(state):
    if state == 'S1':
        return 'a1'
    elif state == 'S2':
        return 'a2'

# next state from action
def next_state(state, action, P, randomness_index):
    if action == 'a1':
        P_current = P['a1']

    elif action  == 'a2':
        P_current = P['a2']

    if state == 'S1':
        P_row_current = P_current[0][:]
    elif state == 'S2':
        P_row_current = P_current[1][:]
    
    if randomness_index == 1:
        state_index = P_row_current.index(max(P_row_current))
    elif randomness_index == 2:
        a = P_row_current[0]
        random_sample = np.random.uniform(0,1)
        if random_sample <= a:
            state_index = 0
        else:
            state_index = 1

    if state_index == 0:
        next_state = 'S1'
    elif state_index == 1:
        next_state = 'S2'
    
    return next_state

# states
x = ['S1', 'S2']

# actions
U = ['a1', 'a2']

# Kernal   
P = {
    'a1':[[0.1, 0.9],[0.9, 0.1]],
    'a2':[[0.2, 0.8],[0.7, 0.3]]
}

# Initial state
Initial_state = 'S1'
# 1-> deterministic 2-> non deterministic
randomness_index = 2

State_evolution = [Initial_state]
Action_evolution = []

# Simulation
for i in range(1, 100):
    if i == 1:
        state = Initial_state
    
    # Action to be taken
    action  = phi_x(state)
    Action_evolution.append(action)
    
    # Next state
    state = next_state(state, action, P, randomness_index)
    State_evolution.append(state)

print(State_evolution)
print(Action_evolution)

print(State_evolution.count('S1'))
print(State_evolution.count('S2'))