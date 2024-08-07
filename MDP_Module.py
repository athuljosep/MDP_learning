# Module for MDP functions

# Value Iteration Function
def value_iteration(states, actions, P, R, gamma, theta, n_iter):

    # Initializing values as zeros
    V = {s: 0 for s in states}
    V_history  = []

    for i in range (1, n_iter):
        
        delta = 0
        
        for s in states:
            
            v = V[s]
            V[s] = max(sum(P[(s, a)][s_prime] * (R[(s, a)][s_prime] + gamma * V[s_prime]) for s_prime in states) for a in actions)
            delta = max(delta, abs(v - V[s]))
        
        V_history.append(list(V.items()))

        if delta < theta:
            
            break
        
        i = i+1 

    V_final = V

    return V_final, V_history

# Policy Iteration Function
def policy_iteration(states, actions, P, R, gamma, theta, n_iter):
    # Initializing values as zeros
    V = {s: 0 for s in states}
    V_history  = []
    for i in range (1, n_iter):
        delta = 0
        for s in states:
            v = V[s]
            a = pi(s)
            V[s] = sum(P[(s, a)][s_prime] * (R[(s, a)][s_prime] + gamma * V[s_prime]) for s_prime in states)
            delta = max(delta, abs(v - V[s]))
        V_history.append(list(V.items()))
        if delta < theta:
            break
        i = i+1

    policy = {}
    for s in states:
        action_values = {}
        for a in actions:
            action_values[a] = sum(P[(s, a)][s_prime] * (R[(s, a)][s_prime] + gamma * V[s_prime]) 
                                   for s_prime in states)
        policy[s] = max(action_values, key=action_values.get)

    return V_history, policy