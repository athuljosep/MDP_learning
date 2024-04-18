from GridWorld import GridWorld
from ValueIteration import ValueIteration

problem = GridWorld('data/world00.csv')

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

problem.visualize_value_policy(policy=solver.policy, values=solver.values)
problem.random_start_policy(policy=solver.policy)