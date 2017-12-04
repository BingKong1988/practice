import numpy as np
import pprint
import sys
if '../' not in sys.path:
    sys.path.append('../')
from lib.envs.gridworld import GridworldEnv
from policy_eval import policy_eval

pp = pprint.PrettyPrinter()
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration ALgorithm.

    Args:
        env:
        theta:
        discount_factor:
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.

        Args:
            state: the state to consider (int).
            V: the value to use as an estimator, vector of length env.nS.

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state
        for s in range(env.nS):
            # Do a one-step-lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy, V

if __name__ == '__main__':
    policy, v = value_iteration(env)
    print('Policy Probability Distribution:')
    print(policy)
    print('')

    print('Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):')
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print('')

    print('Value Function:')
    print(v)
    print('')

    print('Reshaped Grid Value Function:')
    print(v.reshape(env.shape))
    print('')
