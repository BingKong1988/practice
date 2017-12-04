import numpy as np
import pprint
import sys
if '../' not in sys.path:
    sys.path.append('../')
from lib.envs.gridworld import GridworldEnv
from policy_eval import policy_eval

pp = pprint.PrettyPrinter()
env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy improvement algorithm. Iterativly evaluates and imporves a policy until an optimal policy is found.

    Args:
        env: the OpenAI environment.
        policy_eval_fn: Policy evaluation function that takes 3 argument:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """
    # Start with a ramdom policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
             return policy, V

if __name__ == '__main__':
    policy, v = policy_improvement(env)
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
