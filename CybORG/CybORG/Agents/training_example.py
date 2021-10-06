
from CybORG import CybORG
import inspect
import numpy as np

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

MAX_STEPS_PER_GAME = 200
MAX_EPS = 1


def run_training_example(scenario):
    print("Setup")
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    agent_name = 'Red'
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    # cyborg = OpenAIGymWrapper(agent_name=agent_name, env = CybORG(path, 'sim'))
    # cyborg = CybORG(path, "sim")
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=CybORG(path, 'sim'))
    observation = cyborg.reset(agent=agent_name)
    action_space = cyborg.get_action_space(agent_name) # 8, 3, 4, 1, 9, 2, 139, 9, 8, 1, 4
    
    
    state_ls = []
    node_ptrs = 0
    adj = []
    #exit(0)
    action_count = 0
    agent = TestAgent()
    state_ls.append(observation)
    for i in range(MAX_EPS):  # laying multiple games
        print(f"\rTraining Game: {i}", end='', flush=True)
        reward = 0
        for j in range(MAX_STEPS_PER_GAME):  # step in 1 game
        
            print(f"before: {action_space}")
            action = agent.get_action(observation, action_space)
            print(f"action taken: {type(action)} - {action}")
            next_observation, r, done, info = cyborg.step(action=action)
            
            top_state = state_ls[-1]
            if is_diff(top_state, next_observation):
                state_ls.append(next_observation)
                node_ptrs += 1
                adj.append((node_ptrs - 1, node_ptrs))
            print(f"reward: {r}")
            action_space = info.get('action_space')
            print(f"after: {action_space}")
            print("mask", info["selection_masks"])
            if r != 0:
                print("done")
                exit(0)
            reward += r

            agent.train(observation)  # training the agent
            observation = next_observation
            if done or j == MAX_STEPS_PER_GAME - 1:
                # print(f"Training reward: {reward}")
                break
        observation = cyborg.reset(agent=agent_name)
        agent.end_episode()

    print(len(state_ls))
    print(adj)
    exit(0)

def is_diff(state1, state2):
    
    diff = np.subtract(state1, state2)
    
    if np.all((diff == 0)):
        return False
    return True

if __name__ == "__main__":
    # run_training_example('Scenario1')
    scenario_name = "Scenario1"
    #scenario_name = "scenario_12_hosts_2flag"
    print(scenario_name)
    run_training_example(scenario_name)
