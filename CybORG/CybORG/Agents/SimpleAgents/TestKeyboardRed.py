from CybORG import CybORG
import inspect

from CybORG.Agents.SimpleAgents.BlueMonitorAgent import BlueMonitorAgent
from CybORG.Agents.SimpleAgents.KeyboardAgent import KeyboardAgent
from CybORG.Agents.Wrappers.RedTableWrapper import RedTableWrapper

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

if __name__ == "__main__":
    print("Setup")
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    

    
    cyborg = RedTableWrapper(env=CybORG(path, 'sim',agents={'Blue':BlueMonitorAgent}), output_mode='table')
    agent_name = 'Red'
    #cyborg = RedTableWrapper(env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))), output_mode="table")

    results = cyborg.reset(agent=agent_name)
    observation = results.observation
    action_space = results.action_space

    agent = KeyboardAgent()

    reward = 0
    done = False
    while True:
        print("before:", action_space)
        action = agent.get_action(observation, action_space)
        results = cyborg.step(agent=agent_name, action=action)

        reward += results.reward
        observation = results.observation
        print("after")
        action_space = results.action_space
        print(action_space)
        break
        if done:
            print(f"Game Over. Total reward: {reward}")
            break
