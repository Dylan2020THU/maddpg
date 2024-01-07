import numpy as np
from make_env import make_env
from maddpg_agent import MADDPG
from maddpg_agent import ReplayBuffer
import os
import time


scenario = "simple_adversary"
env = make_env(scenario)
n_agents = env.n
actor_dims = []


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


current_path = os.path.dirname(os.path.realpath(__file__))
model_path = current_path + '/models/' + scenario
timestamp = time.strftime("%Y%m%d%H%M%S")
os.makedirs(model_path, exist_ok=True)

for agent_i in range(n_agents):
    actor_dims.append(env.observation_space[agent_i].shape[0])
critic_dims = sum(actor_dims)

n_actions = env.action_space[0].n
maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64,
                       gamma=0.99, tau=0.01, agent_param_path=model_path)
memory = ReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

PRINT_INTERVAL = 500
NUM_EPISODES = 30000
NUM_STEPS = 25
total_steps = 0
REWARD_BUFFER = []
IS_TEST = False
best_score = 0

if IS_TEST:
    maddpg_agents.load_checkpoint()

for episode_i in range(NUM_EPISODES):
    obs = env.reset()
    score = 0
    done = [False] * n_agents
    episode_step = 0
    print(f"Episode: {episode_i} Reward: {REWARD_BUFFER[-1] if REWARD_BUFFER else 0}")
    while not any(done):
        if IS_TEST:
            env.render()
        actions = maddpg_agents.get_actions(obs)
        next_obs, reward, done, info = env.step(actions)

        state = obs_list_to_state_vector(obs)
        next_state = obs_list_to_state_vector(next_obs)

        if episode_step > NUM_STEPS:
            done = [True] * n_agents

        memory.add_memo(obs, state, actions, reward, next_obs, next_state, done)

        if total_steps % 100 == 0 and not IS_TEST:
            maddpg_agents.learn(memory)

        obs = next_obs

        score += sum(reward)
        total_steps += 1
        episode_step += 1

        REWARD_BUFFER.append(score)
        avg_score = np.mean(REWARD_BUFFER[-100:])
        if not IS_TEST:
            if avg_score >= best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if episode_i % PRINT_INTERVAL == 0 and episode_i > 0:
            print(f"Episode: {episode_i} Avg. Score: {avg_score}")
