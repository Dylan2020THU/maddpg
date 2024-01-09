# OpenAI simple adversary using MADDPG: training
# Dylan
# 2024.1.9

import numpy as np
from make_env import make_env
from maddpg_agent import Agent, ReplayBuffer
import os
import time
import torch
import torch.nn as nn
import random

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

scenario = "simple_adversary"
env = make_env(scenario)
NUM_AGENT = env.n
obs_dim = []
for agent_i in range(NUM_AGENT):
    obs_dim.append(env.observation_space[agent_i].shape[0])
state_dim = sum(obs_dim)
action_dim = env.action_space[0].n  # scalar 5


def multi_obs_to_state(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 1e-2
BATCH_SIZE = 64

agents = []
for agent_i in range(NUM_AGENT):
    print(f"Initalizing agent {agent_i}")
    agent = Agent(obs_dim=obs_dim[agent_i], state_dim=state_dim, n_agent=NUM_AGENT,
                  action_dim=action_dim, agent_i=agent_i, alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM,
                  fc2_dims=HIDDEN_DIM, gamma=GAMMA, tau=TAU)
    agent.replay_buffer = ReplayBuffer(capacity=1000000, obs_dim=obs_dim[agent_i], state_dim=state_dim,
                                       action_dim=action_dim, n_agent=NUM_AGENT, batch_size=BATCH_SIZE)
    agents.append(agent)

# maddpg_agents = MADDPG(obs_dim, state_dim, NUM_AGENT, action_dim, alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64,
#                        gamma=0.99, tau=0.01, agent_param_path=model_path)
# memory = ReplayBuffer(1000000, state_dim, obs_dim, action_dim, NUM_AGENT, batch_size=1024)

PRINT_INTERVAL = 500
NUM_EPISODE = 10000
NUM_STEP = 25
IS_TEST = False
best_score = 0

# if IS_TEST:
#     agents.load_checkpoint()

REWARD_BUFFER = []
for episode_i in range(NUM_EPISODE):
    multi_obs = env.reset()  # shape of obs: 8*10*10
    episode_reward = 0

    for step_i in range(NUM_STEP):

        # Execute action at and observe reward rt and new state st+1
        multi_actions = []
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_i]
            single_action = agent.get_action(single_obs)  # take action based on obs
            multi_actions.append(single_action)
        multi_next_obs, multi_reward, multi_done, info = env.step(multi_actions)
        state = multi_obs_to_state(multi_obs)
        next_state = multi_obs_to_state(multi_next_obs)

        # Add memory (obs, next_obs, state, next_state, action, reward, done)
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_i]
            next_single_obs = multi_next_obs[agent_i]
            single_action = multi_actions[agent_i]  # 5 continuous actions
            single_reward = multi_reward[agent_i]
            single_done = multi_done[agent_i]
            agent.replay_buffer.add_memo(single_obs, next_single_obs, state, next_state, single_action, single_reward,
                                         single_done)

        # TODO: start learning
        # Collect next actions of all agents
        batch_multi_next_actions = []
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses, batch_next_obses, batch_states, batch_new_states, batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample()

            batch_obses_tensor = torch.tensor(batch_obses, dtype=torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses, dtype=torch.float).to(device)
            batch_single_new_action = agent.target_actor.forward(batch_next_obses_tensor)
            batch_multi_next_actions.append(batch_single_new_action)

        batch_multi_next_actions_tensor = torch.cat(batch_multi_next_actions, dim=1).to(device)

        # Update critic and actor
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses, batch_next_obses, batch_states, batch_new_states, batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample()

            batch_obses_tensor = torch.tensor(batch_obses, dtype=torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float).to(device)
            batch_next_states_tensor = torch.tensor(batch_new_states, dtype=torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones).to(device)
            # Calculate target Q value using target critic
            critic_target_value = agent.target_critic.forward(batch_next_states_tensor,
                                                              batch_multi_next_actions_tensor).flatten()
            critic_target_value = batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_value

            # Calculate current Q value using critic
            critic_value = agent.critic.forward(batch_states_tensor, batch_actions_tensor).flatten()

            # Update critic
            critic_loss = nn.MSELoss()(critic_value, critic_target_value.detach())
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            # Update actor
            actor_loss = -agent.critic.forward(batch_states_tensor, agent.actor.forward(batch_obses_tensor)).mean()
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            # Update target critic
            for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

            # Update target actor
            for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum(multi_reward)

    REWARD_BUFFER.append(episode_reward)
    print(f"Episode: {episode_i} Reward: {REWARD_BUFFER[-1] if REWARD_BUFFER else 0}")

# Save models
current_path = os.path.dirname(os.path.realpath(__file__))
model_path = current_path + '/models/' + scenario
timestamp = time.strftime("_%Y%m%d%H%M%S")
agent_path = model_path + timestamp
os.makedirs(model_path + timestamp, exist_ok=True)

for agent_i in range(NUM_AGENT):
    agent = agents[agent_i]
    agent.save_model(agent_path + f'/agent_{agent_i}_{timestamp}.pth')

# Save the rewards as txt file
reward_path = os.path.join(current_path, "model/" + scenario + f'/reward_{timestamp}.csv')
np.savetxt(current_path + f'/reward_{scenario}_{timestamp}.txt', REWARD_BUFFER)

# # Critic update
# next_actions = self.actor_target(next_states)
# target_Q = self.critic_target(next_states,        batch_obses, batch_next_obses, batch_states, batch_new_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample()
# batch_obses, batch_next_obses, batch_states, batch_new_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample()
#
# next_actions.detach())  # .detach() means the gradient won't be backpropagated to the actor
# target_Q = rewards + (GAMMA * target_Q * (1 - dones))
# current_Q = self.critic(states, actions)
# critic_loss = nn.MSELoss()(current_Q, target_Q.detach())  # nn.MSELoss() means Mean Squared Error
# self.critic_optimizer.zero_grad()  # .zero_grad() clears old gradients from the last step
# critic_loss.backward()  # .backward() computes the derivative of the loss
# self.critic_optimizer.step()  # .step() is to update the parameters
#
# # Actor update
# actor_loss = -self.critic(states,
#                           self.actor(states)).mean()  # .mean() is to calculate the mean of the tensor
# self.actor_optimizer.zero_grad()
# actor_loss.backward()
# self.actor_optimizer.step()
#
# # Update target networks
# for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#     target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
#
# for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#     target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

#
#     actions = agents.get_action(multi_obs)  # all agents take actions simultaneously
#     next_obs, reward, done, info = env.step(actions)
#     state = multi_obs_to_state(multi_obs)  # shape of state: 28=8+10+10
#     next_state = multi_obs_to_state(next_obs)
#     # store transition (st; at; rt; st+1) in R
#
#     agent.replay_buffer.add_memo(multi_obs, state, actions, reward, next_obs, next_state, done)
#     obs = next_obs
#     episode_reward += sum(reward)
#     total_steps += 1
#
#     if total_steps % 100 == 0 and not IS_TEST:
#         agents.learn(memory)
#
#     if done[0]:
#         break
#
# # todo
#
# score = 0
# done = [False] * NUM_AGENT
# episode_step = 0
# print(f"Episode: {episode_i} Reward: {REWARD_BUFFER[-1] if REWARD_BUFFER else 0}")
# while not any(done):
#     if IS_TEST:
#         env.render()
#     actions = maddpg_agents.get_actions(obs)
#     next_obs, reward, done, info = env.step(actions)
#
#     state = multi_obs_to_state(obs)
#     next_state = multi_obs_to_state(next_obs)
#
#     if episode_step > NUM_STEP:
#         done = [True] * NUM_AGENT
#
#     memory.add_memo(obs, state, actions, reward, next_obs, next_state, done)
#
#     if total_steps % 100 == 0 and not IS_TEST:
#         maddpg_agents.learn(memory)
#
#     obs = next_obs
#
#     score += sum(reward)
#     total_steps += 1
#     episode_step += 1
#
#     REWARD_BUFFER.append(score)
#     avg_score = np.mean(REWARD_BUFFER[-100:])
#     if not IS_TEST:
#         if avg_score >= best_score:
#             maddpg_agents.save_checkpoint()
#             best_score = avg_score
#     if episode_i % PRINT_INTERVAL == 0 and episode_i > 0:
#         print(f"Episode: {episode_i} Avg. Score: {avg_score}")
