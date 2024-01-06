import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import gym
# from map_env.make_env import make_env
from make_env import make_env
import pyglet


# print(np.__version__)
# print(torch.__version__)
# print(gym.__version__)
# print(pyglet.__version__)
#
# env = make_env('simple_adversary')
# print('number of agents', env.n)
# print("observation space,", env.observation_space)
# print("action space", env.action_space)
# print("n actions", env.action_space[0].n)
#
# observation = env.reset()
# print(observation)
#
# tmp_a = np.array([1,0,0,0,0])
#
# actions = [tmp_a,tmp_a,tmp_a]
# next_obs, reward, done, info = env.step(actions)
# print(reward)
# print(done)

class ReplayBuffer:
    def __init__(self, capacity, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.capacity = capacity
        self.counter = 0
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.actor_dims = actor_dims
        self.state_cap = np.empty((self.capacity, critic_dims))
        self.new_state_cap = np.empty((self.capacity, critic_dims))
        self.reward_cap = np.empty((self.capacity, self.n_agents))
        self.done_cap = np.empty((self.capacity, self.n_agents), dtype=bool)

        self.init_actor_capacity()

    def init_actor_capacity(self):
        self.actor_state_cap = []
        self.actor_new_state_cap = []
        self.actor_action_cap = []

        for agent_i in range(self.n_agents):
            self.actor_state_cap.append(np.empty((self.capacity, self.actor_dims[agent_i])))
            self.actor_new_state_cap.append(np.empty((self.capacity, self.actor_dims[agent_i])))
            self.actor_action_cap.append(np.empty((self.capacity, self.n_actions)))

    def add_memo(self, raw_obs, state, action, reward, raw_next_obs, next_state, done):
        if self.counter % self.capacity == 0 and self.counter > 0:
            self.init_actor_capacity()

        idx = self.counter % self.capacity

        for agent_i in range(self.n_agents):
            self.actor_state_cap[agent_i][idx] = raw_obs[agent_i]
            self.actor_new_state_cap[agent_i][idx] = raw_next_obs[agent_i]
            self.actor_action_cap[agent_i][idx] = action[agent_i]

        self.state_cap[idx] = state
        self.new_state_cap[idx] = next_state
        self.reward_cap[idx] = reward
        self.done_cap[idx] = done
        self.counter += 1

    def sample(self):
        max_cap = min(self.counter, self.capacity)

        batch_sequence = np.random.choice(max_cap, self.batch_size, replace=False)

        states = self.state_cap[batch_sequence]
        rewards = self.reward_cap[batch_sequence]
        next_states = self.new_state_cap[batch_sequence]
        dones = self.done_cap[batch_sequence]

        actor_states = []
        actor_new_states = []
        actions = []

        for agent_i in range(self.n_agents):
            actor_states.append(self.actor_state_cap[agent_i][batch_sequence])
            actor_new_states.append(self.actor_new_state_cap[agent_i][batch_sequence])
            actions.append(self.actor_action_cap[agent_i][batch_sequence])

        return actor_states, states, actions, rewards, actor_new_states, next_states, dones

    def ready(self):
        if self.counter >= self.batch_size:
            return True
        return False


class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir):
        super(Critic, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # TODO: not sure if self.device or device

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(Actor, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # TODO: not sure if self.device or device

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = torch.softmax(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class MADDPGAgent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_i, chkpt_dir,
                 alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f"agent_{agent_i}"

        self.actor = Actor(alpha=alpha, input_dims=actor_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                           n_actions=n_actions, name=self.agent_name + "_actor", chkpt_dir=chkpt_dir)
        self.critic = Critic(beta=beta, input_dims=critic_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                             n_agents=n_agents, n_actions=n_actions, name=self.agent_name + "_critic", chkpt_dir=chkpt_dir)
        self.target_actor = Actor(alpha, actor_dims, fc1_dims, fc2_dims, n_actions, self.agent_name + "_target_actor",
                                  chkpt_dir)
        self.target_critic = Critic(beta, critic_dims, fc1_dims, fc2_dims, n_agents, n_actions,
                                    self.agent_name + "_target_critic", chkpt_dir)

        self.update_net(tau=1)

    def update_net(self, tau=None):
        if tau == None:
            tau = self.tau

        actor_state_dict = dict(self.actor.named_parameters())
        target_actor_state_dict = dict(self.target_actor.named_parameters())

        for param in actor_state_dict:
            actor_state_dict[param] = tau * actor_state_dict[param].clone() + (1 - tau) * target_actor_state_dict[
                param].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        critic_state_dict = dict(self.critic.named_parameters())
        target_critic_state_dict = dict(self.target_critic.named_parameters())

        for param in critic_state_dict:
            critic_state_dict[param] = tau * critic_state_dict[param].clone() + (1 - tau) * target_critic_state_dict[
                param].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def get_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = torch.rand(self.n_actions).to(self.actor.device)
        action = actions + noise
        return action.detach().cpu().numpy()[0]

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64,
                 gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario

        for agent_i in range(self.n_agents):
            self.agents.append(MADDPGAgent(actor_dims=actor_dims[agent_i], critic_dims=critic_dims,
                                     n_actions=n_actions, agent_i=agent_i, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print("Saving parameters...")
        for agent_i in self.agents:
            agent_i.save_models()

    def load_checkpoint(self):
        print("Loading parameters...")
        for agent_i in self.agents:
            agent_i.load_models()

    def get_actions(self, raw_obs):
        actions = []
        for agent_i, agent in enumerate(self.agents):
            action = agent.get_action(raw_obs[agent_i])
            actions.append(action)

        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample()
        device = self.agent[0].actor.device

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_i, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_i], dtype=torch.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = torch.tensor(actor_states[agent_i], dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_i])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_i, agent in enumerate(self.agents):
            critic_target_value = agent.target_critic.forward(new_states, new_actions).flatten()
            critic_target_value[dones[:0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_i] + agent.gamma * critic_target_value
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_paramaters()


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == "__main__":
    scenario = "simple"
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    for agent_i in range(n_agents):
        actor_dims.append(env.observation_space[agent_i].shape[0])
    critic_dims = sum(actor_dims)

    n_agents = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_agents,
                           scenario=scenario, alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64,
                           gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg')
    memory = ReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 100
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
        while not any(done):
            if IS_TEST:
                env.render()
            actions = maddpg_agents.get_actions(obs)
            next_obs, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            next_state = obs_list_to_state_vector(next_state)

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
