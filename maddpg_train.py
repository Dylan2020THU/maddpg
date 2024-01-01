
from make_env import make_env
from maddpg_agent import MADDPGAgent


# # Initialize environment
env = make_env('simple_adversary')
print(env.observation_space)
print(env.action_space)
print(env.action_space[0].n)

obs = env.reset()
print(obs)
NUM_AGENTS = env.n

agents = []
for agent_i in range(NUM_AGENTS):  # initialize agents
    print("Initializing agent", agent_i)
    agent = MADDPGAgent(len(get_state(env)), n_RB * len(env.V2V_power_dB_List))  # CG
    agents.append(agent)


# STATE_DIM = env.observation_space.shape[0]
# ACTION_DIM = env.action_space.shape[0]
#
# # Hyperparameters
# NUM_EPISODE = 100
# NUM_STEP = 200
# EPSILON_START = 1.0
# EPSILON_END = 0.02
# EPSILON_DECAY = 10000
#
# # Initialize agent
# agent = DDPGAgent(STATE_DIM, ACTION_DIM)
#
# # Training Loop
# REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
# for episode_i in range(NUM_EPISODE):
#     state, others = env.reset()  # state: ndarray, others: dict
#     episode_reward = 0
#
#     for step_i in range(NUM_STEP):
#         # Select action
#         epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY],
#                             fp=[EPSILON_START, EPSILON_END])  # interpolation
#         random_sample = random.random()
#         if random_sample <= epsilon:
#             action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)
#         else:
#             action = agent.get_action(state)
#         # Execute action at and observe reward rt and observe new state st+1
#         next_state, reward, done, truncation, info = env.step(action)
#         # Store transition (st; at; rt; st+1) in R
#         agent.replay_buffer.add_memo(state, action, reward, next_state, done)
#
#         state = next_state
#         episode_reward += reward
#
#         agent.update()
#         if done:
#             break
#
#     REWARD_BUFFER[episode_i] = episode_reward
#
#     print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
#
# current_path = os.path.dirname(os.path.realpath(__file__))
# model = current_path + '/models/'
# timestamp = time.strftime("%Y%m%d%H%M%S")
#
# # Save models
# torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
# torch.save(agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')
#
# # Close environment
# env.close()
#
# # Save the rewards as txt file
# np.savetxt(current_path + f'/ddpg_reward_{timestamp}.txt', REWARD_BUFFER)
#
# # # Plot rewards using ax.plot()
# # plt.plot(REWARD_BUFFER)
# # plt.xlabel('Episode')
# # plt.ylabel('Reward')
# # plt.title('DDPG Reward')
# # plt.grid()
# # plt.show()
# # plt.savefig(current_path + f'/ddpg_reward_{timestamp}.png')
