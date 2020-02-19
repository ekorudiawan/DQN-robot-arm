import os
import gym
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Class for saving agent experience
class ReplayMemory(object):
    def __init__(self, max_size=10000, n_states=4):
        self.max_size = max_size 
        self.memory_counter = 0
        self.states_memory = np.zeros((max_size, n_states), dtype=np.float32)
        self.next_states_memory = np.zeros((max_size, n_states), dtype=np.float32)
        self.actions_memory = np.zeros(max_size, dtype=np.int64)
        self.rewards_memory = np.zeros(max_size, dtype=np.float32)
        self.dones_memory = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.max_size
        self.states_memory[index] = state
        self.actions_memory[index] = action
        self.rewards_memory[index] = reward
        self.next_states_memory[index] = next_state
        self.dones_memory[index] = done
        self.memory_counter += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.memory_counter, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.states_memory[batch]
        actions = self.actions_memory[batch]
        rewards = self.rewards_memory[batch]
        next_states = self.next_states_memory[batch]
        dones = self.dones_memory[batch]
        return states, actions, rewards, next_states, dones

# Neural Network architecture for estimating action value
class QNetwork(nn.Module):
    def __init__(self, lr=0.0001, n_states=4, n_actions=6, checkpoint_dir="../weights", filename="weights"):
        super(QNetwork, self).__init__()
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(checkpoint_dir, filename)
        # Detail of architecture
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, n_actions)
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def save_checkpoint(self):
        print('... Save checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... Load checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DQNAgent(object):
    def __init__(self, alpha=0.0005, gamma=0.99, eps=1.0, eps_decay=0.995, eps_min=0.01, tau=0.001, max_size=100000, batch_size=64, update_rate=4, n_states=0, n_actions=0, checkpoint_dir="../weigths"):
        self.gamma = gamma 
        self.eps = eps 
        self.eps_decay = eps_decay 
        self.eps_min = eps_min
        self.tau = tau
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.action_space = [i for i in range(n_actions)]

        # Replay memory
        self.memory = ReplayMemory(max_size=max_size, n_states=n_states)

        # Q-Network
        self.qnetwork_local = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions, checkpoint_dir=checkpoint_dir, filename="qnetwork_local.pth")
        self.qnetwork_target = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions, checkpoint_dir=checkpoint_dir, filename="qnetwork_target.pth")

        self.counter = 0
    
    def decrement_epsilon(self):
        self.eps *= self.eps_decay 
        if self.eps < self.eps_min:
            self.eps = self.eps_min

    def epsilon_greedy(self, state):
        if np.random.random() > self.eps:
            state = T.tensor([state],dtype=T.float).to(self.qnetwork_local.device)
            actions = self.qnetwork_local.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = self.memory.sample_memory(self.batch_size)
        t_states = T.tensor(states).to(self.qnetwork_local.device)
        t_actions = T.tensor(actions).to(self.qnetwork_local.device)
        t_rewards = T.tensor(rewards).to(self.qnetwork_local.device)
        t_next_states = T.tensor(next_states).to(self.qnetwork_local.device)
        t_dones = T.tensor(dones).to(self.qnetwork_local.device)
        return t_states, t_actions, t_rewards, t_next_states, t_dones

    def save_models(self):
        self.qnetwork_local.save_checkpoint()
        self.qnetwork_target.save_checkpoint()

    def load_models(self):
        self.qnetwork_local.load_checkpoint()
        self.qnetwork_target.load_checkpoint()

    def learn(self, state, action, reward, next_state, done):
        # Save experience to memory
        self.store_transition(state, action, reward, next_state, done)

        # If not enough memory then skip learning
        if self.memory.memory_counter < self.batch_size:
            return

        # Update target network parameter every update rate
        if self.counter % self.update_rate == 0:
            self.soft_update(self.tau)

        # Take random sampling from memory
        states, actions, rewards, next_states, dones = self.sample_memory()

        # Update action value
        indices = np.arange(self.batch_size)
        q_pred = self.qnetwork_local.forward(states)[indices, actions]
        q_next = self.qnetwork_target.forward(next_states).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        self.qnetwork_local.optimizer.zero_grad()
        loss = self.qnetwork_local.loss(q_target, q_pred).to(self.qnetwork_local.device)
        loss.backward()
        self.qnetwork_local.optimizer.step()
        self.counter += 1

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def regular_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data + target_param.data)

if __name__ == "__main__":
    env = gym.make('gym_robot_arm:robot-arm-v0')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(alpha=0.0001, n_states=n_states, n_actions=n_actions)
    writer = SummaryWriter('./log/dqn_trial_1')
    load_models = False
    n_episodes = 2000
    n_steps=300

    # Load weights
    if load_models:
        agent.eps = agent.eps_min
        agent.load_models()

    total_reward_hist = []
    avg_reward_hist = []
    for episode in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0
        for t in range(n_steps):
            # Render after episode 1800
            if episode > 1800:
                env.render()
            action = agent.epsilon_greedy(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if not load_models:
            agent.decrement_epsilon()

        # Save model 
        if episode > 1000 and episode % 200 == 0:
            agent.save_models()
        

        total_reward_hist.append(total_reward)
        avg_reward = np.average(total_reward_hist[-100:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "Epsilon : {:.2f}".format(agent.eps), "Total Reward : {:.2f}".format(total_reward), "Avg Reward : {:.2f}".format(avg_reward))
        # Tensorboard log
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Avg Reward', avg_reward, episode)
        # writer.add_scalar('Epsilon', agent.eps, episode)
    fig, ax = plt.subplots()
    t = np.arange(n_episodes)
    ax.plot(t, total_reward_hist, label="Total Reward")
    ax.plot(t, avg_reward_hist, label="Average Reward")
    ax.set_title("Reward vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()