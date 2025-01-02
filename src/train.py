from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from copy import deepcopy
import time
# import matplotlib.pyplot as plt
from evaluate import evaluate_HIV, evaluate_HIV_population



class LSTMunit(nn.Module):
    def __init__(self, batch, hidden, input, device):
        super(LSTMunit, self).__init__()
        init_weight = lambda *shape : nn.Parameter(torch.randn(*shape))
        triple = lambda : (init_weight(input, hidden), 
                            init_weight(hidden, hidden), 
                            init_weight(hidden))
        self.Wxi, self.Whi, self.bi = triple()
        self.Wxf, self.Whf, self.bf = triple()
        self.Wxo, self.Who, self.bo = triple()
        self.Wxc, self.Whc, self.bc = triple()

        self.previous_long_term = torch.zeros((batch, hidden), device = device)
        self.previous_hidden = torch.zeros((batch, hidden), device = device)
        self.fc1 = nn.Linear(hidden, env.action_space.n)
    def forward(self, x):
        self.input_gate = torch.sigmoid(x@self.Wxi + self.previous_hidden@self.Whi + self.bi)
        self.forget_gate = torch.sigmoid(x@self.Wxf + self.previous_hidden@self.Whf + self.bf)
        self.output_gate = torch.sigmoid(x@self.Wxo + self.previous_hidden@self.Who + self.bo)

        self.input_node = torch.tanh(x@self.Wxc + self.previous_hidden@self.Whc + self.bc)

        self.long_term = self.forget_gate*self.previous_long_term + self.input_gate*self.input_node
        self.short_term = self.output_gate * torch.tanh(self.long_term)
        x = self.fc1(self.short_term)
        # Detach recurrent states for next timestep
        self.previous_long_term = self.long_term.detach()
        self.previous_hidden = self.short_term.detach()

        return x

import torch
import torch.nn as nn

class SignalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        """
        LSTM model for signal processing and action prediction.
        
        Parameters:
        - input_size: Number of features per time step (6 in this case).
        - hidden_size: Number of units in the hidden layer of the LSTM.
        - num_layers: Number of stacked LSTM layers.
        - output_size: Number of possible actions (4 in this case).
        - device: The device to run the model on (CPU or GPU).
        """
        super(SignalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to map LSTM output to action probabilities
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, seq_length, input_size).
        - hidden: Tuple of hidden and cell states, if provided.

        Returns:
        - out: Output tensor of shape (batch_size, output_size).
        - hidden: Updated hidden and cell states.
        """
        batch_size = x.size(0)

        # Initialize hidden states if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            hidden = (h0, c0)

        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)

        # Use the last hidden state (from the last time step) for classification
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layer to output action probabilities
        out = self.fc(out)  # (batch_size, output_size)
        return out, hidden



    
class DQN(nn.Module):
    def __init__(self, input=6, output=4):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, output))


        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights of the model using a random distribution.
        You can customize this to use specific initializations like Xavier or He initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
# class ReplayBuffer:
#     def __init__(self, capacity, device, max_episode):
#         self.capacity = int(capacity)  # Capacity of the buffer
#         self.data = []
#         self.index = 0  # Index of the next cell to be filled
#         self.device = device
#         self.max_episode = max_episode

#     def append(self, s, a, r, s_, d, episode):
#         # Convert all inputs to tensors and move them to the target device
#         s = torch.tensor(s, dtype=torch.float32, device=self.device)
#         a = torch.tensor(a, dtype=torch.long, device=self.device)
#         r = torch.tensor(r, dtype=torch.float32, device=self.device)
#         s_ = torch.tensor(s_, dtype=torch.float32, device=self.device)
#         d = torch.tensor(d, dtype=torch.float32, device=self.device)  # Done is float for 1 - d calculation
#         if len(self.data) < self.capacity:
#             self.data.append(None)
#         self.data[self.index] = (s, a, r, s_, d)
#         self.index = (self.index + 1) % self.capacity

#     def sample(self, batch_size, step, episode):
#         low_step, high_step, step_data = [], [], []
#         if episode >=1:
#             low_step.append(step -1 - (200*episode + batch_size))
#             high_step.append(low_step[-1] + batch_size)
#         else:
#             low_step.append(step-1-batch_size)
#             high_step.append(step-1)
#         for i in range(episode):
#             low_step.append(low_step[-1] + 200)
#             high_step.append(high_step[-1] + 200)
#         # print("high, low, step", low_step, high_step, step)
#         step_data = [self.data[low:high] for low, high in zip(low_step, high_step)]
#         flat_step_data = [item for sublist in step_data for item in sublist]
#         # Randomly sample `batch_size` transitions from step_data
#         batch = random.sample(flat_step_data, batch_size)
#         # No need to move to device again; they are already on the correct device
#         return tuple(map(torch.stack, zip(*batch)))

#     def __len__(self):
#         return len(self.data)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)  
# The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    config = {
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.02,
          'epsilon_max': 1,
          'epsilon_decay_period': 21000,
          'epsilon_delay_decay': 100,
          'batch_size': 700,
          'max_episode': 200,
          'nb_gradient_steps' : 1,
          'udpate_target_freq' : 400}
    dqn_network = DQN()
    # dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dqn_network = LSTMunit(input_size = env.observation_space.shape[0], hidden_size = 200, num_layers= 1, device=dev)
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_policy = self.dqn_network.to(self.device)
        self.model_target = deepcopy(self.model_policy).to(self.device)
        self.max_episode = self.config['max_episode']
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device) # self.max_episode
        self.lr = self.config['learning_rate']
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.update_target_frequency = self.config['udpate_target_freq']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.nb_gradient_steps = self.config['nb_gradient_steps']
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model_policy.parameters(), lr= self.lr)


    def greedy_action(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q,  = self.model_policy(observation)
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random == True: #Exploration
            action = env.action_space.sample()
        else: #Exploitation
            action = self.greedy_action(observation)
        return action
    
    # def gradient_step(self, step, episode, double_dqn):
    #     if (len(self.memory) > self.batch_size and episode ==0) or (episode >=1 and len(self.memory) > 200*(episode)  + self.batch_size):

    #         X, A, R, Y, T = self.memory.sample(self.batch_size, step, episode)
    #         if double_dqn == True:
    #             next_actions = self.model_policy(Y).argmax(dim=1)  
    #             Q_target_next = self.model_target(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1)
    #             update = R + self.gamma * Q_target_next
    #         else:
    #             QYmax = self.model_target(Y).max(1)[0].detach()
    #             update = R + self.gamma * QYmax 
    #         Q_XA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1)).squeeze(1)
    #         loss = self.criterion(Q_XA, update)
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step() 

    def gradient_step(self): #, step, episode
        if len(self.memory) > self.batch_size:
        # if (len(self.memory) > self.batch_size and episode ==0) or (episode >=1 and len(self.memory) > 200*(episode)  + self.batch_size):

            X, A, R, Y, D = self.memory.sample(self.batch_size) # , step, episode
            QYmax = self.model_target(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def update_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self, env):
        episode = 0
        episode_return = []
        epsilon = self.epsilon_max
        state, _ = env.reset()
        step = 1
        episode_cum_reward = 0
        best_score = 0
        compteur_lr = 0
        while episode < self.max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            if np.random.rand() < epsilon:
                action = self.act(state, True) #Exploration
            else:
                action = self.act(state, False) # Exploitation

            # Step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, trunc) # ,episode
                                                                          
            episode_cum_reward += reward
            # Train

            for _ in range(self.nb_gradient_steps): 
                self.gradient_step() # , step, episode
                # self.gradient_step()

            if step % self.update_target_frequency  == 0:
                self.model_target.load_state_dict(self.model_policy.state_dict())

            if epsilon == self.epsilon_min:
                if compteur_lr == 5:
                    self.lr = self.lr/2
                    print("New lr : ", self.lr)
                    self.update_learning_rate(self.optimizer, self.lr)
                compteur_lr +=1
            if done or trunc : 
                
                # validation_score = evaluate_HIV(agent=self.model_policy, nb_episode=1)
                episode += 1
                print(f"Episode {episode} | ", 
                      f"Epsilon {epsilon:6.4f} | ", 
                    #   ", batch size ", '{:5d}'.format(len(self.memory[episode-1])), 
                      f"Episode return {episode_cum_reward:.2e}",
                    #   ", validation score ", '{:4.1f}'.format(validation_score),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                
                if episode_cum_reward > best_score:
                    best_score = episode_cum_reward
                    # self.best_model = deepcopy(self.model_policy).to(self.device)
                    self.save("policy")
                episode_cum_reward = 0

            else:
                state = next_state
            step += 1
            # if episode % 20 == 0:
            #     plt.semilogy(episode_return)
        print("best score", best_score)
        return episode_return


    def save(self, path):
        os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(self.model_policy.state_dict(), os.path.join(path, "model.pth"))
        print(f"Model saved to {os.path.join(path, 'model.pth')}")

    def load(self):
        file_path = "mva-rl-assignment-Antoinelunven/src/policy/model_128_lr.pth"
        # file_path = os.path.join(os.getcwd(), 'src\\policy', 'model_128_lr.pth')
        print(file_path)
        # model_path = os.path.join(path, "model.pth")
        # print(model_path)
        self.model_policy.load_state_dict(torch.load(file_path))
        self.model_policy.eval()
        print(f"Model loaded from {file_path}")
        return 

# class MCTSNode:
#     def __init__(self, state, parent=None, action=None):
#         self.state = state  # Current state
#         self.parent = parent  # Parent node
#         self.action = action  # Action taken to reach this node
#         self.children = []  # List of child nodes
#         self.visits = 0  # Number of visits to this node
#         self.value = 0  # Total reward accumulated at this node

#     def is_fully_expanded(self, action_samples):
#         return len(self.children) >= action_samples

#     def best_child(self, exploration_weight=1.41):
#         """Select the child node with the highest UCB score."""
#         ucb_scores = [
#             (child.value / child.visits if child.visits > 0 else 0)
#             + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
#             for child in self.children
#         ]
#         return self.children[np.argmax(ucb_scores)]
    
# class MCTSAgent:
#     def __init__(self, model, action_space, gamma=0.99, rollout_depth=200, action_samples=4):
#         """
#         Args:
#             model: A Q-network model to predict action values.
#             action_space: The environment's action space.
#             gamma: Discount factor for rollouts.
#             rollout_depth: Maximum depth for rollout simulations.
#             action_samples: Number of actions to sample at each state.
#         """
#         self.model = model
#         self.action_space = action_space
#         self.gamma = gamma
#         self.rollout_depth = rollout_depth
#         self.action_samples = action_samples

#     def rollout(self, env, state):
#         """Simulate a random playthrough from the given state."""
#         cumulative_reward = 0
#         discount = 1.0

#         for _ in range(self.rollout_depth):
#             action = self.action_space.sample()  # Random action
#             next_state, reward, done, truncated, _ = env.step(action)
#             cumulative_reward += discount * reward
#             discount *= self.gamma

#             if done or truncated:
#                 break
#             state = next_state

#         return cumulative_reward

#     def expand(self, node, env):
#         """Expand a node by sampling an action and adding a child node."""
#         if not node.is_fully_expanded(self.action_samples):
#             action = self.action_space.sample()  # Sample a random action
#             next_state, reward, done, _, _ = env.step(action)

#             child_node = MCTSNode(next_state, parent=node, action=action)
#             node.children.append(child_node)
#             return child_node, reward
#         return None, 0

#     def backpropagate(self, node, reward):
#         """Propagate the simulation result up the tree."""
#         while node:
#             node.visits += 1
#             node.value += reward
#             reward *= self.gamma  # Discount the reward
#             node = node.parent

#     def search(self, env, state, iterations):
#         """Run MCTS for a fixed number of iterations."""
#         root = MCTSNode(state)

#         for _ in range(iterations):
#             # Step 1: Select a promising node
#             node = root
#             while node.is_fully_expanded(self.action_samples) and node.children:
#                 node = node.best_child()

#             # Step 2: Expand the node
#             child_node, reward = self.expand(node, env)

#             if child_node:
#                 # Step 3: Simulate (rollout) if it's a leaf
#                 reward += self.rollout(env, child_node.state)

#             # Step 4: Backpropagate the reward
#             self.backpropagate(node, reward)

#         # Return the action leading to the best child
#         best_child = root.best_child(exploration_weight=0)  # Pure exploitation
#         return best_child.action

#     def act(self, env, state, iterations=100):
#         """Choose an action using MCTS."""
#         return self.search(env, state, iterations)
    

# class Node:
#     def __init__(self, state, parent = None, action = None, c = 1.41):
#         self.state = state
#         self.action = action 
#         self.children = []
#         self.visit = 0
#         self.value = 0

#     def best_node(self, c = 1.41):
#         for child in range(self.children):
#             if child.visit != 0:
#                 mean_term = child.value/child.visit 
#             else:
#                 mean_term = 0
#             variance_term = self.c*np.sqrt(np.log(self.visit)/child.value)
#             ucb_reward = mean_term + variance_term
#         return self.children[np.argmax(ucb_reward)]
    
# class MCTS:
#     dqn_network = DQN()
#     def __init__(self):
#         self.gamma = 0.99
#         self.rollout = 100
#         self.iterations = 400
#         self.model = self.dqn_network
        
#     def expansion(self, node, env, itr):
#         reward_list = np.zeros((4))
#         if itr == 0:
#             return None, 0
#         else:
#             for i in range(self.iterations):
#                 for  k in range(4):
#                     action = env.action_space.sample()
#                     next_state, reward, done, trunc, _ = env.step(k)
#                     new_node = Node(next_state)
                    
#                     reward += self.simulation(itr)
#                     reward_list[k] = reward_list
#                 node.append(new_node)
#             return node, reward
        
#     def simulation(self, itr):
#         discount = 1.0
        
#         for _ in range(self.rollout):
#             action = env.action_space.sample()
#             next_state, reward, done, trunc, _ = env.step(action)
#             reward += discount * reward
#             discount *= self.gamma
#         return reward
    
#     def backpropagation(self, reward, node):
#         while node:
#             node.value *=self.gamma
#             node.vists += 1
#             node = node.parent    
#             reward *= self.gamma

#     def train(self, env, max_episodes):
#         episode = 0
#         state, _ = env.reset()
#         episode_reward = []
#         while episode < max_episodes:
#             root = Node(state)
#             #Monte carlo
#             for itr in range(200):
#                 print("iteration", itr)
#                 node = root
#                 while len(node.children) > 4 :
#                     node = node.best_node()
#                 node, reward = self.expansion(node, env, itr)
#                 reward += self.simulation(itr)
#                 self.backpropagation(reward, node)
#             best_child = root.best_node(c=0)  # Pure exploitation
#             cumulated_reward = 0
#             for action in best_child:
#                 print(action)
#                 next_state, rew, done, trunc, _ = env.step(action)
#                 cumulated_reward += rew
#             episode_reward.append(cumulated_reward)

#             episode = episode +1
#         return episode_reward
            





