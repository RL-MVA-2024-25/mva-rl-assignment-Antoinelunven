from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
# from fast_env import FastHIVPatient
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.multiprocessing as mp

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



    
class Deterministic_DQN(nn.Module):
    def __init__(self, input=6, output=4, layer = 256):
        super(Deterministic_DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(layer, layer),
            nn.ReLU(),
            
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            # nn.Dropout(p=0.3),
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            nn.Linear(layer, output))
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

class Stochastic_DQN(nn.Module):
    def __init__(self, input=6, output=4, layer = 256):
        super(Stochastic_DQN, self).__init__()

        self.linear_input = nn.Linear(input, layer)
        self.linear_normal = nn.Linear(layer, layer)
        self.batch_norm = nn.BatchNorm1d(layer)
        self.fc = nn.Sequential(
            nn.Linear(input, layer),
            nn.ReLU(),
            # nn.LayerNorm(layer),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.LayerNorm(layer),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.BatchNorm1d(layer),
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            # nn.BatchNorm1d(layer),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.Linear(layer, 512),
            # nn.ReLU(),

            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.LayerNorm(layer),
            
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.LayerNorm(layer),
            nn.Linear(layer, layer),
            nn.ReLU(),
            # nn.LayerNorm(layer),
            # nn.Linear(layer, layer),
            # nn.ReLU(),
            # nn.LayerNorm(layer),
            )
        self.mu_layer = nn.Linear(layer, output)
        self.std_layer = nn.Linear(layer, output)
        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        # x = F.relu(self.linear_input(x))
        # print("x1", x.shape)
        # x = F.relu(self.linear_normal(x))
        # print("x2", x.shape)
        # x = self.batch_norm(x)
        # print("batch norm", x.shape)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        std = F.softplus(log_std) + 1e-6
        # std = torch.exp(log_std) 
        return mu, std
    
    def sample_action(self, x):
        mu, log_std = self(x)
        # print("sample", log_std)
        std = F.softplus(log_std)+ 1e-6
        # std = torch.exp(log_std) 
        action = mu + torch.randn_like(mu) * std
        return action

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
#     def __init__(self, capacity, device, state_dim = 6, action_dim = 1):
#         self.capacity = int(capacity)
#         self.device = device
#         self.index = 0
#         self.size = 0

#         # Preallocate tensors on the GPU
#         self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
#         self.actions = torch.zeros((capacity, action_dim), dtype=torch.long, device=device)
#         self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
#         self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
#         self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

#     def append(self, s, a, r, s_, d):
#         """
#         Append a new transition to the buffer.
#         """
#         # print(s.dtype, a.dtype, r.dtype, s_.dtype)
#         self.states[self.index] = s if isinstance(s, torch.Tensor) else torch.tensor(s, device=self.device)
#         self.actions[self.index] = a if isinstance(a, torch.Tensor) else torch.tensor(a, device=self.device)
#         self.rewards[self.index] = r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
#         self.next_states[self.index] = s_ if isinstance(s_, torch.Tensor) else torch.tensor(s_, device=self.device)
#         self.dones[self.index] = d if isinstance(d, torch.Tensor) else torch.tensor(d, device=self.device)

#         self.index = (self.index + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size):
#         """
#         Efficient sampling directly from preallocated tensors.
#         """
#         befin_map = time.perf_counter()
#         indices = torch.randint(0, self.size, (batch_size,), device=self.device)  # GPU-based sampling
#         map_time = time.perf_counter() - befin_map
#         # Directly index the pre-allocated tensors
#         befin_batch = time.perf_counter()
#         states = self.states[indices]
#         actions = self.actions[indices]
#         rewards = self.rewards[indices]
#         next_states = self.next_states[indices]
#         dones = self.dones[indices]
#         batch_time = time.perf_counter() - befin_batch
#         return states, actions, rewards, next_states, dones, batch_time, map_time

#     def __len__(self):
#         return self.size

    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        # s, a, r, s_, d = torch.Tensor(s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device), d.to(self.device)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)  
# env = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)  

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
          'epsilon_min': 0.01,
          'epsilon_max': 1,
          'epsilon_decay_period': 25000,
          'epsilon_delay_decay': 3000,
          'batch_size': 200,
          'max_episode': 400,
          'nb_sample' : 15,
          'max_gradient_steps' : 8,
          'epsilon_seuil' : 0.25,
          'deterministic' : False,
          'episode_seuil' : 30,
          'explore_episodes' : 80,
          'patience_lr' : 7,
          'udpate_target_freq' : 400}
    dqn_network_deterministic = Deterministic_DQN()
    dqn_network_stochastic = Stochastic_DQN()
    # dqn_network = LSTMunit(input_size = env.observation_space.shape[0], hidden_size = 200, num_layers= 1, device=dev)
    def __init__(self):        

        self.deterministic = self.config['deterministic']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.deterministic == True:
            self.model_policy = self.dqn_network_deterministic.to(self.device)
            self.model_target = deepcopy(self.model_policy).to(self.device)
        else:
            self.model_policy = self.dqn_network_stochastic.to(self.device)
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
        self.max_gradient_steps = self.config['max_gradient_steps']
        self.patience = self.config['patience_lr']
        self.explore_episodes = self.config['explore_episodes']
        if self.deterministic == True:
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            # self.criterion = self.gaussian_nll_loss
            self.criterion = torch.nn.SmoothL1Loss()
            # self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_policy.parameters(), lr= self.lr)
        self.epsilon_seuil = self.config["epsilon_seuil"]
        self.scheduler  = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=self.patience, verbose= True)
        self.sample = self.config['nb_sample']
        self.episode_seuil = self.config['episode_seuil']
        
        self.gradient_time  = 0
        self.batch_time = 0
        self.map_time = 0
        self.compteur_stop = 0
        self.sampling_time = 0
        self.episode_time = 0
        self.epsilon  = self.epsilon_max
        self.step = 1
        self.gradient_steps = 0
        self.var = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.mu = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.previous_best = 0
        self.episode_seuil += self.explore_episodes

    def gaussian_nll_loss(self, mu, std, target):
        """
        Negative Log-Likelihood loss for a Gaussian distribution.
        """
        var = std ** 2  # Variance
        nll = 0.5 * torch.log(2 * torch.pi * var) + ((target - mu) ** 2) / (2 * var)
        return torch.mean(nll)

    def greedy_action(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q,  = self.model_policy(observation)
            return torch.argmax(Q).item()
        
    def Bayesian_TS(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        # self.model_policy.eval()
        mu, std = self.model_policy(observation) 
        # self.model_policy.train()
        Q_sample = mu + torch.randn_like(mu) * std
        # print("mu", mu.cpu().detach().numpy())
        # print("std", (torch.randn_like(mu) * std).cpu().detach().numpy())
        # print('Q sample', Q_sample.cpu().detach().numpy())
        self.var = torch.mean(std)
        self.mu = torch.mean(mu)
        return torch.argmax(Q_sample).item()
    
    def act(self, observation, use_random=False):
        observation = np.sign(observation)*np.log(1+np.abs(observation))
        if self.deterministic == True:
            if self.step > self.epsilon_delay:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
            if np.random.rand() < self.epsilon:
                action = env.action_space.sample()
                return action
            else:
                action = self.greedy_action(observation)           
            return action
        else:
            if self.gradient_steps == 0:
                action = env.action_space.sample()
                return action
            else:
                action = self.Bayesian_TS(observation)
                return action  
        
    def gradient_step(self, double_dqn): #, step, episode
        start_sampling = time.perf_counter()
        X, A, R, Y, D  = self.memory.sample(self.batch_size) # , step, episode
        # print(X.shape, A.shape, R.shape, Y.shape, D.shape)
  
        X, A, R, Y, D = X.to(self.device, non_blocking=True), A.to(self.device, non_blocking=True), R.to(self.device, non_blocking=True), Y.to(self.device, non_blocking=True), D.to(self.device, non_blocking=True)
        R = torch.sign(R) * torch.log(1 + torch.abs(R))
        X = torch.sign(X) * torch.log(1 + torch.abs(X))
        Y = torch.sign(Y) * torch.log(1 + torch.abs(Y))
        self.sampling_time += time.perf_counter() - start_sampling
        
        if self.deterministic == True:
            if double_dqn :
                next_actions = self.model_policy(Y).argmax(dim=1)  # Actions with the highest Q-value
                QY_next = self.model_target(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
                update = R + self.gamma * QY_next * (1 - D)
                QXA = self.model_policy(X).gather(1, A.to(torch.long).squeeze(1)).unsqueeze(1)
                loss = self.criterion(QXA, update.unsqueeze(1).squeeze(1))
            else:
                QYmax = self.model_target(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                QXA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
        else:
            if double_dqn :
                pass
                # next_actions = self.model_policy_sto_der(Y).argmax(dim=1)  # Actions with the highest Q-value
                # QY_next = self.model_target_sto_der(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
                # update = R + self.gamma * QY_next * (1 - D)
                # QXA = self.model_policy_sto_der(X).gather(1, A.to(torch.long).unsqueeze(1)).squeeze(1)
                # loss = self.criterion(QXA, update.unsqueeze(1).squeeze(1))
            else:
                QYmax = self.model_target.sample_action(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                # mu, std = self.model_policy(X)
              
                # mu = mu.gather(1, A.to(torch.long).unsqueeze(1))
                # loss = self.criterion(mu, std, update.unsqueeze(1))
                # print(f" X_moy {X.mean().item():2.3f}, X_min {X.min().item():2.3f}, X_max {X.max().item():2.3f}, T_moy {Y.mean().item():2.3f}, T_min {Y.min().item():2.3f}, T_max {Y.max().item():2.3f}, R_moy {R.mean().item():2.3f}, R_min {R.min().item():2.3f}, R_max {R.max().item():2.3f}")

                # print(f"mu: {mu.mean().item()}, std: {std.mean().item()}, update: {update.mean().item()}, loss : {loss.item()}, X {X.mean().item()}, T {Y.mean().item()}, R {R.mean().item()},")


                QXA = self.model_policy.sample_action(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))

                # print(f"QXA : {QXA.mean().item()}, update: {update.mean().item()}, loss : {loss.item()}, X {X.mean().item()}, T {Y.mean().item()}, R {R.mean().item()},")

        # self.model_policy.train()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # self.model_policy.eval()

# Custom function for scaling learning rate
    
    def early_stop(self, best_score):

        if self.compteur_stop == int(self.max_gradient_steps*self.patience) +2:
            return True
        if abs(best_score - self.previous_best) < 1.1: # Because both values are not strictly equal
            self.compteur_stop += 1
        else:
            self.compteur_stop = 0
            self.previous_best = best_score
        return False
        
    def gradient_steps_calculation(self, episode):
        k = 2  # Adjust scaling factor for exponential steepness
        min_steps = 1  # Minimum number of gradient steps
        max_steps = self.max_gradient_steps  # Maximum number of gradient steps
        if self.deterministic == True:
            if self.epsilon >0.55:
                return 1

            elif self.epsilon > self.epsilon_seuil:
                scale = np.exp(-k * (self.epsilon - self.epsilon_seuil) / (1 - self.epsilon_seuil))
            else:
                scale = 1- np.exp(-k * (self.epsilon - self.epsilon_min) / (self.epsilon_seuil - self.epsilon_min))
            self.gradient_steps =  int(min_steps + (max_steps - min_steps) * scale)
        else:
            if episode < self.explore_episodes:
                self.gradient_steps = 0
            if self.explore_episodes < episode <= self.episode_seuil:
                scale = 1-np.exp(-k * (episode - self.explore_episodes) / (self.episode_seuil - self.explore_episodes))
                self.gradient_steps = int(min_steps + (max_steps) * scale)
            else:
                # print("compteur stop", self.compteur_stop, "division", self.compteur_stop % self.patience, "patience", self.patience)
                if self.compteur_stop % self.patience == 0  and self.compteur_stop !=0 and self.gradient_steps !=1:
                    self.gradient_steps = self.gradient_steps -1

    def train(self, env):
        episode = 0
        episode_return, var_return = [], []
        state, _ = env.reset()
        episode_cum_reward = 0
        best_score = 0
        env_duration = 0
        
        cumulated_var = 0
        cumulated_mu = 0


        # state = np.sign(state)*np.log(1+np.abs(state))

        while episode < self.max_episode:

            if episode != 0:
                if trunc == True :
                    self.episode_time = time.perf_counter()
                    torch.cuda.synchronize()
            # Observation vs exploitation
            if self.deterministic == True:
                action = self.act(state)
            else:
                action = self.act(state)
            
            # Step
            
            next_state, reward, done, trunc, _ = env.step(action)
            # next_state = np.sign(next_state)*np.log(1+np.abs(next_state))
            env_start = time.perf_counter()
            self.memory.append(state , action, reward,next_state, trunc) # ,episode
            env_duration += time.perf_counter() - env_start
            episode_cum_reward += reward
            
            cumulated_var += self.var
            cumulated_mu += self.mu
            # Train
            if trunc == True:
                self.gradient_steps_calculation(episode)
            

            if self.deterministic == True:
                if self.epsilon < self.epsilon_seuil and trunc == True:
                    stop = self.early_stop(best_score)
                    self.scheduler.step(best_score)
                    if stop :
                        print(f"Best score {best_score:.2e}")
                        return episode_return
            else:
                if episode > self.episode_seuil and trunc == True:
                    stop = self.early_stop(best_score)
                    self.scheduler.step(best_score)
                    if stop :
                        print(f"Best score {best_score:.2e}")
                        return episode_return
                    
            if len(self.memory) > self.batch_size:    
                for i in range(self.gradient_steps):
                    self.gradient_step(double_dqn=False)
            

            if self.step % self.update_target_frequency  == 0:
                self.model_target.load_state_dict(self.model_policy.state_dict())

            if done or trunc :           
                episode += 1
                if self.deterministic:
                    epsilon_or_variance = f"Epsilon {self.epsilon:6.4f} | "
                else:
                    epsilon_or_variance = f"Variance {cumulated_var.cpu().detach().numpy().item():.2e} | Mean {cumulated_mu.cpu().detach().numpy().item():.2e} | "
                torch.cuda.synchronize()
                print(f"Episode {episode} | ",
                    epsilon_or_variance,
                    f"Episode return {episode_cum_reward:.2e} | ",                    
                    f"Episode time {(time.perf_counter() - self.episode_time):1.1f} | ",
                    f"Gradient steps {self.gradient_steps}",
                    sep='')
                self.gradient_time = 0
                self.sampling_time = 0
                self.map_time = 0
                self.batch_time = 0
                env_duration = 0
                state, _ = env.reset()
                # state = np.sign(state)*np.log(1+np.abs(state))
                episode_return.append(episode_cum_reward)
                var_return.append(cumulated_var)

                if episode_cum_reward > best_score:
                    
                    best_score = episode_cum_reward
                    self.save("policy")
                episode_cum_reward = 0
                cumulated_var = 0

            else:
                state = next_state
            self.step += 1
        print(f"Best score {best_score:.2e}")
        return episode_return


    def save(self, path):
        os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(self.model_policy.state_dict(), os.path.join(path, "new_lr.pth"))
        print(f"Model saved to {os.path.join(path, 'model.pth')}")

    def load(self):
        file_path = os.path.join(os.getcwd(), 'src/policy', 'Deep_qn_1st_stochastic.pth')
        print(file_path)

        self.model_policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
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
            





