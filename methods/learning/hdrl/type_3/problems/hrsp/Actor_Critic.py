import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    """
    Actor 
    """
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x))
        return out

class Critic(nn.Module):
    '''
    Critic
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out

class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        self.actor = Actor(self.action_dim, self.state_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()

    def get_action(self, s):
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = Actor_Critic(env)
    reward = []
    for episode in range(200):
        s, _ = env.reset()
        env.render()
        done = False
        ep_r = 0
        while not done:
            a, log_prob = model.get_action(s)

            s_,rew,done,_,__,  = env.step(a)

            ep_r += rew

            model.learn(log_prob,s,s_,rew)

            s = s_
        reward.append(ep_r)
        print(f"episode:{episode} ep_r:{ep_r}")

    plt.plot(reward)
    plt.show()



