from __future__ import annotations

import gym
import numpy as np
from numpy import random as rd 
from gym import spaces

 
class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        gamma=0.99,
        min_stock=1,
        initial_capital=1e2,
        buy_cost_pct=0,
        sell_cost_pct=0,
        reward_scaling=1e-1,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        label_ary = config["label_array"]
        if_train = config["if_train"]
        
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.label_ary = label_ary.astype(np.float32)
        self.if_train = if_train
        
        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.min_stock = min_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None
        self.actions_memory = []

        # environment information
        self.env_name = "StockEnv"
        
        # amount + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 3 * stock_dim + self.tech_ary.shape[1]
        
        self.stocks_cool_down = None
        self.action_dim = 3
        self.max_step = self.price_ary.shape[0] - 1
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        # self.action_space = gym.spaces.Discrete(3)
        self.action_space = spaces.Box(low=0, high=1, shape=(stock_dim, self.action_dim))



    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]
            
        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.actions_memory = []
        return self.get_state(price)  # state


    def step(self, actions):
        actions = (actions).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        # actions = self.label_ary[self.day]
        self.stocks_cool_down += 1

        min_action = int(self.min_stock)  # stock_cd
        
        # print(actions)
        for index in np.where(actions <= -min_action)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.amount += (
                    price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                )
                self.stocks_cool_down[index] = 0
        for index in np.where(actions >= min_action)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                # print(self.amount // price[index], actions)
                buy_num_shares = min(self.amount // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.amount -= (
                    price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                self.stocks_cool_down[index] = 0
    
        self.actions_memory.append(actions)

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        
        self.episode_return += reward
        # if done:
        #     # reward = self.gamma_reward
        #     self.episode_return = total_asset / self.initial_total_asset
        #     # print(self.actions_memory)
        
        if done:
            return state, self.episode_return , done, False
        else:
            return state, reward, done, False

        # return state, reward, done, False


    def get_state(self, price):
        amount = np.array(self.amount * (2**-6), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    
def sample_from_env(seed, env, weights):    
    rd.seed(seed)
    done = False
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], [] 
    obs = env.reset()

    day = 0
    while not done:
        day += 1
        action = weights[day].reshape(-1)
        next_obs, reward, done, _, _ = env.step(action)
        action = [0,1] if weights[day] else [1,0]
        
        obs_.append(obs)
        next_obs_.append(next_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done)
        obs = next_obs
        
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }