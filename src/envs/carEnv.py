import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import envs.config as config
import pickle

from gym import Env, spaces
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class carEnv(Env):
    def __init__(self):
        super(carEnv, self).__init__()

        random.seed(config.SEED)

        # read demonstrations
        with open('src/demonstrations/safe_demo_1.pkl', 'rb') as f:
            self.demonstrations = pickle.load(f)

        obs_num = self.demonstrations[0]['observations'][0].shape[0]

        # observation_space: (min(obs_num, TOP_K d_obs) + 1 agent) * [posx, posy, vx, vy]
        observation_shape = ((min(obs_num, config.TOP_K + 1)) * 4,)
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=observation_shape)
        # action_space: (ax, ay)
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,))

        # initialize the states of agent and obs
        self.agent_state = None
        self.obs_state = None

        # traj's id
        self.__traj_id = None
        self.traj = None
        self.timestep = 0
        self.goal_state = None

        # create a canvas
        plt.ion()
        plt.close()
        self.fig = plt.figure(figsize=(9, 4))
        self.agent_size = None

        # vis
        self.is_vis = False

        self.max_episode_length = 1000





    def reset(self):
        self.success = 0
        self.collision_num = 0
        self.timestep = 0
        self.step_num = 0
        self.__traj_id = random.choice(range(len(self.demonstrations)))
        self.traj = self.demonstrations[self.__traj_id]
        start_state, self.goal_state = self.traj['observations'][0], self.traj['observations'][-1][-1, :]
        self.agent_state, self.obs_state = start_state[-1, :], start_state[:-1, :]
        self.obs_state = self.remove_dist_obs(self.agent_state, self.obs_state)

        vis_range = max(1, np.amax(np.abs(start_state[:, :2])))
        self.agent_size = 100 / vis_range ** 2
        if self.is_vis:
            self.draw_fig()

        # concatenate obv
        self.obs_state_removed = self.remove_dist_obs(self.agent_state, self.obs_state)
        obv = np.concatenate([self.obs_state_removed, self.agent_state[None, :]], axis=0).flatten()

        return obv


    def step(self, action):
        self.timestep += 1
        self.step_num += 1
        if self.timestep >= len(self.traj['observations']):
            self.timestep = len(self.traj['observations']) - 1
        # step of the agent
        dsdt = np.concatenate([self.agent_state[2:], action])  # YY: dsdt = [vx, vy, ax, ay]
        self.agent_state = self.agent_state + dsdt * config.TIME_STEP

        # step of the obs

        self.obs_state = self.traj['observations'][self.timestep][:-1, :]
        self.obs_state_removed = self.remove_dist_obs(self.agent_state, self.obs_state)

        # concatenate obv
        obv = np.concatenate([self.obs_state_removed, self.agent_state[None, :]], axis=0).flatten()

        # print("===================================")
        # print(self.goal_state)
        # print(self.agent_state)
        # print(self.obs_state)
        # print(np.linalg.norm(self.obs_state[:, :2] - self.agent_state[:2], axis = 1))

        done = False

        reward = - np.linalg.norm(self.agent_state[:2] - self.goal_state[:2])
        if self.step_num >= 50:
            done = True

        if np.linalg.norm(self.agent_state[:2] - self.goal_state[:2]) < config.DIST_MIN_CHECK:
            # print("Done, goal reached!")
            # done = True
            self.success = 1

        elif not np.all(np.linalg.norm(self.obs_state[:, :2] - self.agent_state[:2], axis = 1) > config.DIST_MIN_CHECK):
            # print("Collision detected!")
            # done = True
            self.collision_num += 1


        if self.is_vis:
            self.draw_fig()

        return obv, reward, done, {'success': self.success, 'collision_num': self.collision_num}  # don't need to define reward because it's irl


    def render(self, mode="human"):
        # TODO
        if mode == "human":
            self.is_vis = True
        if mode == "rgb_array":
            return self.draw_fig()


    def close(self):
        # TODO
        plt.clf()


    def draw_fig(self):
        plt.clf()

        # show the dynamic obstacles
        plt.scatter(self.obs_state[:, 0], self.obs_state[:, 1],
                    color='darkorange',
                    s=self.agent_size, label='Dynamic Obstacle', alpha=0.6)
        # show the agent
        plt.scatter(self.agent_state[0], self.agent_state[1],
                    color='green',
                    s=self.agent_size, label='Agent', alpha=0.6)

        plt.scatter(self.goal_state[0], self.goal_state[1],
                    color='deepskyblue',
                    s=self.agent_size, label='Target', alpha=0.6)


        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)

        self.fig.canvas.draw()

        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if not self.is_vis:
            return image
        else:
            plt.pause(0.01)


    def get_traj_id(self):
        return self.__traj_id


    # YY: agent_state with shape (4), obs_states with shape (obs_num, 4)
    def remove_dist_obs(self, agent_state, obs_state):
        topk_mask = np.argsort(np.sum(np.square((obs_state - agent_state)[:, :2]), axis=1))[:config.TOP_K]
        return obs_state[topk_mask, :]

    # return shape [agent_num, state_dim]
    def get_start_goal_states(self):
        self.traj = self.demonstrations[self.__traj_id]
        return self.traj['observations'][0], self.traj['observations'][-1][:, :2]

if __name__ == "__main__":
    env = carEnv()
    obs = env.reset()

    # Get agent's traj's actions
    with open('src/demonstrations/safe_demo_1.pkl', 'rb') as f:
        demonstrations = pickle.load(f)
    agent_actions = demonstrations[env.get_traj_id()]['actions']
    for i in range(10):
        agent_actions = demonstrations[i]['actions']
        print(len(agent_actions))

    tm = 0
    env.render()
    while True:
        # Take a random action
        action = env.action_space.sample()
        print(tm)
        if tm >= len(agent_actions):
            tm = -1
        action = agent_actions[tm][-1, :]
        # action = np.array([0.01, 0.01])
        obs, reward, done, info = env.step(action)

        # Render the game
        # env.render()

        if done == True:
            break

        tm += 1

    env.close()