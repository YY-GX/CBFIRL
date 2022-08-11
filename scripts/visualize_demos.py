import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import envs.config as config
import pickle
import os
from gym import Env, spaces
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.architectures import relu_net
import tensorflow as tf
from garage.tf.policies import GaussianMLPPolicy
from garage.envs import GymEnv
from envs.carEnv import carEnv
import seaborn as sns
from global_utils.utils import *

def show_safe_demo():
    demo_pth = 'src/demonstrations/safe_demo_stop.pkl'
    # Get agent's traj's actions
    with open(demo_pth, 'rb') as f:
        demonstrations = pickle.load(f)
    imgs = []
    for i in range(10, 20):
        env = carEnv(demo=demo_pth, seed=i)
        obs = env.reset()
        agent_actions = demonstrations[env.get_traj_id()]['actions']
        # agent_actions = demonstrations[i]['actions']

        tm = 0
        env.render('no_vis')
        # env.render('human')
    
        while True:
            imgs.append(env.render('rgb_array'))
            if tm >= len(agent_actions):
                tm = -1
            action = agent_actions[tm][-1, :]
            obs, reward, done, info = env.step(action)
    
            if done == True:
                break
    
            tm += 1
    
        env.close()
    save_video(imgs, f"data/visual_videos/demo/demo_8obs_stop.avi")
        

def show_unsafe_demo():
    with open('src/demonstrations/unsafe_states_1_10.pkl', 'rb') as f:
        S_u, A_u = pickle.load(f)

    fig = plt.figure(figsize=(9, 4))

    size = 100

    for i, s_u in enumerate(S_u):
        plt.clf()
        a_u = A_u[i][-1, :]
        a_u /= max(np.abs(a_u))
        plt.arrow(s_u[-1, 0], s_u[-1, 1], a_u[0], a_u[1])

        # show the dynamic obstacles
        plt.scatter(s_u[:-1, 0], s_u[:-1, 1],
                    color='darkorange',
                    s=size, label='Dynamic Obstacle', alpha=0.6)
        # show the agent
        plt.scatter(s_u[-1, 0], s_u[-1, 1],
                    color='green',
                    s=size, label='Agent', alpha=0.6)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)

        plt.pause(1)


def collect_learned_reward():
    # load reward
    log_path = f"data/yy/22_07_2022_17_00_38"  # r(s, a)
    # log_path = f"data/yy/25_07_2022_18_57_41"  # r(s)


    # log_path = f'data/obs16/03_08_2022_16_26_14'  # r(s) -> 1000 epoch
    log_path = 'data/obs16/03_08_2022_22_59_25'  # r(s, a) -> 1000 epoch

    # log_path = f'data/obs16/04_08_2022_17_23_23'  # r(s) -> 400 epoch
    log_path = 'data/obs16/05_08_2022_14_49_34'  # r(s, a) -> 400 epoch


    demo_pth = 'src/demonstrations/safe_demo_16obs_stop.pkl'
    dim = 52

    save_dictionary = {}
    env = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    with tf.Session() as sess:
        ph_obv = tf.placeholder(tf.float32, shape=(dim), name='ph_obv')
        a = tf.placeholder(tf.float32, shape=(2,), name='ph_a')




        rew_input = tf.reshape(ph_obv, [1, dim])  # r(s)
        rew_input = tf.concat([tf.reshape(ph_obv, [1, dim]), tf.reshape(a, [1, 2])], axis=1)  # r(s, a)







        with tf.variable_scope('skill_0/discrim/reward'):
            loss_reward = relu_net(rew_input, dout=1, **{})

        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'skill_0/discrim/reward')):
            save_dictionary[f'reward_0_{idx}'] = var

        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")




        # Get agent's traj's actions
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)


        NUM_DEMO = 100
        reward_all_traj = []
        for i in range(NUM_DEMO):
            print('Iter ', i)
            env = carEnv(demo=demo_pth)
            obs = env.reset()
            agent_actions = demonstrations[env.get_traj_id()]['actions']
            agent_actions = demonstrations[i]['actions']

            tm = 0
            # env.render()
            reward_one_traj = []
            while True:
                if tm >= len(agent_actions):
                    tm = -1
                action = agent_actions[tm][-1, :]

                obs, reward, done, info = env.step(action)

                loss_reward_ = sess.run(loss_reward, {ph_obv: obs, a: action})

                reward_one_traj.append(loss_reward_[0][0])

                if info['success'] == 1:
                    break

                if done == True:
                    break

                tm += 1

            # # show the reward map
            # start_obs, end_obs = env.get_start_goal_states()
            # start_obs, end_obs = start_obs[-1, :2], end_obs[-1, :]
            # for i in range(start_obs[0], end_obs[0], 0.1):
            #     for j in range(start_obs[1], end_obs[1], 0.1):

            reward_all_traj.append(reward_one_traj)
            env.close()

        with open('data/debug_data/reward_s_a_16obs_stop_400epoch.pkl', 'wb') as f:
            pickle.dump(reward_all_traj, f)



def visualize_learned_reward():
    # load reward
    log_path = f"data/yy/22_07_2022_17_00_38"  # r(s, a)
    log_path = f"data/yy/25_07_2022_18_57_41"  # r(s)
    log_path = 'data/yy/01_08_2022_23_58_24'  # r(s) stop


    # log_path = f'data/obs16/03_08_2022_16_26_14'  # r(s) -> 1000 epoch
    # log_path = 'data/obs16/03_08_2022_22_59_25'  # r(s, a) -> 1000 epoch

    # log_path = f'data/obs16/04_08_2022_17_23_23'  # r(s) -> 400 epoch
    # log_path = 'data/obs16/05_08_2022_14_49_34'  # r(s, a) -> 400 epoch


    demo_pth = 'src/demonstrations/safe_demo_16obs_stop.pkl'
    demo_pth = 'src/demonstrations/safe_demo_stop.pkl'

    dim = 36  # 52

    save_dictionary = {}
    env = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    with tf.Session() as sess:
        ph_obv = tf.placeholder(tf.float32, shape=(dim), name='ph_obv')
        a = tf.placeholder(tf.float32, shape=(2,), name='ph_a')




        rew_input = tf.reshape(ph_obv, [1, dim])  # r(s)
        # rew_input = tf.concat([tf.reshape(ph_obv, [1, dim]), tf.reshape(a, [1, 2])], axis=1)  # r(s, a)







        with tf.variable_scope('skill_0/discrim/reward'):
            loss_reward = relu_net(rew_input, dout=1, **{})

        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'skill_0/discrim/reward')):
            save_dictionary[f'reward_0_{idx}'] = var

        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")




        # Get agent's traj's actions
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)


        NUM_DEMO = 1
        for i in range(NUM_DEMO):
            print('Iter ', i)
            env = carEnv(demo=demo_pth)
            obs = env.reset()
            agent_actions = demonstrations[env.get_traj_id()]['actions']
            agent_actions = demonstrations[1]['actions']

            tm = 0
            # env.render()
            imgs = []
            start_obs, end_obs = env.get_start_goal_states()
            start_obs, end_obs = start_obs[-1, :2], end_obs[-1, :]
            fig = plt.figure(figsize=(12, 9))
            print(start_obs[1], end_obs[1], start_obs[0], end_obs[0])
            while True:
                plt.clf()
                if tm >= len(agent_actions):
                    tm = -1
                action = agent_actions[tm][-1, :]

                obs, reward, done, info = env.step(action)

                obv_obs = obs[:-4]

                vis_map = []

                # show the reward map
                for i in np.arange(start_obs[1], end_obs[1], 0.1):
                    ls = []
                    for j in np.arange(start_obs[0], end_obs[0], 0.1):
                        rew = sess.run(loss_reward, {ph_obv: np.concatenate([obv_obs, np.array([j, i, 0, 0])]), a: action})
                        ls.append(rew[0][0])
                    vis_map.append(ls)

                # if tm == 0:
                #     ax = sns.heatmap(np.array(vis_map), annot=True, )
                # else:
                #     ax = sns.heatmap(np.array(vis_map), cbar=False, annot=True, )
                ax = sns.heatmap(np.array(vis_map), annot=True, )
                ax.invert_yaxis()

                # print(end_obs[0], end_obs[1])
                # print(start_obs[0], start_obs[1])
                # plt.scatter(end_obs[0], end_obs[1],
                #             color='green',
                #             s=100, label='Target', alpha=0.6)
                #
                # plt.scatter(start_obs[0], start_obs[1],
                #             color='yellow',
                #             s=100, label='Start', alpha=0.6)



                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                imgs.append(image)

                if info['success'] == 1:
                    break

                if done == True:
                    break

                tm += 1

            save_video(imgs, f"data/visual_videos/ss/demo_0_8obs_stop.avi", fps=5)

            env.close()

        # with open('data/debug_data/reward_s_a_16obs_stop_400epoch.pkl', 'wb') as f:
        #     pickle.dump(reward_all_traj, f)

if __name__ == "__main__":
    show_safe_demo()
    # show_unsafe_demo()

    # visualize_learned_reward()