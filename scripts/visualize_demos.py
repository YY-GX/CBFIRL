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
from envs.carEnv_vel import carEnv as carEnv_vel
import seaborn as sns
from global_utils.utils import *
import random

from models.vel_cbf import build_training_graph_init
from models.comb_cbf import build_training_graph_init as build_training_graph_init_4state


def show_safe_demo():
    demo_pth = 'src/demonstrations/safe_demo_16obs_stop.pkl'
    demo_pth = 'src/demonstrations/16obs_acc_farther_target.pkl'

    # Get agent's traj's actions
    with open(demo_pth, 'rb') as f:
        demonstrations = pickle.load(f)
    imgs = []
    for i in range(0, 10):
        env = carEnv(demo=demo_pth, seed=i)
        # # yy: velocity demo
        # env = carEnv_vel(demo=demo_pth, seed=i)
        env.reset()
        # agent_actions = demonstrations[env.get_traj_id()]['actions']
        agent_actions = demonstrations[i]['actions']
        # # yy: velocity actions
        # agent_actions = [ob[:, 2:] for ob in demonstrations[env.get_traj_id()]['observations']]

        tm = 0
        env.render('no_vis')

        while True:
            imgs.append(env.render('rgb_array'))
            if tm >= len(agent_actions):
                tm = -1
            action = agent_actions[tm][-1, :]
            obs, reward, done, info = env.step(action)
    
            if done == True:
                break
    
            tm += 1

        print(tm)
        env.close()
    save_video(imgs, f"data/visual_videos/demo/16_farther_target.avi")
        


def show_unsafe_demo():
    with open('src/demonstrations/unsafe_states_vel_top12.pkl', 'rb') as f:
        S_u, _ = pickle.load(f)
    random.shuffle(S_u)

    # with open('src/demonstrations/unsafe_states_16obs_close.pkl', 'rb') as f:
    #     S_u, _ = pickle.load(f)
    #     S_u = S_u[:10]

    # fig = plt.figure(figsize=(9, 4))

    size = 100
    fig = plt.figure(figsize=(9, 9))

    # # 1
    # s_ = np.array([[3.17693561e-01, 1.04224133e+00, -4.58403956e-04, 2.91030616e-01],
    #                    [7.20915675e-01, 1.20146918e+00, -6.04842119e-02, 2.36023106e-02],
    #                    [1.81653216e-01, 8.17508757e-01, -2.31951941e-02, -1.06028296e-01],
    #                    [3.60976666e-01, 5.99796593e-01, -3.26026753e-02, -9.46359783e-02],
    #                    [9.69331861e-01, 9.34264004e-01, -3.18852700e-02, 1.04853600e-01],
    #                    [7.04886138e-01, 4.64102238e-01, 2.44546141e-02, 1.53624892e-01],
    #                    [9.56991494e-01, 6.15198612e-01, -5.54611906e-02, -1.43265963e-01],
    #                    [3.61837506e-01, 1.45298409e+00, 3.53539549e-02, 1.60570771e-01],
    #                    [1.02801633e+00, 1.19346309e+00, 1.99165657e-01, 1.42859235e-01],
    #                    [9.74807858e-01, 1.41387892e+00, -8.45868737e-02, 1.28579391e-02],
    #                    [1.64279938e-02, 1.36938488e+00, -9.76583958e-02, 1.62398428e-01],
    #                    [4.61973995e-01, 7.05967844e-02, -1.73432156e-02, -1.04260698e-01],
    #                    [5.22402346e-01, 9.31208789e-01, -5.70394158e-01, -4.78162527e-01]])

    # # 2
    # s_ = np.array([[ 1.2132,  0.6727, -0.0084, -0.0141],
    #    [ 0.9607,  0.6837, -0.0153,  0.1246],
    #    [ 0.8423,  1.1289, -0.0711, -0.0384],
    #    [ 0.9398,  1.2857,  0.0735, -0.0497],
    #    [ 1.7411,  0.9739,  0.1093, -0.0915],
    #    [ 1.6691,  0.3678,  0.0378, -0.0756],
    #    [ 1.4384,  0.1674,  0.0192,  0.0164],
    #    [ 1.1022,  0.0538,  0.1072,  0.0047],
    #    [ 0.5425,  0.1114,  0.1402,  0.0301],
    #    [ 0.1919,  0.5605,  0.0043,  0.1608],
    #    [ 0.3556,  0.1575, -0.0851, -0.1306],
    #    [ 0.0709,  1.2817,  0.0218,  0.0156],
    #    [ 1.2412,  0.9734,  0.4953,  0.1285]])

    # # 3
    # s_ = np.array([[2.4849e-01, 5.3124e-01, -5.2372e-02, 1.4009e-01],
    #        [5.7848e-01, 2.7823e-01, -2.7767e-02, 4.8025e-02],
    #        [2.4119e-01, 8.7356e-01, 6.0317e-02, -1.5443e-02],
    #        [2.6124e-01, 1.0239e+00, 1.1442e-01, 1.4964e-01],
    #        [8.3424e-01, 5.0023e-01, -7.7870e-02, -2.5453e-02],
    #        [4.6343e-01, 1.3049e+00, 6.6911e-02, 1.4414e-01],
    #        [1.2499e+00, 1.8051e-01, -8.3307e-02, 7.5290e-03],
    #        [1.9965e-01, 1.4318e+00, -9.2927e-02, 8.5867e-02],
    #        [1.2424e+00, 5.9495e-01, -1.3943e-01, 1.0729e-01],
    #        [1.1557e+00, 8.2178e-01, 8.4135e-02, 1.3362e-03],
    #        [1.3548e+00, 3.4141e-01, 1.3730e-01, -3.0315e-02],
    #        [1.1028e+00, 9.8173e-01, 1.0365e-01, 4.5295e-03],
    #        [1.2564e-02, 1.7708e-01, 3.0930e-02, 7.9221e-01]])

    s_ = np.array([[1.2132, 0.6727, -0.0084, -0.0141],
                      [0.9607, 0.6837, -0.0153, 0.1246],
                      [0.8423, 1.1289, -0.0711, -0.0384],
                      [0.9398, 1.2857, 0.0735, -0.0497],
                      [1.7411, 0.9739, 0.1093, -0.0915],
                      [1.6691, 0.3678, 0.0378, -0.0756],
                      [1.4384, 0.1674, 0.0192, 0.0164],
                      [1.1022, 0.0538, 0.1072, 0.0047],
                      [0.5425, 0.1114, 0.1402, 0.0301],
                      [0.1919, 0.5605, 0.0043, 0.1608],
                      [0.3556, 0.1575, -0.0851, -0.1306],
                      [0.0709, 1.2817, 0.0218, 0.0156]])

    # # simple
    # s_ = np.array([[0.7162, 0.7162, 0., 0.],
    #           [0.4775, 0.4775, 0., 0.],
    #           [0.2387, 0.2387, 0., 0.],
    #           [7.9114, 5.1217, 3.7491, 2.3693]])
    #
    #
    S_u = [s_]





    for i, s_u in enumerate(S_u[:10]):
        # s_u = s_u[0]



        # print(np.array(s_u))
        # if s_u.shape[0] == 0:
        #     continue
        # print(s_u)
        plt.clf()
        # v1, v2 = s_u[-1, 2], s_u[-1, 3]
        # v1 /= max(np.abs(v1), np.abs(v2))
        # v2 /= max(np.abs(v1), np.abs(v2))
        # plt.arrow(s_u[-1, 0], s_u[-1, 1], v1, v2)

        # show the dynamic obstacles
        plt.scatter(s_u[:-1, 0], s_u[:-1, 1],
                    color='darkorange',
                    s=size, label='Dynamic Obstacle', alpha=0.6)
        # show the agent
        plt.scatter(s_u[-1, 0], s_u[-1, 1],
                    color='green',
                    s=size, label='Agent', alpha=0.6)

        plt.xlim(-0.0, 1.5)
        plt.ylim(-0.0, 1.5)

        plt.savefig('data/visual_videos/h_visualize/new_vel_ori_v2.png')

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

    log_path = 'data/vel/baselines/share'  # r(s) stop
    log_path = 'data/new_comb/j3/share'


    # log_path = f'data/obs16/03_08_2022_16_26_14'  # r(s) -> 1000 epoch
    # log_path = 'data/obs16/03_08_2022_22_59_25'  # r(s, a) -> 1000 epoch

    # log_path = f'data/obs16/04_08_2022_17_23_23'  # r(s) -> 400 epoch
    # log_path = 'data/obs16/05_08_2022_14_49_34'  # r(s, a) -> 400 epoch


    demo_pth = 'src/demonstrations/safe_demo_16obs_stop.pkl'
    # demo_pth = 'src/demonstrations/safe_demo_stop.pkl'

    dim = 52  # 8  # 52

    save_dictionary = {}
    env = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    with tf.Session() as sess:
        ph_obv = tf.placeholder(tf.float32, shape=(dim), name='ph_obv')
        a = tf.placeholder(tf.float32, shape=(2,), name='ph_a')




        rew_input = tf.reshape(ph_obv, [1, dim])  # r(s)
        # rew_input = tf.concat([tf.reshape(ph_obv, [1, dim]), tf.reshape(a, [1, 2])], axis=1)  # r(s, a)







        with tf.variable_scope('skill/discrim/reward'):
            loss_reward = relu_net(rew_input, dout=1, **{})

        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'skill/discrim/reward')):
            save_dictionary[f'reward_{idx}'] = var

        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")




        # Get agent's traj's actions
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)
            # Only use pos
            for i, demo in enumerate(demonstrations):
                demo['observations'] = [ob[:, :2] for ob in demo['observations']]
                demonstrations[i] = demo


        NUM_DEMO = 1
        for m in range(NUM_DEMO):
            print('Iter ', m)
            # yy: old
            env = carEnv(demo=demo_pth)

            # yy: new
            # env = carEnv_vel(demo=demo_pth)

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

                # yy: old
                obv_obs = obs[:-4]
                # # yy: new
                # obv_obs = obs[:-2]

                vis_map = []

                # show the reward map
                for i in np.arange(start_obs[1], end_obs[1], 0.1):
                    ls = []
                    for j in np.arange(start_obs[0], end_obs[0], 0.1):
                        # yy: old (vel: (0, 0) here)
                        rew = sess.run(loss_reward, {ph_obv: np.concatenate([obv_obs, np.array([j, i, 0, 0])]), a: action})
                        # # yy: new
                        # rew = sess.run(loss_reward,
                        #                {ph_obv: np.concatenate([obv_obs, np.array([j, i])]), a: action})
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

            save_video(imgs, f"data/visual_videos/new_comb/rewards.avi", fps=5)

            env.close()

        # with open('data/debug_data/reward_s_a_16obs_stop_400epoch.pkl', 'wb') as f:
        #     pickle.dump(reward_all_traj, f)



def visualize_h_value():
    num_obs = 16

    with tf.Session() as sess:
        s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list \
            = build_training_graph_init_4state(num_obs)
        sess.run(tf.global_variables_initializer())

        # Restore CBF NN
        cbf_path = "data/comb/16obs_airl_cbf_debug/cbf"
        # cbf_path = "data/simple/airl_cbf_debug/cbf"  # simple
        cbf_path = "data/new_comb_new_demo/cbf_posx_posy_velx_vely/cbf"
        cbf_path = "data/trpo_cbf/pretrain_airl/cbf"
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        print(">> Length of save_dictionary_cbf: ", len(save_dictionary_cbf))
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        saver_cbf = tf.train.Saver(save_dictionary_cbf)
        if os.path.exists(cbf_path):
            saver_cbf.restore(sess, f"{cbf_path}/model")

        # visualize state

        # ax = sns.heatmap(np.array(vis_map), annot=True, )
        # ax.invert_yaxis()

        # 1
        s_obs = np.array([[3.17693561e-01, 1.04224133e+00, -4.58403956e-04, 2.91030616e-01],
                           [7.20915675e-01, 1.20146918e+00, -6.04842119e-02, 2.36023106e-02],
                           [1.81653216e-01, 8.17508757e-01, -2.31951941e-02, -1.06028296e-01],
                           [3.60976666e-01, 5.99796593e-01, -3.26026753e-02, -9.46359783e-02],
                           [9.69331861e-01, 9.34264004e-01, -3.18852700e-02, 1.04853600e-01],
                           [7.04886138e-01, 4.64102238e-01, 2.44546141e-02, 1.53624892e-01],
                           [9.56991494e-01, 6.15198612e-01, -5.54611906e-02, -1.43265963e-01],
                           [3.61837506e-01, 1.45298409e+00, 3.53539549e-02, 1.60570771e-01],
                           [1.02801633e+00, 1.19346309e+00, 1.99165657e-01, 1.42859235e-01],
                           [9.74807858e-01, 1.41387892e+00, -8.45868737e-02, 1.28579391e-02],
                           [1.64279938e-02, 1.36938488e+00, -9.76583958e-02, 1.62398428e-01],
                           [4.61973995e-01, 7.05967844e-02, -1.73432156e-02, -1.04260698e-01]])

        # # 2
        # s_obs = np.array([[1.2132, 0.6727, -0.0084, -0.0141],
        #                [0.9607, 0.6837, -0.0153, 0.1246],
        #                [0.8423, 1.1289, -0.0711, -0.0384],
        #                [0.9398, 1.2857, 0.0735, -0.0497],
        #                [1.7411, 0.9739, 0.1093, -0.0915],
        #                [1.6691, 0.3678, 0.0378, -0.0756],
        #                [1.4384, 0.1674, 0.0192, 0.0164],
        #                [1.1022, 0.0538, 0.1072, 0.0047],
        #                [0.5425, 0.1114, 0.1402, 0.0301],
        #                [0.1919, 0.5605, 0.0043, 0.1608],
        #                [0.3556, 0.1575, -0.0851, -0.1306],
        #                [0.0709, 1.2817, 0.0218, 0.0156]])

        # # 3
        # s_obs = np.array([[2.4849e-01, 5.3124e-01, -5.2372e-02, 1.4009e-01],
        #        [5.7848e-01, 2.7823e-01, -2.7767e-02, 4.8025e-02],
        #        [2.4119e-01, 8.7356e-01, 6.0317e-02, -1.5443e-02],
        #        [2.6124e-01, 1.0239e+00, 1.1442e-01, 1.4964e-01],
        #        [8.3424e-01, 5.0023e-01, -7.7870e-02, -2.5453e-02],
        #        [4.6343e-01, 1.3049e+00, 6.6911e-02, 1.4414e-01],
        #        [1.2499e+00, 1.8051e-01, -8.3307e-02, 7.5290e-03],
        #        [1.9965e-01, 1.4318e+00, -9.2927e-02, 8.5867e-02],
        #        [1.2424e+00, 5.9495e-01, -1.3943e-01, 1.0729e-01],
        #        [1.1557e+00, 8.2178e-01, 8.4135e-02, 1.3362e-03],
        #        [1.3548e+00, 3.4141e-01, 1.3730e-01, -3.0315e-02],
        #        [1.1028e+00, 9.8173e-01, 1.0365e-01, 4.5295e-03]])

        # # 4
        # s_obs = np.array([[0.7162, 0.7162, 0., 0.],
        #        [0.4775, 0.4775, 0., 0.],
        #        [0.2387, 0.2387, 0., 0.]])

        fig = plt.figure(figsize=(12, 9))
        h_ls_ls = []
        v = np.array([-1, -1])
        # v = np.array([0.4953, 0.1285])
        # v = np.array([3.0930e-02, 7.9221e-01])
        # v = np.array([0, 0])

        # v = np.array([2e-2, 2e-2])  # simple
        for idx, i in enumerate(np.linspace(0, 1.5, 150)):
            if idx % 10 == 0:
                print(">> idx: ", idx)
            h_ls = []
            for j in np.linspace(0, 1.5, 150):
                s_agent = np.concatenate([np.array([j, i]), v])
                s_ = np.concatenate([s_obs, s_agent.reshape([1, 4])], axis=0)
                h_ = sess.run([h], feed_dict={s: s_})[0]
                h_ls.append(np.min(h_))
            h_ls_ls.append(h_ls)

        ax = sns.heatmap(np.array(h_ls_ls))
        ax.invert_yaxis()
        fig.canvas.draw()
        plt.savefig('data/visual_videos/h_visualize_strategy2/v_-1_-1.png')

        # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # s_ = np.array([[3.17693561e-01, 1.04224133e+00, -4.58403956e-04, 2.91030616e-01],
        #                    [7.20915675e-01, 1.20146918e+00, -6.04842119e-02, 2.36023106e-02],
        #                    [1.81653216e-01, 8.17508757e-01, -2.31951941e-02, -1.06028296e-01],
        #                    [3.60976666e-01, 5.99796593e-01, -3.26026753e-02, -9.46359783e-02],
        #                    [9.69331861e-01, 9.34264004e-01, -3.18852700e-02, 1.04853600e-01],
        #                    [7.04886138e-01, 4.64102238e-01, 2.44546141e-02, 1.53624892e-01],
        #                    [9.56991494e-01, 6.15198612e-01, -5.54611906e-02, -1.43265963e-01],
        #                    [3.61837506e-01, 1.45298409e+00, 3.53539549e-02, 1.60570771e-01],
        #                    [1.02801633e+00, 1.19346309e+00, 1.99165657e-01, 1.42859235e-01],
        #                    [9.74807858e-01, 1.41387892e+00, -8.45868737e-02, 1.28579391e-02],
        #                    [1.64279938e-02, 1.36938488e+00, -9.76583958e-02, 1.62398428e-01],
        #                    [4.61973995e-01, 7.05967844e-02, -1.73432156e-02, -1.04260698e-01],
        #                    [5.22402346e-01, 9.31208789e-01, 5.70394158e-01, 4.78162527e-01]])

        # h_ = sess.run([h], feed_dict={s: s_})[0]
        # print(h_, np.min(h_))



def visualize_h_value_vel():
    num_obs = 16

    with tf.Session() as sess:
        s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list \
            = build_training_graph_init(num_obs)
        sess.run(tf.global_variables_initializer())

        # Restore CBF NN
        cbf_path = "data/new_vel/baselines_bigger_region/cbf"
        cbf_path = "data/new_comb_new_demo/cbf_new_demo/cbf"
        cbf_path = "data/new_comb_new_demo/cbf_posx_posy/cbf"
        # cbf_path = "data/comb/baselines_repro_1/cbf"
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        print(">> Length of save_dictionary_cbf: ", len(save_dictionary_cbf))
        saver_cbf = tf.train.Saver(save_dictionary_cbf)
        if os.path.exists(cbf_path):
            saver_cbf.restore(sess, f"{cbf_path}/model")

        # visualize state

        # 1
        # s_obs = np.array([[ 0.4282,  0.2759],
        #                    [ 1.0574, -0.0631],
        #                    [ 0.7586,  0.3175]])


        s_obs = np.array([[1.2132, 0.6727, -0.0084, -0.0141],
                       [0.9607, 0.6837, -0.0153, 0.1246],
                       [0.8423, 1.1289, -0.0711, -0.0384],
                       [0.9398, 1.2857, 0.0735, -0.0497],
                       [1.7411, 0.9739, 0.1093, -0.0915],
                       [1.6691, 0.3678, 0.0378, -0.0756],
                       [1.4384, 0.1674, 0.0192, 0.0164],
                       [1.1022, 0.0538, 0.1072, 0.0047],
                       [0.5425, 0.1114, 0.1402, 0.0301],
                       [0.1919, 0.5605, 0.0043, 0.1608],
                       [0.3556, 0.1575, -0.0851, -0.1306],
                       [0.0709, 1.2817, 0.0218, 0.0156]])
        s_obs = s_obs[:, :2]



        fig = plt.figure(figsize=(12, 9))
        h_ls_ls = []

        for idx, i in enumerate(np.linspace(0, 1.5, 150)):
            if idx % 10 == 0:
                print(">> idx: ", idx)
            h_ls = []
            for j in np.linspace(0, 1.5, 150):
                s_agent = np.array([j, i])
                s_ = np.concatenate([s_obs, s_agent.reshape([1, 2])], axis=0)
                h_ = sess.run([h], feed_dict={s: s_})[0]
                h_ls.append(np.min(h_))
            h_ls_ls.append(h_ls)


        print(h_ls_ls)
        ax = sns.heatmap(np.array(h_ls_ls))
        ax.invert_yaxis()
        fig.canvas.draw()
        plt.savefig('data/visual_videos/new_h_visual/posxposy.png')



# def visualize_test_reward():



if __name__ == "__main__":
    # show_safe_demo()
    # show_unsafe_demo()
    # visualize_learned_reward()
    visualize_h_value()
    # visualize_h_value_vel()
