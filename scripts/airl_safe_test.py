#!/usr/bin/env python3
import numpy as np

from envs.carEnv import carEnv
# from envs.carEnv_garage import carEnv
import os
from datetime import datetime
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler, MultiprocessingSampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
import pickle
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import argparse
from garage.experiment import deterministic
from models.architectures import relu_net


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fusion_num', type=int, required=False, default=500)
    # parser.add_argument('--demo_num', type=int, required=False, default=1000)
    # parser.add_argument('--epoch_num', type=int, required=False, default=500)
    # parser.add_argument('--eval_num', type=int, required=False, default=100)
    # parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--demo_path', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--demo_path', type=str, default='src/demonstrations/16obs_acc_farther_target.pkl')
    parser.add_argument('--policy_path', type=str, default='data/trpo_cbf/try/share')
    parser.add_argument('--seed', type=int, required=False, default=10)
    args = parser.parse_args()
    return args
# 'data/comb/16obs_airl_cbf_debug/share'

args = parse_args()
log_path = args.policy_path
demo_pth = args.demo_path

# YY: params
EVAL_TRAJ_NUM = 100



# Set seeds
seed = args.seed
print("seed: ", seed)
deterministic.set_seed(seed)




irl_models = []
policies = []
algos = []
trainers = []




env = GymEnv(carEnv(demo=demo_pth), max_episode_length=1000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # yy: visualize reward
    dim = 52
    save_dictionary_reward = {}
    # restore reward
    ph_obv = tf.placeholder(tf.float32, shape=(dim), name='ph_obv')
    # a = tf.placeholder(tf.float32, shape=(2,), name='ph_a')

    rew_input = tf.reshape(ph_obv, [1, dim])  # r(s)
    # rew_input = tf.concat([tf.reshape(ph_obv, [1, dim]), tf.reshape(a, [1, 2])], axis=1)  # r(s, a)

    with tf.variable_scope('skill/discrim/reward'):
        loss_reward = relu_net(rew_input, dout=1, **{})

    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill/discrim/reward')):
        save_dictionary_reward[f'reward_{idx}'] = var

    saver = tf.train.Saver(save_dictionary_reward)
    saver.restore(sess, f"{log_path}/model")












    save_dictionary = {}

    policy = GaussianMLPPolicy(name=f'action',
                               env_spec=env.spec,
                               hidden_sizes=(32, 32))
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'action')):
        save_dictionary[f'action_{idx}'] = var

    policies.append(policy)

    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model")

    # Evaluation
    env_test = carEnv(demo=demo_pth, is_test=True)  # YY
    env_test.render('no_vis')

    imgs = []

    done = False
    ob = env_test.reset()
    succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0
    video_traj_num = 10

    coll_ls, succ_ls = [], []
    last_timestep_state_ls = []


    # yy: visualize reward
    start_obs, end_obs = env_test.get_start_goal_states()
    start_obs, end_obs = start_obs[-1, :2], end_obs[-1, :]
    # fig = plt.figure(figsize=(12, 9))
    reward_imgs = []
    yy_cnt = 0



    while traj_cnt < EVAL_TRAJ_NUM:
        if not done:
            ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
            yy_cnt += 1

            # # yy: visualize reward
            # if traj_cnt <= 0:
            #     vis_map = []
            #     # show the reward map
            #     for i in np.arange(start_obs[1], end_obs[1], 0.1):
            #         ls = []
            #         for j in np.arange(start_obs[0], end_obs[0], 0.1):
            #             rew = sess.run(loss_reward, {ph_obv: np.concatenate([ob[:-4], np.array([j, i, 0.1, 0.1])])})
            #             ls.append(rew[0][0])
            #         vis_map.append(ls)
            #     ax = sns.heatmap(np.array(vis_map), annot=True, )
            #     ax.invert_yaxis()
            #     fig.canvas.draw()
            #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #     reward_imgs.append(image)

            if yy_cnt > 90:
                axd = 1



            if traj_cnt <= video_traj_num:
                imgs.append(env_test.render('rgb_array'))
        else:
            print(">> Eval traj num: ", traj_cnt)
            traj_cnt += 1
            coll_ls.append(info['collision_num'])
            coll_cnt = coll_cnt + info['collision_num']
            succ_cnt = succ_cnt + info['success']
            last_timestep_state_ls += env_test.unsafe_states
            ob = env_test.reset()
            done = False

    # yy: visualize reward
    print("length: ", len(reward_imgs))
    # save_video(reward_imgs, f"data/visual_videos/new_comb/v2_rew.avi", fps=5)


    print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM,
          " trajs.")
    print(coll_ls)
    print(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))
    print(succ_ls)
    with open(log_path + "/eval_results.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", Collision traj num: " + str(coll_cnt) + " out of " + str(EVAL_TRAJ_NUM) + " trajs.\n")
        f.write(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)) + '\n')
        f.write(str(coll_ls))

    save_video(imgs, os.path.join(f"{log_path}/policy_videos/eval_10traj.avi"))