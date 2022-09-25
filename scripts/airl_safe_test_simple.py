#!/usr/bin/env python3


from envs.carEnv_simple import carEnv
# from envs.carEnv_garage import carEnv
import os
from datetime import datetime
import gym
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


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fusion_num', type=int, required=False, default=500)
    # parser.add_argument('--demo_num', type=int, required=False, default=1000)
    # parser.add_argument('--epoch_num', type=int, required=False, default=500)
    # parser.add_argument('--eval_num', type=int, required=False, default=100)
    # parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--demo_path', type=str, default='src/demonstrations/simple_3obs.pkl')
    parser.add_argument('--policy_path', type=str, default='data/simple/airl_cbf_debug/share')
    args = parser.parse_args()
    return args

args = parse_args()

# YY: params
NUM_DEMO_USED = 1000
EPOCH_NUM = 1500
EVAL_TRAJ_NUM = 100

now = datetime.now()
#
# log_path = f"data/saved_cbf_policies/24_07_2022_23_29_23"  # r(s, a)
# log_path = f"data/saved_cbf_policies/25_07_2022_21_16_22"  # no reward loss added
# log_path = f"data/saved_cbf_policies/25_07_2022_21_59_46"  # r(T(s, \pi(s)))
# log_path = f"data/saved_cbf_policies/28_07_2022_16_59_37"


log_path = args.policy_path

irl_models = []
policies = []
algos = []
trainers = []


demo_pth = args.demo_path


# YY: Load demonstrations and create environment
with open(demo_pth, 'rb') as f:
    demonstrations = pickle.load(f)

# YY: only retain agent's actions
for traj in demonstrations:
    for i, a in enumerate(traj['actions']):
        traj['actions'][i] = a[-1, :]
    for i, o in enumerate(traj['observations']):
        traj['observations'][i] = traj['observations'][i].flatten()
env = GymEnv(carEnv(demo=demo_pth), max_episode_length=1000)

demonstrations = demonstrations[:NUM_DEMO_USED]



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
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
    env_test = carEnv(demo=demo_pth)  # YY
    env_test.render('no_vis')

    imgs = []

    done = False
    ob = env_test.reset()
    succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0
    video_traj_num = 10

    coll_ls, succ_ls = [], []
    last_timestep_state_ls = []
    while traj_cnt < EVAL_TRAJ_NUM:
        if not done:
            ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
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

    print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM,
          " trajs.")
    print(coll_ls)
    print(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))
    print(succ_ls)
    with open(log_path + "/eval_results.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", Collision traj num: " + str(coll_cnt) + " out of " + str(EVAL_TRAJ_NUM) + " trajs.\n")
        f.write(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))

    save_video(imgs, os.path.join(f"{log_path}/policy_videos/eval_simple_random.avi"))