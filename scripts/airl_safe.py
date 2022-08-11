#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import numpy as np

from envs.carEnv import carEnv
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
import dowel
from dowel import logger, tabular
import argparse

import envs.config as config_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_num', type=int, required=False, default=2000)
    parser.add_argument('--demo_num', type=int, required=False, default=1000)
    parser.add_argument('--epoch_num', type=int, required=False, default=400)
    parser.add_argument('--eval_num', type=int, required=False, default=10)
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


def demo_remove_top_k(demos, topk):
    for i, demo in enumerate(demos):
        obvs = demo['observations']
        for j, obv in enumerate(obvs):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
            demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    return demos






args = parse_args()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True






# params
NUM_DEMO_USED = args.demo_num
EPOCH_NUM = args.epoch_num
EVAL_TRAJ_NUM = args.eval_num
demo_pth = args.demo_pth


now = datetime.now()
log_path = f"data/obs16/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

irl_models = []
policies = []
algos = []
trainers = []



# YY: Load demonstrations and create environment
with open(demo_pth, 'rb') as f:
    demonstrations = pickle.load(f)

demonstrations = demo_remove_top_k(demonstrations, config_file.TOP_K)

# YY: only retain agent's actions
for traj in demonstrations:
    for i, a in enumerate(traj['actions']):
        traj['actions'][i] = a[-1, :]
    for i, o in enumerate(traj['observations']):
        traj['observations'][i] = traj['observations'][i].flatten()
env = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)

demonstrations = [demonstrations[:NUM_DEMO_USED]]


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{log_path}/skill_{index}')
        trainer = Trainer(snapshotter)

        irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
                         state_only=False, fusion=True,
                         max_itrs=10,
                         name=f'skill_{index}',
                         fusion_num=args.fusion_num)

        policy = GaussianMLPPolicy(name=f'action',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))

        # Add policy params
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'action')):
            save_dictionary[f'action_{index}_{idx}'] = var

        # Add reward params
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill_{index}/discrim/reward')):
            print(var.name)
            print(f'reward_{index}_{idx}')
            save_dictionary[f'reward_{index}_{idx}'] = var

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = None

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=2,
                    discrim_train_itrs=10,
                    policy_ent_coeff=0.0,
                    discount=0.99,
                    max_kl_step=0.01)
        trainers.append(trainer)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    sess.run(tf.global_variables_initializer())
    env_test = carEnv(demo=demo_pth)  # YY
    for i in range(len(demonstrations)):
        # Training
        trainer = trainers[i]

        sampler = RaySampler(agents=policies[i],
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)
        algos[i]._sampler = sampler

        logger.remove_all()
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.TensorBoardOutput(f"{log_path}/policy_logs/"))
        logger.log('Starting up...')

        trainer.setup(algos[i], env)
        trainer.train(n_epochs=EPOCH_NUM, batch_size=10000)

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model")


        # Evaluation
        policy = policies[i]

        imgs = []

        done = False
        ob = env_test.reset()
        env_test.render('no_vis')
        succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0

        coll_ls, succ_ls = [], []
        while traj_cnt < EVAL_TRAJ_NUM:
            if not done:
                ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
                imgs.append(env_test.render('rgb_array'))
            else:
                print(">> Eval traj num: ", traj_cnt)
                traj_cnt += 1
                coll_ls.append(info['collision_num'])
                coll_cnt = coll_cnt + info['collision_num']
                succ_cnt = succ_cnt + info['success']
                ob = env_test.reset()
                done = False

        print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM,
              " trajs.")
        print(coll_ls)
        print(np.mean(coll_ls), np.std(coll_ls))
        print(succ_ls)

        with open(log_path + "/eval_results.txt", 'w', encoding='utf-8') as f:
            f.write(
                ">> Success traj num: " + str(succ_cnt) + ", Collision traj num: " + str(coll_cnt) + " out of " + str(
                    EVAL_TRAJ_NUM) + " trajs.\n")
            f.write(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))

        save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))
    env_test.close()

    # saver = tf.train.Saver(save_dictionary)
    # saver.save(sess, f"{log_path}/model")

