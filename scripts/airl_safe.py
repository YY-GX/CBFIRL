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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_num', type=int, required=False, default=500)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args










args = parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# YY: params
NUM_DEMO_USED = 1000
EPOCH_NUM = 500




now = datetime.now()
log_path = f"data/yy/{now.strftime('%d_%m_%Y_%H_%M_%S')}"




irl_models = []
policies = []
algos = []
trainers = []


# YY: Load demonstrations and create environment
with open('src/demonstrations/safe_demo_correct.pkl', 'rb') as f:
    demonstrations = pickle.load(f)

# YY: only retain agent's actions
for traj in demonstrations:
    for i, a in enumerate(traj['actions']):
        traj['actions'][i] = a[-1, :]
    for i, o in enumerate(traj['observations']):
        traj['observations'][i] = traj['observations'][i].flatten()
env = GymEnv(carEnv(), max_episode_length=50)



demonstrations = [demonstrations[:NUM_DEMO_USED]]





config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{log_path}/skill_{index}')
        trainer = Trainer(snapshotter)

        irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
                         state_only=True, fusion=True,
                         max_itrs=5,
                         name=f'skill_{index}',
                         fusion_num=args.fusion_num)
        # for idx, var in enumerate(
        #     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        #                       scope=f'skill_{index}')):
        #     save_dictionary[f'my_skill_{index}_{idx}'] = var

        # policy = GaussianMLPPolicy(name=f'policy_{index}',
        #                            env_spec=env.spec,
        #                            hidden_sizes=(32, 32))
        # for idx, var in enumerate(
        #     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        #                       scope=f'policy_{index}')):
        #     save_dictionary[f'my_policy_{index}_{idx}'] = var


        policy = GaussianMLPPolicy(name=f'action',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))
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


    print("============================")
    print(save_dictionary)

    sess.run(tf.global_variables_initializer())
    # env_test = gym.make('InvertedPendulum-v2')
    env_test = carEnv()  # YY
    for i in range(len(demonstrations)):
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

        policy = policies[i]

        # # YY: save policy
        # with open(f'{log_path}/policy.pickle', 'wb') as handle:
        #     pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


        imgs = []

        done = False
        ob = env_test.reset()
        succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0
        EVAL_TRAJ_NUM = 100
        MAX_TIMESTEP = 1000





        # # Evaluation
        # while traj_cnt < EVAL_TRAJ_NUM:
        #     if not done:
        #         timestep_reach_flag = False
        #         if cnt > MAX_TIMESTEP:
        #             done = True
        #             info['success'] = False
        #             cnt = 0
        #             print(">> Reach {0} timestep".format(MAX_TIMESTEP))
        #             timestep_reach_flag = True
        #             continue
        #         if cnt % 200 == 0:
        #             print(">> cnt: ", cnt)
        #         ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
        #         imgs.append(env_test.render('rgb_array'))
        #         cnt += 1
        #     else:
        #         print(">> Eval traj num: ", traj_cnt)
        #         print(">> succ_cnt: ", succ_cnt)
        #         traj_cnt += 1
        #         succ_cnt = succ_cnt + 1 if info['success'] else succ_cnt
        #         if not info['success'] and not timestep_reach_flag:
        #             coll_cnt += 1
        #         print(">> coll_cnt: ", coll_cnt)
        #         ob = env_test.reset()
        #         done = False
        #         cnt = 0
        #
        # print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM, " trajs.")

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

        # for timestep in range(1000):
        #     ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
        #     imgs.append(env_test.render('rgb_array'))

        # print(env_test.render('rgb_array').shape)
        # imgs.append(env_test.render('rgb_array'))

        save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))
    env_test.close()

    saver = tf.train.Saver(save_dictionary)
    saver.save(sess, f"{log_path}/model")














# snapshotter = Snapshotter()
# with TFTrainer(snapshotter) as trainer:
#     trainer.restore('data/local/experiment/')
#     trainer.resume(n_epochs=500, batch_size=4000)
#     env = gym.make('InvertedDoublePendulum-v2')
#     for repeat in range(3):
#         ob = env.reset()
#         policy = trainer._algo.policy
#         policy.reset()
#         imgs = []
#         for timestep in range(1000):
#             ob, rew, done, info = env.step(policy.get_action(ob)[0])
#             imgs.append(env.render('rgb_array'))
#         save_video(imgs, os.path.join(f"policy_videos/skill_{repeat}.avi"))
#     env.close()
