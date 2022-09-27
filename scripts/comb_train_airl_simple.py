#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import numpy as np

from envs.carEnv_simple import carEnv as carEnv_simple
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
from models.comb_airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
import pickle
import dowel
from dowel import logger, tabular
import argparse

import envs.config as config_file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser()
    now = datetime.now()
    main_pth = 'data/simple/airl_cbf_debug'
    parser.add_argument('--fusion_num', type=int, required=False, default=2000)
    parser.add_argument('--demo_num', type=int, required=False, default=1000)
    parser.add_argument('--epoch_num', type=int, required=False, default=51)
    parser.add_argument('--log_pth', type=str, default=main_pth + "/log")
    parser.add_argument('--share_pth', type=str, default=main_pth + "/share")
    parser.add_argument('--airl_pth', type=str, default=main_pth + "/airl")
    parser.add_argument('--cbf_pth', type=str, default=main_pth + "/cbf")
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/simple_3obs.pkl')
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


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]

    # print(str(i.name) for i in not_initialized_vars)  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))




args = parse_args()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True


log_path = args.log_pth
share_path = args.share_pth  # reward and action params
airl_path = args.airl_pth





# params
NUM_DEMO_USED = args.demo_num
EPOCH_NUM = args.epoch_num
demo_pth = args.demo_pth


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

demonstrations = demonstrations[:NUM_DEMO_USED]


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary_share = {}
    save_dictionary_airl = {}




    snapshotter = Snapshotter(f'{share_path}/skill')
    trainer = Trainer(snapshotter)

    # policy
    policy = GaussianMLPPolicy(name=f'action',
                               env_spec=env.spec,
                               hidden_sizes=(32, 32))

    # AIRL
    irl_model = AIRL(env=env, expert_trajs=demonstrations,
                     state_only=True, fusion=True,
                     max_itrs=10,
                     name=f'skill',
                     fusion_num=args.fusion_num,
                     policy=policy,
                     log_path=log_path,
                     env_eval=carEnv_simple(demo=demo_pth))

    # Add airl params
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'skill')):
        save_dictionary_airl[f'my_skill_{idx}'] = var

    # Add policy params
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'action')):
        save_dictionary_share[f'action_{idx}'] = var
        save_dictionary_airl[f'action_{idx}'] = var

    # Add reward params
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'skill/discrim/reward')):

        save_dictionary_share[f'reward_{idx}'] = var

    # restore policy and airl
    if os.path.exists(airl_path):
        saver = tf.train.Saver(save_dictionary_airl)
        saver.restore(sess, f"{airl_path}/model")



    # sess.run(tf.global_variables_initializer())
    initialize_uninitialized(tf.get_default_session())

    # Restore CBF NN
    cbf_path = args.cbf_pth
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



    baseline = LinearFeatureBaseline(env_spec=env.spec)

    sampler = None

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                index=0,
                sampler=sampler,
                irl_model=irl_model,
                generator_train_itrs=2,
                discrim_train_itrs=10,
                policy_ent_coeff=0.0,
                discount=0.99,
                max_kl_step=0.01)








    # Training
    sampler = RaySampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)
    algo._sampler = sampler

    logger.remove_all()
    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.TensorBoardOutput(f"{log_path}/policy_logs/"))
    logger.log('Starting up...')

    trainer.setup(algo, env)
    trainer.train(n_epochs=EPOCH_NUM, batch_size=10000)

    # save model
    saver_share = tf.train.Saver(save_dictionary_share)
    saver_share.save(sess, f"{share_path}/model")

    saver_airl = tf.train.Saver(save_dictionary_airl)
    saver_airl.save(sess, f"{airl_path}/model")

    # save unsafe states
    with open("src/demonstrations/unsafe_classified_debug.pkl", 'wb') as f:
        pickle.dump(irl_model.unsafe_states_ls, f)

# os.system("python scripts/airl_safe_test_simple.py --demo_path " + args.demo_pth + " --policy_path " + share_path)
# os.system("python scripts/airl_safe_test.py --demo_path " + args.demo_pth + " --policy_path " + share_path)