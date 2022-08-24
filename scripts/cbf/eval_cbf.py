import sys
sys.dont_write_bytecode = True

import os
import h5py
import argparse
import numpy as np
import tensorflow as tf

import core
import config

# -------------------------------- new import --------------------------------
from garage.tf.policies import GaussianMLPPolicy
from envs.carEnv import carEnv
from garage.envs import GymEnv
import pickle
from garage.experiment import deterministic
import random
from datetime import datetime
from models.architectures import relu_net
import envs.config as config_file
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    now = datetime.now()
    parser.add_argument('--goal_reaching_weight', type=float, required=False, default=0.1)
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--share_path', type=str, default=None)
    parser.add_argument('--cbf_path', type=str, default=None)
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()
    return args


def get_action_graph(num_agents, ob, policy):
    print(ob.shape)
    dist, mean, log_std = policy.build(ob, name='action').outputs
    samples = dist.sample(seed=deterministic.get_tf_seed_stream())
    # print(samples[0, 0, :])
    # action = policy.action_space.unflatten_n(np.squeeze(samples, 1))[0]
    return tf.reshape(samples, [1, 2])




def build_training_graph(num_agents, env, policy, goal_reaching_weight=0.1):
    # policy = GaussianMLPPolicy(name='action',
    #                            env_spec=env.spec,
    #                            hidden_sizes=(32, 32))



    # s is the state vectors of the agents, si = [xi, yi, vx_i, vy_i]
    s = tf.placeholder(tf.float32, [num_agents, 4])
    # g is the goal states
    g = tf.placeholder(tf.float32, [num_agents, 2], name='ph_goal')
    # observation
    # ob = tf.placeholder(tf.float32, [(min(num_agents, config.TOP_K + 1)) * 4, ])
    obv = tf.placeholder(tf.float32, shape=(1, min(config.TOP_K + 1, num_agents) * 4), name='ph_obv')
    obv_next = tf.placeholder(tf.float32, shape=(1, min(config.TOP_K + 1, num_agents) * 4), name='ph_obv_next')
    # ob = tf.placeholder(tf.float32)
    # other_as = tf.placeholder(tf.float32, [min(num_agents - 1, config.TOP_K), 2], name='ph_other_as')
    other_as = tf.placeholder(tf.float32, [num_agents - 1, 2], name='ph_other_as')
    # Indicator to indicate the safe and unsafe
    indicator = tf.placeholder(tf.int32)


    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # YY: shape: [num_agents, num_agents, 4]



    # a = tf.placeholder(tf.float32, [num_agents, 2])


    # h is the CBF value of shape [num_agents, TOP_K, 1], where TOP_K represents
    # the K nearest agents
    h, mask, indices = core.network_cbf(x=x, r=config.DIST_MIN_THRES, indices=None)
    # a is the control action of each agent, with shape [num_agents, 2]
    # a = core.network_action(s=s, g=g, obs_radius=config.OBS_RADIUS, indices=indices)  #  YY: this is what we need to replace

    a_agent = get_action_graph(num_agents, obv, policy)
    # a_agent = policy.get_action(ob)[0]
    # print(a_agent)
    a = tf.concat([other_as, a_agent], 0)


    # compute the value of loss functions and the accuracies
    # loss_dang is for h(s) < 0, s in dangerous set
    # loss safe is for h(s) >=0, s in safe set
    # acc_dang is the accuracy that h(s) < 0, s in dangerous set is satisfied
    # acc_safe is the accuracy that h(s) >=0, s in safe set is satisfied
    (loss_dang, loss_safe, acc_dang, acc_safe, dang_h, safe_h) = core.loss_barrier_flex(
        h=h, s=s, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, indicator=indicator, indices=indices)
    # loss_dang_deriv is for doth(s) + alpha h(s) >=0 for s in dangerous set
    # loss_safe_deriv is for doth(s) + alpha h(s) >=0 for s in safe set
    # loss_medium_deriv is for doth(s) + alpha h(s) >=0 for s not in the dangerous
    # or the safe set
    (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
        ) = core.loss_derivatives_flex(s=s, a=a, h=h, x=x, r=config.DIST_MIN_THRES,
        indices=indices, ttc=config.TIME_TO_COLLISION, indicator=indicator, alpha=config.ALPHA_CBF)

    # TODO: delete this one
    # the distance between the a and the nominal a  YY: this is the goal reaching loss
    # loss_action = core.loss_actions(
    #     s=s, g=g, a=a, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)


    # # TODO: Add reward loss [r(s, a)]
    # a_reward_input = tf.reshape(a[-1, :], [1, 2])
    # rew_input = tf.concat([obv, a_reward_input], axis=1)
    # with tf.variable_scope('reward'):
    #     loss_reward = tf.reduce_sum(-relu_net(rew_input))


    # TODO: Add reward loss [r(T(s, pi(s)))]
    dsdt = tf.concat([tf.reshape(s[-1, 2:], (1, 2)), tf.reshape(a[-1, :], (1, 2))], axis=1)  # YY: dsdt = [vx, vy, ax, ay]
    agent_state = tf.reshape((s[-1, :] + dsdt * config.TIME_STEP), (1, 4))
    rew_input = tf.concat([obv_next[:, :-4], agent_state], axis=1)
    with tf.variable_scope('reward'):
        loss_reward = tf.reduce_sum(-relu_net(rew_input))



    # goal_reaching_weight = 0
    loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, goal_reaching_weight * loss_reward]  # YY: 0.01 original for loss_action
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    loss_safety, loss_goal_reaching = tf.math.add_n([2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv]), loss_reward

    weight_loss = [
        config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator, obv_next, loss_safety, loss_goal_reaching, h, dang_h, safe_h


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list

def demo_remove_top_k(demos, topk):
    for i, demo in enumerate(demos):
        obvs = demo['observations']
        for j, obv in enumerate(obvs):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
            demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    return demos

def state_remove_top_k(state, topk):
    # print(np.sum(np.square((state[:-1, :] - state[-1, :])[:, :2]), axis=1))
    topk_mask = np.argsort(np.sum(np.square((state[:-1, :] - state[-1, :])[:, :2]), axis=1))[:topk]
    return np.concatenate([state[:-1, :][topk_mask, :], state[-1, :][None, :]], axis=0)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    demo_pth = args.demo_pth

    share_path = args.share_path
    cbf_path = args.cbf_path

    env_graph = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    env = carEnv(demo=demo_pth)
    goal_reaching_weight = args.goal_reaching_weight

    writer = SummaryWriter(share_path)



    with tf.Session() as sess:

        policy = GaussianMLPPolicy(name='action',
                                   env_spec=env_graph.spec,
                                   hidden_sizes=(32, 32))

        s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator, obv_next, loss_safety, loss_goal_reaching, h, dang_h, safe_h =\
            build_training_graph(args.num_agents, env_graph, policy, goal_reaching_weight)

        sess.run(tf.global_variables_initializer())


        # Restore cbf params
        save_dictionary_cbf = {}
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)
        if os.path.exists(cbf_path):
            print("Reading CBF path ... ")
            saver_cbf.restore(sess, f"{cbf_path}/model")


        '''
        Prepare safe and unsafe states
        '''
        # read demonstrations
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)

        S_s = [s for traj in demonstrations for s in traj['observations']]
        A_s = [a for traj in demonstrations for a in traj['actions']]

        # # Use the following code for the first time to generate unsafe states
        # S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=1.0)
        # with open('src/demonstrations/unsafe_states_10_10_16obs.pkl', 'wb') as f:
        #     pickle.dump([S_u, A_u], f)
        #
        # print(len(S_s), len(S_u))

        # Use the following code after the first time to generate unsafe states
        with open('src/demonstrations/unsafe_states_1_10_16obs.pkl', 'rb') as f:
            S_u, A_u = pickle.load(f)

        S_u_eval = S_u[len(S_u) // 2:]


        # print(len(S_s), len(S_u))

        # for s_s in S_s[:num]:
        #     print(s_s)
        #     h_, dang_h_, safe_h_ = sess.run([h, dang_h, safe_h], feed_dict={s: s_s, indicator: 1})
        #
        #     # print(h_)
        #     print(dang_h_)
        #     print(safe_h_)
        #     print("------------------------------------------------------------------")
        #
        # print("======================================================================")

        acc_dang_ls, acc_safe_ls = [], []
        S_u_eval = S_u_eval[-int(len(S_u_eval) * 0.01):]
        print(len(S_u_eval))
        for i, s_u in enumerate(S_u_eval):
            if i % 100 == 0:
                print(i)
            # print(s_u)
            h_, dang_h_, safe_h_ = sess.run([h, dang_h, safe_h], feed_dict={s: s_u, indicator: 0})

            num_dang = tf.cast(tf.shape(dang_h_)[0], tf.float32)
            acc_dang = tf.reduce_sum(tf.cast(
                tf.less_equal(dang_h_, 0), tf.float32)) / (1e-5 + num_dang)

            num_safe = tf.cast(tf.shape(safe_h_)[0], tf.float32)
            acc_safe = tf.reduce_sum(tf.cast(
                tf.greater(safe_h_, 0), tf.float32)) / (1e-5 + num_safe)

            acc_dang_ls.append(acc_dang.eval(session=tf.compat.v1.Session())   )
            acc_safe_ls.append(acc_safe.eval(session=tf.compat.v1.Session())   )
            # # print(h_)
            # print(dang_h_)
            # print(safe_h_)
            # print("------------------------------------------------------------------")
        print(acc_dang_ls)
        print(np.mean(acc_dang_ls), np.mean(acc_safe_ls))
        # s_rand = np.random.rand(17, 4)
        # h_, dang_h_, safe_h_ = sess.run([h, dang_h, safe_h], feed_dict={s: s_rand, indicator: 0})
        # print(s_rand)
        # print(h_)

if __name__ == '__main__':
    main()