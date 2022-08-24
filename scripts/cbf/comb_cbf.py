import sys

sys.dont_write_bytecode = True

import os
import h5py
import argparse
import numpy as np
import tensorflow as tf

import comb_core
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

np.set_printoptions(4)


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
    parser.add_argument('--is_hundred', type=int, default=0)
    parser.add_argument('--training_epoch', type=int, required=False, default=20000)
    args = parser.parse_args()
    return args


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list


# TODO: Add idx to each unsafe state and remember to save the idx
def generate_unsafe_states(S_s, A_s, num_ratio=0.1, num_unsafe_state_each_frame=3):
    # randomly select some frames to create unsafe states
    mask = random.sample(range(len(S_s)), int(num_ratio * len(S_s)))
    S_u_init, A_u_init = np.array(S_s)[mask], np.array(A_s)[mask]
    S_u, A_u = [], []
    for i, s_u in enumerate(S_u_init):
        # randomly select one obstacle
        rand_idx = random.choice(range(s_u[:-1, :].shape[0]))
        # create num_unsafe_state_each_frame unsafe stdemo_pthates around the selected obstacle
        for _ in range(num_unsafe_state_each_frame):
            s_agent = s_u[:-1, :]
            s_agent = s_agent[rand_idx, :]
            # add bias based on the state of the selected obstacle's state, then we get the unsafe state of our agent
            axis_range = config.DIST_MIN_THRES / np.sqrt(2)
            x_direction, y_direction = 2 * random.random() - 1, 2 * random.random() - 1
            x_bias, y_bias = x_direction * axis_range, y_direction * axis_range
            v_range, a_range = 1, 2  # DONE TODO: check the range of velocity and acceleration
            s_agent = np.array(
                [s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])
            a_agent = np.array([-1 * x_direction * a_range, -1 * y_direction * a_range])
            # combine the unsafe state of agent with other states
            s_agent = np.concatenate([s_u[:-1, :], s_agent.reshape(1, -1)], axis=0)

            a_agent = np.concatenate([A_u_init[i][:-1, :], a_agent.reshape(1, -1)], axis=0)
            S_u.append((s_agent, rand_idx))
            A_u.append(a_agent)
    return S_u, A_u


def demo_remove_top_k(demos, topk):
    for i, demo in enumerate(demos):
        obvs = demo['observations']
        for j, obv in enumerate(obvs):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
            demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    return demos


def build_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()

    # tensor to accumulate gradients over multiple steps
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]

    # count how many steps we have accumulated
    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)
    # add the gradient to the accumulation tensor
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    # divide the accumulated gradient by the number of accumulation steps
    gradient_vars = [(accumulator / accumulation_counter, var) \
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
    # seperate the gradient of CBF and the controller
    gradient_vars_h = []
    gradient_vars_a = []
    for accumulate_grad, var in gradient_vars:
        print(var.name)
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        elif 'reward' in var.name:
            continue
        else:
            raise ValueError

    train_step_h = optimizer.apply_gradients(gradient_vars_h)
    # re-initialize the accmulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_h


def get_action_graph(ob, policy):
    dist, mean, log_std = policy.build(ob, name='action').outputs
    samples = dist.sample(seed=deterministic.get_tf_seed_stream())
    return tf.reshape(samples, [1, 2])


def build_training_graph_init(num_obs):
    """
    Description:
        Train CBF NN. Only minimizing loss_barrier.

    Args:
        num_obs: number of obstacles

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    dang_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_dang_mask')  # dang mask
    safe_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_safe_mask')  # safe mask

    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13)
    h, mask = comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES)

    # Get loss_safe and loss_dang and corresponding accuracy
    (loss_dang, loss_safe, acc_dang, acc_safe) = comb_core.loss_barrier(h=h,
                                                                        dang_mask_reshape=dang_mask_reshape,
                                                                        safe_mask_reshape=safe_mask_reshape)

    # Compute objective loss
    loss_list = [2 * loss_dang, loss_safe]
    acc_list = [acc_dang, acc_safe]
    weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list


def build_training_graph_deriv(num_obs, policy):
    """
    Description:
        Train policy to be safer. Freeze CBF NN and only minimizing loss_deriv.

    Args:
        num_obs: number of obstacles
        policy: the policy for the agent

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    s_next = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4],
                            name='ph_state')  # next timestep state

    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13). yy: freeze CBF NN here!!
    h, mask = tf.stop_gradient(comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES))

    # Get action via the policy
    a = get_action_graph(tf.reshape(s, [1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy)

    # Get loss and accuracy for derivative part
    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = \
        comb_core.loss_derivatives(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)

    # Compute objective loss
    loss_list = [2 * loss_dang_deriv, loss_safe_deriv]
    acc_list = [acc_dang_deriv, acc_safe_deriv]
    weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, s_next, h, loss, loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, loss_list, acc_list


def train_init_CBF_NN(demo_path,
                      log_path,
                      cbf_save_path,
                      num_obs,
                      is_load_unsafe_states=False,
                      unsafe_ratio=0.1,
                      unsafe_state_path='src/demonstrations/comb_unsafe_states_1_10.pkl'):
    """
    Description:
        Train initial CBF NN

    Args:
        demo_path: path for demonstrations
        log_path: path for log files of tensorboard
        cbf_save_path: path for saving CBF NN
        num_obs: number of obstacles
        is_load_unsafe_states: load unsafe states or not
        unsafe_ratio: ratio of #unsafe states to #safe states
        unsafe_state_path: path for unsafe states

    Returns:
        None
    """

    # Tensorboard logger
    log_path = log_path
    writer = SummaryWriter(log_path)

    # Load demonstrations
    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    # Only retain topk nearest obs in demonstrations
    demonstrations = demo_remove_top_k(demonstrations, config.TOP_K)

    # Get set of states and actions from demonstrations
    S_s = [s for traj in demonstrations for s in traj['observations']]
    A_s = [a for traj in demonstrations for a in traj['actions']]

    # Collect unsafe states. Load if already exist.
    if is_load_unsafe_states:
        with open(unsafe_state_path, 'rb') as f:
            S_u, A_u = pickle.load(f)
    else:
        S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=unsafe_ratio)
        with open(unsafe_state_path, 'wb') as f:
            pickle.dump([S_u, A_u], f)

    # Add idx -1 to each s_s S_s
    S_s = [(s_s, -1) for s_s in S_s]
    # Divide S_s, S_u
    traj_length = 50
    S_s_eval = S_s[len(S_s) // 2:]
    S_u_eval = S_u[len(S_u) // 2:]
    S_s = S_s[:len(S_s) // 2]
    S_u = S_u[:len(S_u) // 2]
    TRAIN_STEPS = len(S_s) // traj_length
    unsafe_length = len(S_u) // TRAIN_STEPS
    print(">> Total training steps: ", TRAIN_STEPS)
    print(">> Training | Safe states num: ", len(S_s), ", Unsafe states num: ", len(S_u))
    print(">> Evaluation | Safe states num: ", len(S_s_eval), ", Unsafe states num: ", len(S_u_eval))

    # Start training
    print("---------- Start Training ----------")
    with tf.Session() as sess:
        # Construct training graph
        s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list = build_training_graph_init(num_obs)
        zero_ops, accumulate_ops, train_step_h = build_optimizer(loss)
        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
        # Prepare for saving CBF NN
        cbf_path = cbf_save_path
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)

        # Preparation
        loss_lists_np = []
        acc_lists_np = []
        # Iterate training steps
        for istep in range(TRAIN_STEPS):
            print('>> Step: ', istep)
            # Each iteration's amount of safe and unsafe states
            S_s_iter = S_s[istep * traj_length: (istep + 1) * traj_length]
            S_u_iter = S_u[istep * unsafe_length: (istep + 1) * unsafe_length]
            # Re-initialize accumulators for each iteration
            sess.run(zero_ops)

            # Main training parts
            # Mix up S_s & S_u
            for i in range(len(S_s_iter) + len(S_u_iter)):
                if len(S_s_iter) == 0:
                    s_, idx = S_u_iter.pop(0)
                elif len(S_u_iter) == 0:
                    s_, idx = S_s_iter.pop(0)
                elif random.random() > .5:  # select one safe state
                    s_, idx = S_s_iter.pop(0)
                else:
                    s_, idx = S_u_iter.pop(0)

                # Create dan & safe mask
                k = min(num_obs + 1, config.TOP_K + 1)
                dang_mask_reshape_ = np.zeros([k]).astype(bool)
                safe_mask_reshape_ = np.ones([k]).astype(bool)
                safe_mask_reshape_[-1] = False
                if idx != -1:  # no dangerous h
                    dang_mask_reshape_[idx] = True
                    safe_mask_reshape_[idx] = False

                out, loss_, loss_dang_, loss_safe_, acc_dang_, acc_safe_ =\
                    sess.run([accumulate_ops, loss, loss_dang, loss_safe, acc_dang, acc_safe],
                             feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})

                # Original codebase way to add accuracy and loss
                loss_list_np, acc_list_np = out[-2], out[-1]
                loss_lists_np.append(loss_list_np)
                acc_lists_np.append(acc_list_np)

            # Actual Optimization
            sess.run(train_step_h)

            # Evaluation
            EVAL_STEPS, EVAL_RATIO = 10, 0.5
            if np.mod(istep, EVAL_STEPS) == 0:
                s_u_eval, s_s_eval = \
                    random.sample(S_u_eval, int(len(S_u_eval) * EVAL_RATIO)), random.sample(S_s_eval, int(len(S_s_eval) * EVAL_RATIO))
                # s_u_eval, s_s_eval = [(s_u, 0) for s_u in s_u_eval], [(s_s, 1) for s_s in s_s_eval]
                eval_ls = s_u_eval + s_s_eval
                random.shuffle(eval_ls)
                acc_dang_ls, acc_safe_ls = [], []
                for _ in range(len(eval_ls)):
                    s_, idx = eval_ls.pop()
                    # Create dan & safe mask
                    k = min(num_obs + 1, config.TOP_K + 1)
                    dang_mask_reshape_ = np.zeros([k]).astype(bool)
                    safe_mask_reshape_ = np.ones([k]).astype(bool)
                    safe_mask_reshape_[-1] = False
                    if idx != -1:  # no dangerous h
                        dang_mask_reshape_[idx] = True
                        safe_mask_reshape_[idx] = False
                    acc_dang_, acc_safe_ = \
                        sess.run([acc_dang, acc_safe],
                                 feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})
                    if acc_dang_ != -1:
                        acc_dang_ls.append(acc_dang_)
                    acc_safe_ls.append(acc_safe_)
                writer.add_scalar('EVAL_ACC_DANGER', np.mean(acc_dang_ls), istep // EVAL_STEPS)
                writer.add_scalar('EVAL_ACC_SAFE', np.mean(acc_safe_ls), istep // EVAL_STEPS)

            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                acc_ls = np.array(count_accuracy(acc_lists_np))
                print('Step: {}, Loss: {}, Accuracy: {}'.format(
                    istep, np.mean(loss_lists_np, axis=0),
                    acc_ls))
                writer.add_scalar('Ori_Acc_barrier_dangerous', acc_ls[0], istep // config.DISPLAY_STEPS)
                writer.add_scalar('Ori_Acc_barrier_safe', acc_ls[1], istep // config.DISPLAY_STEPS)
                loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []
        # Save CBF NN
        saver_cbf.save(sess, f"{cbf_path}/model")


def train_policy(num_obs, policy, s_, s_next_):
    # Construct computational graph
    s, s_next, h, loss, loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, loss_list, acc_list = \
        build_training_graph_deriv(num_obs, policy)
    # Restore CBF NN
    cbf_path = "data/comb/1_cbf_init_single/cbf"
    save_dictionary_cbf = {}
    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'cbf')):
        save_dictionary_cbf[f'cbf_{idx}'] = var
    saver_cbf = tf.train.Saver(save_dictionary_cbf)
    if os.path.exists(cbf_path):
        saver_cbf.restore(tf.get_default_session(), f"{cbf_path}/model")
    # Compute loss
    loss_safety = tf.get_default_session().run([loss], feed_dict={s: s_, s_next: s_next_})

    # Return safety loss (i.e., loss_derivative). By minimizing this loss, the policy could be safer
    return loss_safety

if __name__ == '__main__':
    train_init_CBF_NN(demo_path="src/demonstrations/safe_demo_16obs_stop.pkl",
                      log_path="data/comb/1_cbf_init_single/log",
                      cbf_save_path="data/comb/1_cbf_init_single/cbf",
                      num_obs=16)