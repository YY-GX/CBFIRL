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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models.architectures import relu_net



np.set_printoptions(4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--goal_reaching_weight', type=float, required=False, default=0.1)
    parser.add_argument('--training_epoch', type=int, required=False, default=20000)
    parser.add_argument('--demo_path', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--policy_path', type=str, default='data/obs16/04_08_2022_17_23_23')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


def build_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()

    # variables_names = [v.name for v in tf.trainable_variables()]
    # for k in variables_names:
    #     print("Variable: ", k)

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
    train_step_a = optimizer.apply_gradients(gradient_vars_a)
    # re-initialize the accmulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))
    
    return zero_ops, accumulate_ops, train_step_h, train_step_a



def get_action_graph(num_agents, ob, policy):
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
    s = tf.placeholder(tf.float32, [num_agents, 4], name='ph_state')
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
    indicator = tf.placeholder(tf.float32)


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
    (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier_flex(
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

    return s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator, obv_next, loss_safety, loss_goal_reaching


# def build_training_graph(num_agents, env, policy, goal_reaching_weight):
#     # s is the state vectors of the agents, si = [xi, yi, vx_i, vy_i]
#     s = tf.placeholder(tf.float32, [num_agents, 4], name='ph_state')
#     # g is the goal states
#     g = tf.placeholder(tf.float32, [num_agents, 2], name='ph_goal')
#     # observation
#     # ob = tf.placeholder(tf.float32, [(min(num_agents, config.TOP_K + 1)) * 4, ])
#     obv = tf.placeholder(tf.float32, shape=(1, num_agents * 4), name='ph_obv')
#     obv_next = tf.placeholder(tf.float32, shape=(1, num_agents * 4), name='ph_obv_next')
#     # ob = tf.placeholder(tf.float32)
#     other_as = tf.placeholder(tf.float32, [min(num_agents - 1, config.TOP_K), 2], name='ph_other_as')
#
#
#     # x is difference between the state of each agent and other agents
#     x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # YY: shape: [num_agents, num_agents, 4]
#
#     # h is the CBF value of shape [num_agents, TOP_K, 1], where TOP_K represents
#     # the K nearest agents
#     h, mask, indices = core.network_cbf(x=x, r=config.DIST_MIN_THRES, indices=None)
#     # a is the control action of each agent, with shape [num_agents, 2]
#     # a = core.network_action(s=s, g=g, obs_radius=config.OBS_RADIUS, indices=indices)  #  YY: this is what we need to replace
#
#     a_agent = get_action_graph(num_agents, obv, policy)
#     a = tf.concat([other_as, a_agent], 0)
#
#
#     # compute the value of loss functions and the accuracies
#     # loss_dang is for h(s) < 0, s in dangerous set
#     # loss safe is for h(s) >=0, s in safe set
#     # acc_dang is the accuracy that h(s) < 0, s in dangerous set is satisfied
#     # acc_safe is the accuracy that h(s) >=0, s in safe set is satisfied
#     (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
#         h=h, s=s, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, indices=indices)
#     # loss_dang_deriv is for doth(s) + alpha h(s) >=0 for s in dangerous set
#     # loss_safe_deriv is for doth(s) + alpha h(s) >=0 for s in safe set
#     # loss_medium_deriv is for doth(s) + alpha h(s) >=0 for s not in the dangerous
#     # or the safe set
#     (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
#         ) = core.loss_derivatives(s=s, a=a, h=h, x=x, r=config.DIST_MIN_THRES,
#         indices=indices, ttc=config.TIME_TO_COLLISION, alpha=config.ALPHA_CBF)
#
#     # TODO: delete this one
#     # the distance between the a and the nominal a  YY: this is the goal reaching loss
#     loss_action = core.loss_actions(
#         s=s, g=g, a=a, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)
#
#
#     # # TODO: Add reward loss [r(s, a)]
#     # a_reward_input = tf.reshape(a[-1, :], [1, 2])
#     # rew_input = tf.concat([obv, a_reward_input], axis=1)
#     # with tf.variable_scope('reward'):
#     #     loss_reward = tf.reduce_sum(-relu_net(rew_input))
#
#
#     # TODO: Add reward loss [r(T(s, pi(s)))]
#     dsdt = tf.concat([tf.reshape(s[-1, 2:], (1, 2)), tf.reshape(a[-1, :], (1, 2))], axis=1)  # YY: dsdt = [vx, vy, ax, ay]
#     agent_state = tf.reshape((s[-1, :] + dsdt * config.TIME_STEP), (1, 4))
#     rew_input = tf.concat([obv_next[:, :-4], agent_state], axis=1)
#     with tf.variable_scope('reward'):
#         loss_reward = tf.reduce_sum(-relu_net(rew_input))
#
#
#     loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, goal_reaching_weight * loss_reward]  # YY: 0.01 original for loss_action
#     # loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv]  # YY: 0.01 original for loss_action
#     acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
#
#     weight_loss = [
#         config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
#     loss = 10 * tf.math.add_n(loss_list + weight_loss)
#
#     return s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, obv_next, loss_reward


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list


def main():
    # Params
    args = parse_args()
    goal_reaching_weight = args.goal_reaching_weight
    TRAIN_STEP = args.training_epoch
    demo_path = args.demo_path
    policy_path = args.policy_path


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_path = f"data/yy/17_07_2022_16_52_03"
    log_path = f"data/yy/22_07_2022_16_38_54"
    log_path = f"data/yy/22_07_2022_17_00_38"  # r(s, a)
    log_path = f"data/yy/25_07_2022_18_57_41"  # r(s)
    log_path = f"data/yy/27_07_2022_17_41_55"  # r(s)
    log_path = policy_path



    now = datetime.now()
    save_path = f"data/saved_cbf_policies/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

    writer = SummaryWriter(save_path)

    env_graph = GymEnv(carEnv(), max_episode_length=50)
    env = carEnv()



    accumulation_steps = config.INNER_LOOPS



    with tf.Session() as sess:

        policy = GaussianMLPPolicy(name='action',
                                   env_spec=env_graph.spec,
                                   hidden_sizes=(32, 32))

        # s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, obv_next, loss_reward = build_training_graph(args.num_agents, env, policy, goal_reaching_weight)
        # zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)

        s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator, obv_next, loss_safety, loss_goal_reaching =\
            build_training_graph(args.num_agents, env_graph, policy, goal_reaching_weight)
        zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)

        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)

        # all_other_vars = [v for v in tf.global_variables() if 'action' not in v.name]
        # sess.run(tf.variables_initializer(var_list=all_other_vars))
        sess.run(tf.global_variables_initializer())

        # Restore policy params

        save_dictionary = {}
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'action')):
            if idx > 6:
                break
            save_dictionary[f'action_0_{idx}'] = var


        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'reward')):
            print(var.name)
            save_dictionary[f'reward_0_{idx}'] = var


        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")

        state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)

        loss_lists_np = []
        acc_lists_np = []
        dist_errors_np = []
        init_dist_errors_np = []

        safety_ratios_epoch = []
        safety_ratios_epoch_lqr = []
        
        for istep in range(TRAIN_STEP):
            ob = env.reset()
            traj_id = env.get_traj_id()
            # read demonstrations
            with open(demo_path, 'rb') as f:
                demonstrations = pickle.load(f)

            all_actions = demonstrations[traj_id]['actions']

            # randomly generate the initial states and goals
            # s_np, g_np = core.generate_data(args.num_agents, config.DIST_MIN_THRES)
            s_np, g_np = env.get_start_goal_states()

            s_np_lqr, g_np_lqr = np.copy(s_np), np.copy(g_np)
            init_dist_errors_np.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))
            sess.run(zero_ops)

            demo_loss_ls, loss_reward_ls = [], []
            # run the system with the safe controller
            for i in range(len(all_actions)):

                # a_agent = policy.get_action(ob)[0]
                # YY: Get other_a
                other_a = all_actions[i][:-1, :]  # YY: obs num * 2

                # a_agent = sess.run([a_agent], feed_dict={ob: ob})
                # # YY: Get ob
                # ob, rew, done, info = env.step(a_agent)

                # if np.random.uniform() < config.ADD_NOISE_PROB:
                #     noise = np.random.normal(size=np.shape(a_agent)) * config.NOISE_SCALE
                #     a_agent = a_agent + noise

                # a_input = tf.concat([other_a, a_agent], 0)

                # computes the control input a_np using the safe controller

                a_np = sess.run([a], feed_dict={s: s_np, g: g_np, obv: ob.reshape([1, 36]), other_as: other_a})

                # a_np, out = sess.run([a, accumulate_ops], feed_dict={s: s_np, g: g_np, obv: ob.reshape([1, 36]), other_as: other_a})

                # if np.random.uniform() < config.ADD_NOISE_PROB:
                #     noise = np.random.normal(size=np.shape(a_np)) * config.NOISE_SCALE
                #     a_np = a_np + noise

                # YY: Get ob # simulate the system for one step
                ob_next, rew, done, info = env.step(a_np[0][-1, :])
                # ob_next, rew, done, info = env.step(a_np[-1, :])
                s_np = ob.reshape([-1, 4])

                out, loss_reward_s, loss_s = sess.run([accumulate_ops, loss_reward, loss],
                                     feed_dict={s: s_np, g: g_np, obv: ob.reshape([1, 36]), other_as: other_a, obv_next: ob_next.reshape([1, 36])})
                out = out[0]
                demo_loss_ls.append(loss_s)
                loss_reward_ls.append(loss_reward_s)
                ob = ob_next

                # # simulate the system for one step
                # s_np = s_np + np.concatenate([s_np[:, 2:], a_np], axis=1) * config.TIME_STEP

                # computes the safety rate
                safety_ratio = 1 - np.mean(core.ttc_dangerous_mask_np(
                    s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
                safety_ratio = np.mean(safety_ratio == 1)
                safety_ratios_epoch.append(safety_ratio)
                loss_list_np, acc_list_np = out[-2], out[-1]
                loss_lists_np.append(loss_list_np)
                acc_lists_np.append(acc_list_np)
                
                if np.mean(
                    np.linalg.norm(s_np[:, :2] - g_np, axis=1)
                    ) < config.DIST_MIN_CHECK:
                    break


            dist_errors_np.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))



            if np.mod(istep // 10, 2) == 0:
                sess.run(train_step_h)
            else:
                sess.run(train_step_a)







            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                # print('Step: {}, Loss: {}, Accuracy: {}'.format(
                #     istep, np.mean(loss_lists_np, axis=0),
                #     np.array(count_accuracy(acc_lists_np))))
                print('Step: {}'.format(istep))
                loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []


            writer.add_scalar('Loss_objective', np.mean(demo_loss_ls), istep)
            writer.add_scalar('Loss_reward', np.sum(loss_reward_ls), istep)



            if np.mod(istep, config.SAVE_STEPS) == 0 or istep + 1 == config.TRAIN_STEPS:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                saver.save(sess, os.path.join(args.save_path, 'model_iter_{}'.format(istep)))



        # Save model
        saver.save(sess, f"{save_path}/model")


    os.system('python scripts/airl_safe_test.py --policy_path ' + str(save_path))

if __name__ == '__main__':
    main()