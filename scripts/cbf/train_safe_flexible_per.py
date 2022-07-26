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


np.set_printoptions(4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
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




def build_training_graph(num_agents, env, policy):
    # policy = GaussianMLPPolicy(name='action',
    #                            env_spec=env.spec,
    #                            hidden_sizes=(32, 32))



    # s is the state vectors of the agents, si = [xi, yi, vx_i, vy_i]
    s = tf.placeholder(tf.float32, [num_agents, 4], name='ph_state')
    # g is the goal states
    g = tf.placeholder(tf.float32, [num_agents, 2], name='ph_goal')
    # observation
    # ob = tf.placeholder(tf.float32, [(min(num_agents, config.TOP_K + 1)) * 4, ])
    obv = tf.placeholder(tf.float32, shape=(1, num_agents * 4), name='ph_obv')
    # ob = tf.placeholder(tf.float32)
    other_as = tf.placeholder(tf.float32, [min(num_agents - 1, config.TOP_K), 2], name='ph_other_as')
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


    loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv]  # YY: 0.01 original for loss_action
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    weight_loss = [
        config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list

def generate_unsafe_states(S_s, A_s, num_ratio=0.1):
    mask = random.sample(range(len(S_s)), int(num_ratio * len(S_s)))
    S_u_init, A_u_init = np.array(S_s)[mask], np.array(A_s)[mask]
    S_u, A_u = [], []
    for i, s_u in enumerate(S_u_init):
        s_agent = s_u[:-1, :]
        s_agent = s_agent[random.choice(range(s_agent.shape[0])), :]
        axis_range = config.DIST_MIN_THRES / np.sqrt(2)
        x_direction, y_direction = 2 * random.random() - 1, 2 * random.random() - 1
        x_bias, y_bias = x_direction * axis_range, y_direction * axis_range
        v_range, a_range = 1, 2  # DONE TODO: check the range of velocity and acceleration
        s_agent = np.array([s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])
        a_agent = np.array([-1 * x_direction * a_range, -1 * y_direction * a_range])
        s_agent = np.concatenate([s_u[:-1, :], s_agent.reshape(1, -1)], axis=0)
        a_agent = np.concatenate([A_u_init[i][:-1, :], a_agent.reshape(1, -1)], axis=0)
        S_u.append(s_agent)
        A_u.append(a_agent)
    return S_u, A_u



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_path = f"data/yy/yy_500_15_07_2022_17_21_00"
    now = datetime.now()
    save_path = f"data/saved_cbf_policies_flex/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

    env_graph = GymEnv(carEnv(), max_episode_length=50)
    env = carEnv()



    accumulation_steps = config.INNER_LOOPS



    with tf.Session() as sess:

        policy = GaussianMLPPolicy(name='action',
                                   env_spec=env_graph.spec,
                                   hidden_sizes=(32, 32))

        s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator = build_training_graph(args.num_agents, env_graph, policy)
        zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)

        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)

        # all_other_vars = [v for v in tf.global_variables() if 'action' not in v.name]
        # sess.run(tf.variables_initializer(var_list=all_other_vars))
        sess.run(tf.global_variables_initializer())

        # TODO: restore policy params

        save_dictionary = {}
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'action')):
            if idx > 6:
                break
            save_dictionary[f'action_0_{idx}'] = var

        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")






        # saver = tf.train.Saver()
        #
        # if args.model_path:
        #     saver.restore(sess, args.model_path)





        state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)

        loss_lists_np = []
        acc_lists_np = []
        dist_errors_np = []
        init_dist_errors_np = []

        safety_ratios_epoch = []
        safety_ratios_epoch_lqr = []



        # TODO: prepare S_s, S_u, A

        '''
        Prepare safe and unsafe states
        '''
        # read demonstrations
        with open('src/demonstrations/safe_demo_4.pkl', 'rb') as f:
            demonstrations = pickle.load(f)


        S_s = [s for traj in demonstrations for s in traj['observations']]
        A_s = [a for traj in demonstrations for a in traj['actions']]

        S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=0.1)
        with open('src/demonstrations/unsafe_states_1_10.pkl', 'wb') as f:
            pickle.dump([S_u, A_u], f)

        print(len(S_s), len(S_u))


        # Use the following code after the first time to save unsafe states
        # with open('src/demonstrations/unsafe_states_1_10.pkl', 'rb') as f:
        #     S_u, A_u = pickle.load(f)



        traj_length = 50

        TRAIN_STEPS = len(S_s) // traj_length

        unsafe_length = len(S_u) // TRAIN_STEPS


        '''
        Start training!
        '''
        print("Total training steps: ", TRAIN_STEPS)
        for istep in range(TRAIN_STEPS):
            print('Step: ', istep)
            '''
            Each iteration's amount of safe and unsafe states
            '''
            S_s_iter = S_s[istep * traj_length: (istep + 1) * traj_length]
            A_s_iter = A_s[istep * traj_length: (istep + 1) * traj_length]
            S_u_iter = S_u[istep * unsafe_length: (istep + 1) * unsafe_length]
            A_u_iter = A_u[istep * unsafe_length: (istep + 1) * unsafe_length]





            ob = env.reset()
            traj_id = env.get_traj_id()


            all_actions = demonstrations[traj_id]['actions']

            # randomly generate the initial states and goals
            # s_np, g_np = core.generate_data(args.num_agents, config.DIST_MIN_THRES)
            s_np, g_np = env.get_start_goal_states()

            s_np_lqr, g_np_lqr = np.copy(s_np), np.copy(g_np)
            init_dist_errors_np.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))
            sess.run(zero_ops)


            '''
            Main training parts
            '''
            for i, s_s in enumerate(S_s_iter):
                all_agents_actions = A_s_iter[i]
                other_a = all_agents_actions[:-1, :]  # YY: obs num * 2
                a_np, out = sess.run([a, accumulate_ops],
                                 feed_dict={s: s_s, g: g_np, indicator: 1, obv: ob.reshape([1, 36]), other_as: other_a})

            for i, s_u in enumerate(S_u_iter):
                all_agents_actions = A_u_iter[i]
                other_a = all_agents_actions[:-1, :]  # YY: obs num * 2
                a_np, out = sess.run([a, accumulate_ops],
                                     feed_dict={s: s_u, g: g_np, indicator: 1, obv: ob.reshape([1, 36]), other_as: other_a})

            if np.mod(istep // 10, 2) == 0:
                sess.run(train_step_h)
            else:
                sess.run(train_step_a)



            #
            # # run the system with the safe controller
            # for i in range(len(all_actions)):
            #
            #     # a_agent = policy.get_action(ob)[0]
            #     # YY: Get other_a
            #     other_a = all_actions[i][:-1, :]  # YY: obs num * 2
            #
            #     # a_agent = sess.run([a_agent], feed_dict={ob: ob})
            #     # # YY: Get ob
            #     # ob, rew, done, info = env.step(a_agent)
            #
            #     # if np.random.uniform() < config.ADD_NOISE_PROB:
            #     #     noise = np.random.normal(size=np.shape(a_agent)) * config.NOISE_SCALE
            #     #     a_agent = a_agent + noise
            #
            #     # a_input = tf.concat([other_a, a_agent], 0)
            #
            #     # computes the control input a_np using the safe controller
            #     a_np, out = sess.run([a, accumulate_ops], feed_dict={s: s_np, g: g_np, obv: ob.reshape([1, 36]), other_as: other_a})
            #
            #     if np.random.uniform() < config.ADD_NOISE_PROB:
            #         noise = np.random.normal(size=np.shape(a_np)) * config.NOISE_SCALE
            #         a_np = a_np + noise
            #
            #     # YY: Get ob # simulate the system for one step
            #     ob, rew, done, info = env.step(a_np[-1, :])
            #     s_np = ob.reshape([-1, 4])
            #
            #     # # simulate the system for one step
            #     # s_np = s_np + np.concatenate([s_np[:, 2:], a_np], axis=1) * config.TIME_STEP
            #
            #     # computes the safety rate
            #     safety_ratio = 1 - np.mean(core.ttc_dangerous_mask_np(
            #         s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            #     safety_ratio = np.mean(safety_ratio == 1)
            #     safety_ratios_epoch.append(safety_ratio)
            #     loss_list_np, acc_list_np = out[-2], out[-1]
            #     loss_lists_np.append(loss_list_np)
            #     acc_lists_np.append(acc_list_np)
            #
            #     if np.mean(
            #         np.linalg.norm(s_np[:, :2] - g_np, axis=1)
            #         ) < config.DIST_MIN_CHECK:
            #         break
            #
            # # run the system with the LQR controller without collision avoidance as the baseline
            # for i in range(accumulation_steps):
            #     state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
            #     s_ref_lqr = np.concatenate([s_np_lqr[:, :2] - g_np_lqr, s_np_lqr[:, 2:]], axis=1)
            #     a_lqr = -s_ref_lqr.dot(state_gain.T)
            #     s_np_lqr = s_np_lqr + np.concatenate([s_np_lqr[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            #     s_np_lqr[:, :2] = np.clip(s_np_lqr[:, :2], 0, 1)
            #     safety_ratio_lqr = 1 - np.mean(core.ttc_dangerous_mask_np(
            #         s_np_lqr, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            #     safety_ratio = np.mean(safety_ratio == 1)
            #     safety_ratios_epoch_lqr.append(safety_ratio_lqr)
            #
            #     if np.mean(
            #         np.linalg.norm(s_np_lqr[:, :2] - g_np_lqr, axis=1)
            #         ) < config.DIST_MIN_CHECK:
            #         break
            # dist_errors_np.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))
            #
            #
            #
            #
            # if np.mod(istep // 10, 2) == 0:
            #     sess.run(train_step_h)
            # else:
            #     sess.run(train_step_a)
            
            # if np.mod(istep, config.DISPLAY_STEPS) == 0:
            #     print('Step: {}, Loss: {}, Accuracy: {}'.format(
            #         istep, np.mean(loss_lists_np, axis=0),
            #         np.array(count_accuracy(acc_lists_np))))
            #     loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []

            # if np.mod(istep, config.SAVE_STEPS) == 0 or istep + 1 == config.TRAIN_STEPS:
            #     if not os.path.exists(args.save_path):
            #         os.makedirs(args.save_path)
            #     saver.save(sess, os.path.join(args.save_path, 'model_iter_{}'.format(istep)))

        saver.save(sess, f"{save_path}/model")

if __name__ == '__main__':
    main()