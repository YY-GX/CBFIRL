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

np.set_printoptions(4)

def parse_args():
    parser = argparse.ArgumentParser()
    now = datetime.now()
    parser.add_argument('--goal_reaching_weight', type=float, required=False, default=0.1)
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=f"data/saved_cbf_policies_flex/{now.strftime('%d_%m_%Y_%H_%M_%S')}")
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--log_path', type=str, default='data/obs16/04_08_2022_17_23_23')
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


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list

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
            s_agent = np.array([s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])
            a_agent = np.array([-1 * x_direction * a_range, -1 * y_direction * a_range])
            # combine the unsafe state of agent with other states
            s_agent = np.concatenate([s_u[:-1, :], s_agent.reshape(1, -1)], axis=0)

            a_agent = np.concatenate([A_u_init[i][:-1, :], a_agent.reshape(1, -1)], axis=0)
            S_u.append(s_agent)
            A_u.append(a_agent)
    return S_u, A_u


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

    # for i, demo in enumerate(demos):
    #     obvs = demo['observations']
    #     for j, obv in enumerate(obvs):
    #         topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
    #         demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    # return demos


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    demo_pth = args.demo_pth

    # log_path = f"data/yy/yy_500_15_07_2022_17_21_00"
    log_path = 'data/yy/28_07_2022_13_27_02'
    log_path = 'data/yy/22_07_2022_17_00_38'
    log_path = 'data/airl_safe_fusion_num_finetune/02_08_2022_16_16_59'  # r(s)
    # log_path = 'data/flex_initial_policy_r_s_a/02_08_2022_17_17_31'  # r(s, a)




    log_path = 'data/obs16/03_08_2022_16_26_14'  # r(s) 1000 epoch

    log_path = args.log_path

    # now = datetime.now()
    # save_path = f"data/saved_cbf_policies_flex/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    save_path = args.save_path

    env_graph = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    env = carEnv(demo=demo_pth)
    goal_reaching_weight = args.goal_reaching_weight



    accumulation_steps = config.INNER_LOOPS

    writer = SummaryWriter(save_path)



    with tf.Session() as sess:

        policy = GaussianMLPPolicy(name='action',
                                   env_spec=env_graph.spec,
                                   hidden_sizes=(32, 32))

        s, g, obv, other_as, a, loss_list, loss, acc_list, a_agent, indicator, obv_next, loss_safety, loss_goal_reaching =\
            build_training_graph(args.num_agents, env_graph, policy, goal_reaching_weight)
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

        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'reward')):
            print(var.name)
            save_dictionary[f'reward_0_{idx}'] = var




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
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)


        # remove other obs
        # demonstrations = demo_remove_top_k(demonstrations, config.TOP_K)


        S_s = [s for traj in demonstrations for s in traj['observations']]
        A_s = [a for traj in demonstrations for a in traj['actions']]

        # # Use the following code for the first time to generate unsafe states
        # S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=1.0)
        # with open('src/demonstrations/unsafe_states_10_10_16obs.pkl', 'wb') as f:
        #     pickle.dump([S_u, A_u], f)
        #
        # print(len(S_s), len(S_u))


        # Use the following code after the first time to generate unsafe states
        with open('src/demonstrations/unsafe_states_3_10_16obs.pkl', 'rb') as f:
            S_u, A_u = pickle.load(f)



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
            loss_safety_ls_safe, loss_goal_reaching_ls_safe = [], []
            for i, s_s in enumerate(S_s_iter):
                all_agents_actions = A_s_iter[i]
                other_a = all_agents_actions[:-1, :]  # YY: obs num * 2

                ob = state_remove_top_k(s_s, config.TOP_K)

                a_np = sess.run([a], feed_dict={s: s_s, g: g_np, obv: ob.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)]), other_as: other_a})

                ob_next, rew, done, info = env.step(a_np[0][-1, :])
                out, loss_safety_, loss_goal_reaching_ =\
                    sess.run([accumulate_ops, loss_safety, loss_goal_reaching], feed_dict={s: s_s, g: g_np, obv: ob.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)]),
                                                                 other_as: other_a, indicator: 1, obv_next: ob_next.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)])})
                loss_safety_ls_safe.append(loss_safety_)
                loss_goal_reaching_ls_safe.append(loss_goal_reaching_)

                # a_np, out = sess.run([a, accumulate_ops],
                #                  feed_dict={s: s_s, g: g_np, indicator: 1, obv: ob.reshape([1, 4 * args.num_agents]), other_as: other_a})

            loss_safety_ls_safe, loss_goal_reaching_ls_safe = np.mean(loss_safety_ls_safe), np.mean(loss_goal_reaching_ls_safe)

            loss_safety_ls_unsafe, loss_goal_reaching_ls_unsafe = [], []
            for i, s_u in enumerate(S_u_iter):
                all_agents_actions = A_u_iter[i]
                other_a = all_agents_actions[:-1, :]  # YY: obs num * 2

                # print(s_u)

                ob = state_remove_top_k(s_u, config.TOP_K)

                a_np = sess.run([a], feed_dict={s: s_s, g: g_np, obv: ob.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)]), other_as: other_a})

                ob_next, rew, done, info = env.step(a_np[0][-1, :])
                out, loss_safety_, loss_goal_reaching_ =\
                    sess.run([accumulate_ops, loss_safety, loss_goal_reaching], feed_dict={s: s_s, g: g_np, obv: ob.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)]),
                                                                 other_as: other_a, indicator: 0, obv_next: ob_next.reshape([1, 4 * min(config.TOP_K + 1, args.num_agents)])})
                loss_safety_ls_unsafe.append(loss_safety_)
                loss_goal_reaching_ls_unsafe.append(loss_goal_reaching_)
                # a_np, out = sess.run([a, accumulate_ops],
                #                      feed_dict={s: s_u, g: g_np, indicator: 0, obv: ob.reshape([1, 4 * args.num_agents]), other_as: other_a})

            loss_safety_ls_unsafe, loss_goal_reaching_ls_unsafe = np.mean(loss_safety_ls_unsafe), np.mean(loss_goal_reaching_ls_unsafe)

            if np.mod(istep // 10, 2) == 0:
                sess.run(train_step_h)
            else:
                sess.run(train_step_a)

            print('loss_safety_ls_safe: ', loss_safety_ls_safe)
            print('loss_goal_reaching_ls_safe: ', loss_goal_reaching_ls_safe)
            print('loss_safety_ls_unsafe: ', loss_safety_ls_unsafe)
            print('loss_goal_reaching_ls_unsafe: ', loss_goal_reaching_ls_unsafe)

            writer.add_scalar('loss_safety_ls_safe', loss_safety_ls_safe, istep)
            writer.add_scalar('loss_goal_reaching_ls_safe', loss_goal_reaching_ls_safe, istep)
            writer.add_scalar('loss_safety_ls_unsafe', loss_safety_ls_unsafe, istep)
            writer.add_scalar('loss_goal_reaching_ls_unsafe', loss_goal_reaching_ls_unsafe, istep)

        saver.save(sess, f"{save_path}/model")

    os.system('python scripts/airl_safe_test.py --policy_path ' + str(save_path))

if __name__ == '__main__':
    main()