import numpy as np
import tensorflow as tf

from models.architectures import feedforward_energy, relu_net
from models.tf_util import discounted_reduce_sum
from utils.general import TrainingIterator
from utils.hyperparametrized import Hyperparametrized
from utils.math_utils import gauss_log_pdf, categorical_log_pdf, numerical_integral, numerical_mixture_integral
from utils import tensor_utils

LOG_REG = 1e-8
DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'

class ImitationLearning(object, metaclass=Hyperparametrized):
    def __init__(self):
        pass

    def set_demos(self, paths):
        if paths is not None:
            self.expert_trajs = paths
            self.expert_trajs_extracted = self.extract_paths(paths)

    @staticmethod
    def _compute_path_probs(paths, pol_dist_type=None, insert=True,
                            insert_key='a_logprobs', fix=False):
        """
        Returns a N x T matrix of action probabilities
        """
        if insert_key in paths[0]:
            return np.array([path[insert_key] for path in paths])

        if pol_dist_type is None:
            # try to  infer distribution type
            path0 = paths[0]
            if 'log_std' in path0['agent_infos']:
                pol_dist_type = DIST_GAUSSIAN
            elif 'prob' in path0['agent_infos']:
                pol_dist_type = DIST_CATEGORICAL
            else:
                raise NotImplementedError()

        # compute path probs
        Npath = len(paths)
        actions = [path['actions'] for path in paths]
        if pol_dist_type == DIST_GAUSSIAN:
            params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in paths]
            if fix:
                new_params = []
                for i in range(len(params)):
                    new_params.append((params[i][0], np.ones_like(params[i][1]) * -0.3671251))
                params = new_params
            # params = [(path['agent_infos']['mean'], path['agent_infos']['log_std'], path['agent_infos']['original_action']) for path in paths]
            # print(params[0].shape)
            path_probs = [gauss_log_pdf(params[i], actions[i]) for i in range(Npath)]
            # uncomment to add \pi probability adjustment (need to tune the random probability)
            # if paths[0].get('noise') is not None:
            #     for i in range(Npath):
            #         noise = max(paths[i]['noise'][0], 0.0)
            #         path_probs[i] = np.log(noise / (2.0 ** paths[i]['actions'].shape[1]) + (1 - noise) * np.exp(path_probs[i]))
        elif pol_dist_type == DIST_CATEGORICAL:
            params = [(path['agent_infos']['prob'],) for path in paths]
            path_probs = [categorical_log_pdf(params[i], actions[i]) for i in range(Npath)]
        else:
            raise NotImplementedError("Unknown distribution type")

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    @staticmethod
    def _insert_next_state(paths, pad_val=0.0):
        for path in paths:
            if 'observations_next' in path:
                continue
            nobs = path['observations'][1:]
            nact = path['actions'][1:]
            nobs = np.r_[nobs, pad_val*np.expand_dims(np.ones_like(nobs[0]), axis=0)]
            nact = np.r_[nact, pad_val*np.expand_dims(np.ones_like(nact[0]), axis=0)]
            path['observations_next'] = nobs
            path['actions_next'] = nact
        return paths

    @staticmethod
    def _insert_zero_noise(paths):
        for path in paths:
            if 'noise' in path:
                continue
            path['noise'] = np.zeros(path['observations'].shape[0])
        return paths

    @staticmethod
    def _insert_constant_noise(paths, noise):
        for path in paths:
            if 'noise' in path:
                continue
            path['noise'] = np.ones(path['observations'].shape[0]) * noise
        return paths

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=True):
        # print(paths)
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32) for key in keys]

    @staticmethod
    def sample_batch(*args, batch_size=32):
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    def fit(self, paths, **kwargs):
        raise NotImplementedError()

    def eval(self, paths, **kwargs):
        raise NotImplementedError()

    def _make_param_ops(self, vs):
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        assert len(self._params)>0
        self._assign_plc = [tf.placeholder(tf.float32, shape=param.get_shape(), name='assign_%s'%param.name.replace('/','_').replace(':','_')) for param in self._params]
        self._assign_ops = [tf.assign(self._params[i], self._assign_plc[i]) for i in range(len(self._params))]

    def get_params(self):
        params =  tf.get_default_session().run(self._params)
        assert len(params) == len(self._params)
        return params

    def set_params(self, params):
        tf.get_default_session().run(self._assign_ops, feed_dict={
            self._assign_plc[i]: params[i] for i in range(len(self._params))
        })


class TrajectoryIRL(ImitationLearning):
    """
    Base class for models that score entire trajectories at once
    """
    @property
    def score_trajectories(self):
        return True

    def eval_expert_probs(self, expert_paths, policy, insert=False):
        """
        Evaluate expert policy probability under current policy
        """
        if policy.recurrent:
            policy.reset([True]*len(expert_paths))
            expert_obs = self.extract_paths(expert_paths, keys=('observations',))[0]
            agent_infos = []
            for t in range(expert_obs.shape[1]):
                a, infos = policy.get_actions(expert_obs[:, t])
                agent_infos.append(infos)
            agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
            for key in agent_infos_stack:
                agent_infos_stack[key] = np.transpose(agent_infos_stack[key], axes=[1,0,2])
            agent_infos_transpose = tensor_utils.split_tensor_dict_list(agent_infos_stack)
            for i, path in enumerate(expert_paths):
                path['agent_infos'] = agent_infos_transpose[i]
        else:
            for path in expert_paths:
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
        return self._compute_path_probs(expert_paths, insert=insert)



class SingleTimestepIRL(ImitationLearning):
    """
    Base class for models that score single timesteps at once
    """
    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=False):
        return ImitationLearning.extract_paths(paths, keys=keys, stack=stack)

    @staticmethod
    def unpack(data, paths):
        lengths = [path['observations'].shape[0] for path in paths]
        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx+l])
            idx += l
        return unpacked

    @property
    def score_trajectories(self):
        return False

    def eval_expert_probs(self, expert_paths, policy, insert=False, fix=False):
        """
        Evaluate expert policy probability under current policy
        """
        for traj in expert_paths:
            if 'agent_infos' in traj:
                del traj['agent_infos']
            if 'a_logprobs' in traj:
                del traj['a_logprobs']

        if isinstance(policy, np.ndarray):
            return self._compute_path_probs(expert_paths, insert=insert)
        elif hasattr(policy, 'recurrent') and policy.recurrent:
            policy.reset([True]*len(expert_paths))
            expert_obs = self.extract_paths(expert_paths, keys=('observations',), stack=True)[0]
            agent_infos = []
            for t in range(expert_obs.shape[1]):
                a, infos = policy.get_actions(expert_obs[:, t])
                agent_infos.append(infos)
            agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
            for key in agent_infos_stack:
                agent_infos_stack[key] = np.transpose(agent_infos_stack[key], axes=[1,0,2])
            agent_infos_transpose = tensor_utils.split_tensor_dict_list(agent_infos_stack)
            for i, path in enumerate(expert_paths):
                path['agent_infos'] = agent_infos_transpose[i]
        else:
            cum_sum = 0
            for path in expert_paths:
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
                # path['agent_infos']['original_action'] = np.arctanh(path['actions'])
            #     cum_sum += np.sum(np.abs(path['actions'] - actions))
            # return cum_sum
        return self._compute_path_probs(expert_paths, insert=insert, fix=fix)

    def eval_mixture_probs(self, expert_paths, weights, policies, insert=False, fix=True):
        """
        Evaluate expert policy probability under current policy
        """
        cum_sum = 0
        for path in expert_paths:
            dict_pol = [policy.get_actions(path['observations'])[1] for policy in policies]
            mean = [dic['mean'] for dic in dict_pol]
            actions = np.einsum('i,ijk->jk', weights, mean)
            # actions_square = np.einsum('i,ijk->jk', weights, np.square(mean))

            log_std = [dic['log_std'] for dic in dict_pol]
            var = np.einsum('i,ijk->jk', weights, np.square(np.exp(log_std)))
            if fix:
                var = np.square(np.exp(np.ones_like(var) * -0.3671251))

            log_std = np.log(np.clip(np.sqrt(var), 1e-37, None))

            path['agent_infos'] = {'mean': actions, 'log_std': log_std}
        #     cum_sum += np.sum(np.abs(path['actions'] - actions))
        # return cum_sum
        return self._compute_path_probs(expert_paths, insert=insert, fix=True)

    def eval_numerical_integral(self, expert_paths, policy, fix=True):
        """
        Evaluate expert policy probability under current policy
        """
        for path in expert_paths:
            actions, agent_infos = policy.get_actions(path['observations'])
            path['agent_infos'] = agent_infos

        paths = expert_paths
        # compute path probs
        Npath = len(paths)
        actions = [path['actions'] for path in paths]
        params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in paths]
        if fix:
            new_params = []
            for i in range(len(params)):
                new_params.append((params[i][0], np.ones_like(params[i][1]) * -2.7671251))
            params = new_params
        path_probs = [numerical_integral(params[i], actions[i]) for i in range(Npath)]

        return np.array(path_probs)

    def eval_expert_mix(self, expert_paths, policies, weight, fix=True):
        """
        Evaluate expert policy probability under current policy
        """
        infos_mean = []
        infos_logstd = []
        for path in expert_paths:
            pol_infos_mean = []
            pol_infos_logstd = []
            for idx, policy in enumerate(policies):
                actions, agent_infos = policy.get_actions(path['observations'])
                pol_infos_mean.append(agent_infos['mean'])
                pol_infos_logstd.append(agent_infos['log_std'])
            infos_mean.append(pol_infos_mean)
            infos_logstd.append(pol_infos_logstd)

        paths = expert_paths
        # compute path probs
        Npath = len(paths)
        actions = [path['actions'] for path in paths]
        if fix:
            infos_logstd = np.ones_like(infos_logstd) * -2.7671251
        path_probs = [numerical_mixture_integral((infos_mean[i], infos_logstd[i]), actions[i], [weight]) for i in range(Npath)]

        return np.array(path_probs)


class GAIL(SingleTimestepIRL):
    """
    Generative adverserial imitation learning
    See https://arxiv.org/pdf/1606.03476.pdf

    This version consumes single timesteps.
    """
    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=relu_net,
                 discrim_arch_args={},
                 name='gail'):
        super(GAIL, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.set_demos(expert_trajs)

        # build energy model
        with tf.variable_scope(name) as vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            logits = discrim_arch(obs_act, **discrim_arch_args)
            self.predictions = tf.nn.sigmoid(logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.labels))
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self._make_param_ops(vs)


    def fit(self, trajs, batch_size=32, max_itrs=100, **kwargs):
        obs, acts = self.extract_paths(trajs)
        expert_obs, expert_acts = self.expert_trajs_extracted

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch = self.sample_batch(obs, acts, batch_size=batch_size)
            expert_obs_batch, expert_act_batch = self.sample_batch(expert_obs, expert_acts, batch_size=batch_size)
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict={
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.labels: labels,
                self.lr: 1e-3
            })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        obs, acts = self.extract_paths(paths)
        scores = tf.get_default_session().run(self.predictions,
                                              feed_dict={self.act_t: acts, self.obs_t: obs})

        # reward = log D(s, a)
        scores = np.log(scores[:,0]+LOG_REG)
        return self.unpack(scores, paths)


class AIRLStateAction(SingleTimestepIRL):
    """
    This version consumes single timesteps.
    """
    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=relu_net,
                 discrim_arch_args={},
                 l2_reg=0,
                 discount=1.0,
                 name='gcl'):
        super(AIRLStateAction, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.set_demos(expert_trajs)

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            with tf.variable_scope('discrim') as dvs:
                with tf.variable_scope('energy'):
                    self.energy = discrim_arch(obs_act, **discrim_arch_args)
                    # self.energy = -tf.nn.softplus(self.energy)
                # we do not learn a separate log Z(s) because it is impossible to separate from the energy
                # In a discrete domain we can explicitly normalize to calculate log Z(s)
                log_p_tau = -self.energy
                discrim_vars = tf.get_collection('reg_vars', scope=dvs.name)

            log_q_tau = self.lprobs

            if l2_reg > 0:
                reg_loss = l2_reg*tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau-log_pq)
            cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))
            self.discriminator_predict = tf.cast(log_p_tau > log_q_tau, tf.float32)
            self.acc = tf.reduce_mean(self.discriminator_predict * self.labels +
                                      (1 - self.discriminator_predict) * (1 - self.labels))

            self.loss = cent_loss + reg_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self._make_param_ops(_vs)


    def fit(self, paths, policy=None, batch_size=32, max_itrs=100, logger=None, lr=1e-3,**kwargs):
        #self._compute_path_probs(paths, insert=True)
        self.eval_expert_probs(paths, policy, insert=True)
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)
        obs, acts, path_probs = self.extract_paths(paths, keys=('observations', 'actions', 'a_logprobs'))
        expert_obs, expert_acts, expert_probs = self.extract_paths(self.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, path_probs, batch_size=batch_size)

            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs, batch_size=batch_size)

            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict={
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
            if it.itr == max_itrs - 1:
                acc = tf.get_default_session().run(self.acc, feed_dict={self.act_t: act_batch, self.obs_t: obs_batch,
                                                                        self.labels: labels, self.lprobs: lprobs_batch})
                logger.record_tabular('Discriminator_acc', acc)
        if logger:
            energy = tf.get_default_session().run(self.energy,
                                                        feed_dict={self.act_t: acts, self.obs_t: obs})
            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))

            energy = tf.get_default_session().run(self.energy,
                                                        feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs})
            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
            #logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy-logZ))
            logger.record_tabular('IRLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau', np.median(expert_probs))

        return mean_loss


    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        obs, acts = self.extract_paths(paths)

        energy  = tf.get_default_session().run(self.energy,
                                                    feed_dict={self.act_t: acts, self.obs_t: obs})
        energy = -energy[:,0]
        return self.unpack(energy, paths)


class GAN_GCL(TrajectoryIRL):
    """
    Guided cost learning, GAN formulation with learned partition function
    See https://arxiv.org/pdf/1611.03852.pdf
    """
    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=feedforward_energy,
                 discrim_arch_args={},
                 l2_reg = 0,
                 discount = 1.0,
                 init_itrs = None,
                 score_dtau=False,
                 state_only=False,
                 name='trajprior'):
        super(GAN_GCL, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.score_dtau = score_dtau
        self.set_demos(expert_trajs)

        # build energy model
        with tf.variable_scope(name) as vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, None, self.dO], name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, None, self.dU], name='act')
            self.traj_logprobs = tf.placeholder(tf.float32, [None, None], name='traj_probs')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            if state_only:
                obs_act = self.obs_t
            else:
                obs_act = tf.concat([self.obs_t, self.act_t], axis=2)

            with tf.variable_scope('discrim') as vs2:
                self.energy = discrim_arch(obs_act, **discrim_arch_args)
                discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs2.name)

            self.energy_timestep = self.energy
            # Don't train separate log Z because we can't fully separate it from the energy function
            if discount >= 1.0:
                log_p_tau = tf.reduce_sum(-self.energy, axis=1)
            else:
                log_p_tau = discounted_reduce_sum(-self.energy, discount=discount, axis=1)
            log_q_tau = tf.reduce_sum(self.traj_logprobs, axis=1, keep_dims=True)

            # numerical stability trick
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau-log_pq)
            cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))

            if l2_reg > 0:
                reg_loss = l2_reg*tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            #self.predictions = tf.nn.sigmoid(logits)
            self.loss = cent_loss + reg_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self._make_param_ops(vs)

    @property
    def score_trajectories(self):
        return False


    def fit(self, paths, policy=None, batch_size=32, max_itrs=100, logger=None, lr=1e-3,**kwargs):
        self._compute_path_probs(paths, insert=True)
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)
        obs, acts, path_probs = self.extract_paths(paths, keys=('observations', 'actions', 'a_logprobs'))
        expert_obs, expert_acts, expert_probs = self.extract_paths(self.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, path_probs, batch_size=batch_size)

            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs, batch_size=batch_size)
            T = expert_obs_batch.shape[1]

            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0)

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict={
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.labels: labels,
                self.traj_logprobs: lprobs_batch,
                self.lr: lr,
            })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            energy, dtau = tf.get_default_session().run([self.energy_timestep, self.d_tau],
                                                        feed_dict={self.act_t: acts, self.obs_t: obs,
                                                                   self.traj_logprobs: path_probs})
            #logger.record_tabular('GCLLogZ', logZ)
            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
            #logger.record_tabular('GCLAverageLogPtau', np.mean(-energy))
            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('IRLAverageDtau', np.mean(dtau))

            energy, dtau = tf.get_default_session().run([self.energy_timestep, self.d_tau],
                                                        feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs,
                                                                   self.traj_logprobs: expert_probs})
            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
            #logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy))
            logger.record_tabular('IRLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau', np.median(expert_probs))
            logger.record_tabular('IRLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        obs, acts = self.extract_paths(paths)

        scores = tf.get_default_session().run(self.energy,
                                          feed_dict={self.act_t: acts, self.obs_t: obs})
        scores = -scores[:,:,0]
        return scores

