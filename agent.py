import numpy as np
import tensorflow as tf


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'qmix':
            from qmix import QMIX
            self.policy = QMIX(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon,  evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = tf.expand_dims(tf.convert_to_tensor(inputs, dtype=tf.float32),0)

        temp = avail_actions
        
        avail_actions = tf.expand_dims(tf.convert_to_tensor(avail_actions, dtype=tf.float32),0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        

        self.policy.eval_hidden = self.policy.eval_hidden.numpy()
       
        q_value,self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs,hidden_state)

        self.policy.eval_hidden = tf.convert_to_tensor(self.policy.eval_hidden)
        
        q_value=q_value.numpy()[0]

        

        
        for i in range(len(temp)):
            if temp[i]==0.0:
                q_value[i] = - float("inf")
       
        #, self.policy.eval_hidden[:, agent_num, :] 
        
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = tf.keras.backend.argmax(q_value)
        return action

    

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]   
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("model saved")
            self.policy.save_model(train_step)
            print("model saved-----------------------------------------------------------")
            
