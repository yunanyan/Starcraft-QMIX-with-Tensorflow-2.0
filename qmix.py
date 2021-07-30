import tensorflow as tf
import os
from basic_net import RNN
from qmix_net import QMixNet


def torch_gather(x, indices, gather_axis):
    # if pytorch gather indices are
    # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
    #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

    # create a tensor containing indices of each element
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped


class QMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                #map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_model(path_rnn)
                self.eval_qmix_net.load_model(path_qmix)
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.set_weights(self.eval_rnn.get_weights())
        self.target_qmix_net.set_weights(self.eval_qmix_net.get_weights())

       
        if args.optimizer == "RMS":
            self.optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        with tf.GradientTape() as tape:
            self.eval_parameters = list(self.eval_qmix_net.trainable_variables)+list(self.eval_rnn.trainable_variables)

            tape.watch(self.eval_parameters)
           
            episode_num = batch['o'].shape[0]
            self.init_hidden(episode_num)
            for key in batch.keys():  # 把batch里的数据转化成tensor
                if key == 'u':
                    batch[key] = tf.convert_to_tensor(batch[key], dtype=tf.int32)
                else:
                    batch[key] = tf.convert_to_tensor(batch[key], dtype=tf.float32)
            
                s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                                     batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                                     batch['terminated']

            batch_pad = tf.cast(batch["padded"],tf.float32)
            mask = tf.math.subtract(1,batch_pad)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

            # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
            q_evals, q_targets = self.get_q_values(batch, max_episode_len)
            if self.args.cuda:
                s = s.cuda()
                u = u.cuda()
                r = r.cuda()
                s_next = s_next.cuda()
                terminated = terminated.cuda()
                mask = mask.cuda()
                
            # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了

           
            # problem
            q_evals = tf.expand_dims(tf.squeeze(torch_gather(q_evals, tf.cast(u,dtype=tf.int64),3)),0)

            q_targets = q_targets.numpy()
            avail_u_next = avail_u_next.numpy()

            
            for i in range(q_targets.shape[0]):
                for j in range(q_targets.shape[1]):
                    for k in range(q_targets.shape[2]):
                        for l in range(q_targets.shape[3]):
                            if avail_u_next[i][j][k][l]==0.0:
                                q_targets[i][j][k][l]=-9999999
            
            q_targets = tf.convert_to_tensor(q_targets)
            avail_u_next = tf.convert_to_tensor(avail_u_next)
            # 得到target_q
            #q_targets[avail_u_next == 0.0] = - 9999999

            
            q_targets = tf.reduce_max(q_targets,axis=3)
            
            # this is required becase of the torch_gather function

            if len(q_evals.shape)!=3:
                q_evals = tf.squeeze(q_evals,0)

            q_total_eval = self.eval_qmix_net(q_evals, s)
            
            q_total_target = self.target_qmix_net(q_targets, s_next)

           
            targets = r + self.args.gamma * q_total_target * (1 - terminated)

            td_error = (q_total_eval - tf.stop_gradient(targets))
            masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
            self.eval_parameters = list(self.eval_qmix_net.trainable_variables)+list(self.eval_rnn.trainable_variables)

            # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
           
            loss = tf.reduce_sum(masked_td_error ** 2) / tf.reduce_sum(mask)
            print(loss)

        grads = tape.gradient(loss, self.eval_parameters)
        
        
        
        grads = [tf.clip_by_norm(g, self.args.grad_norm_clip)
             for g in grads]
        self.optimizer.apply_gradients(
            zip(grads, self.eval_parameters))

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            
            self.target_rnn.set_weights(self.eval_rnn.get_weights())
            self.target_qmix_net.set_weights(self.eval_qmix_net.get_weights())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]

        

        
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(tf.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(tf.tile(tf.expand_dims(tf.eye(self.args.n_agents),0),multiples=(episode_num,1,1)))        
            inputs_next.append(tf.tile(tf.expand_dims(tf.eye(self.args.n_agents),0),multiples=(episode_num,1,1)))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = tf.concat([tf.reshape(x,[episode_num * self.args.n_agents, -1]) for x in inputs], axis=1)
        inputs_next = tf.concat([tf.reshape(x,[episode_num * self.args.n_agents, -1]) for x in inputs_next], axis=1)

        
        
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id

            
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            
            
            q_eval,self.eval_hidden= self.eval_rnn(inputs,self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            
            
            q_target, self.target_hidden  = self.target_rnn(inputs_next, self.target_hidden )

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = tf.reshape(q_eval,[episode_num, self.n_agents, -1])
            
            q_target = tf.reshape(q_target,[episode_num, self.n_agents, -1])
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = tf.stack(q_evals, axis=1)
        q_targets = tf.stack(q_targets, axis=1)
        
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = tf.zeros([episode_num, self.n_agents, self.args.rnn_hidden_dim])
        self.target_hidden = tf.zeros([episode_num, self.n_agents, self.args.rnn_hidden_dim])

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.eval_qmix_net.save(self.model_dir + '/' + num + '_qmix_net_params.pkl')
        self.eval_rnn.save(self.model_dir + '/' + num + '_rnn_net_params.pkl')
