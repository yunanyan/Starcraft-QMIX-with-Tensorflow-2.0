import tensorflow as tf
import math


def calculate_gain():
    return math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))

def calculate_fan_in_and_fan_out(num_input_fmaps):
    num_input_fmaps = num_input_fmaps
    receptive_field_size = 1
    fan_in = num_input_fmaps * receptive_field_size
    return fan_in

def bound(input_shape):
    fan = calculate_fan_in_and_fan_out(input_shape)
    gain = calculate_gain()
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std 
    return bound


class QMixNet(tf.keras.Model):
    def __init__(self, args):
        super( QMixNet, self).__init__()
        self.args = args

        if args.two_hyper_layers:
            self.hyper_w1 = tf.keras.Sequential()
            self.hyper_w1.add(tf.keras.layers.Dense(args.hyper_hidden_dim,input_shape=(args.state_shape,) ,activation='relu'))
            self.hyper_w1.add(tf.keras.layers.Dense(args.n_agents * args.qmix_hidden_dim,input_shape=(args.hyper_hidden_dim,),activation=None))

            

            self.hyper_w2 = tf.keras.Sequential()
            self.hyper_w2.add(tf.keras.layers.Dense(args.hyper_hidden_dim,input_shape=(args.state_shape,) ,activation='relu'))
            self.hyper_w2.add(tf.keras.layers.Dense(args.qmix_hidden_dim,input_shape=(args.hyper_hidden_dim,),activation=None))

            
        else:
            bound1 = bound(48)
            initializer = tf.random_uniform_initializer(
    minval=-bound1, maxval=bound1, seed=None
)
            self.hyper_w1 =tf.keras.layers.Dense(args.n_agents * args.qmix_hidden_dim,input_shape=(args.state_shape,) ,\
                                                 activation=None,kernel_initializer = initializer)

            
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 =tf.keras.layers.Dense(args.qmix_hidden_dim*1,input_shape=(args.state_shape,) ,activation=None,kernel_initializer = initializer)


        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = tf.keras.layers.Dense(args.qmix_hidden_dim, input_shape = (args.state_shape,),activation = None,kernel_initializer = initializer)

     
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = tf.keras.Sequential()
        self.hyper_b2.add(tf.keras.layers.Dense(args.qmix_hidden_dim,input_shape=(args.state_shape,),activation="relu",kernel_initializer = initializer))
        self.hyper_b2.add(tf.keras.layers.Dense(1,input_shape=(args.qmix_hidden_dim,),activation=None,kernel_initializer = initializer))

        

    def __call__(self,q_values,states):
        episode_num = q_values.shape[0]
        
        q_values = tf.reshape(q_values,[-1, 1, self.args.n_agents])  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        
        states =tf.reshape(states,[-1, self.args.state_shape])# (episode_num * max_episode_len, state_shape)
        

        w1 = tf.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        

        w1 = tf.reshape(w1,[-1, self.args.n_agents, self.args.qmix_hidden_dim])  # (1920, 5, 32)
        b1 = tf.reshape(b1,[-1, 1, self.args.qmix_hidden_dim] ) # (1920, 1, 32)

        hidden = tf.keras.activations.elu(tf.matmul(q_values, w1) + b1)  # (1920, 1, 32)

        

        w2 = tf.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = tf.reshape(w2,[-1, self.args.qmix_hidden_dim, 1] ) # (1920, 32, 1)
        b2 = tf.reshape(b2,[-1, 1, 1])  # (1920, 1， 1)

        q_total = tf.matmul(hidden, w2) + b2  # (1920, 1, 1)
        q_total = tf.reshape(q_total,[episode_num, -1, 1])  # (32, 60, 1)
        
        return q_total
        

        
        
