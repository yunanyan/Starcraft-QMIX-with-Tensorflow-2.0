  
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


class RNN(tf.keras.Model):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        bound1 = bound(42)
            
        initializer = tf.random_uniform_initializer(
    minval=-bound1, maxval=bound1, seed=None
)

        self.fc1 = tf.keras.layers.Dense(args.rnn_hidden_dim,input_shape=(input_shape,),activation =None,kernel_initializer = initializer)
        
        self.rnn = tf.keras.layers.GRUCell(args.rnn_hidden_dim,activation = None)

        bound2 = bound(64)
        initializer2 = tf.random_uniform_initializer(
    minval=-bound2, maxval=bound2, seed=None
)

        self.fc2 = tf.keras.layers.Dense(args.n_actions, input_shape=(args.rnn_hidden_dim, ),activation = None,kernel_initializer = initializer2)

    def __call__(self, obs,hidden_state):
        
        x = tf.keras.activations.relu(self.fc1(obs))
        
        
        h_in = tf.reshape(hidden_state,[-1, self.args.rnn_hidden_dim])
        
        h,shit = self.rnn(x,h_in)
        
        q = self.fc2(h)
        return q,h


