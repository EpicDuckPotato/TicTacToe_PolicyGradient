import tensorflow as tf
import numpy as np

class Actor:
    def __init__(self, learning_rate):
        #placeholder for input batch of board states
        self.states = tf.placeholder(shape=[None, 9], dtype=tf.float32, name='states')
        #the action we want to find the probability of
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
        #the advantage for this action
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name='advantage')

        self.fc1 = tf.layers.dense(self.states, 15, activation=tf.nn.relu)
        self.softmax = tf.layers.dense(self.fc1, 9, activation=tf.nn.softmax)

        self.policy = tf.identity(self.softmax, name='policy')

        self.one_hot_mask = tf.one_hot(self.action, 9, on_value=True, off_value=False, dtype=tf.bool)
        self.prob = tf.boolean_mask(self.policy, self.one_hot_mask)

        #not sure what to call this lol
        self.a_logprob = tf.multiply(self.advantage, tf.log(self.prob + 1e-9))

        #we minimize the negative to maximize the positive
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(-self.a_logprob)

        self.var_init = tf.global_variables_initializer()

        self.score = tf.identity(self.advantage)
        self.scores = []

    def train_batch(self, state_batch, action_batch, advantage_batch, sess):
        sess.run(self.optimizer, feed_dict={self.states: state_batch, self.action: action_batch, self.advantage: advantage_batch})
        self.scores.append(np.mean(sess.run(self.score, feed_dict={self.advantage: advantage_batch})))

    #given state, predict probability of a certain action
    def get_prob(self, state, action, sess):
        return sess.run(self.prob, feed_dict={self.states: state, self.action: action})

    def get_policy(self, state, sess):
        return sess.run(self.policy, feed_dict={self.states: state})

    #for testing
    def get_fc(self, state, sess):
        return sess.run(self.fc, feed_dict={self.states: state})

    def get_policy_batch(self, states, sess):
        return sess.run(self.policy, feed_dict={self.states: states})

    def save(self, sess, name):
        saver = tf.train.Saver()
        saver.save(sess,'C:/Anoop/Python Projects/TicTacToe_A2C/trained_models/' + name)

    def plot_scores(self, fname):
        import matplotlib.pyplot as plt
        plt.plot(self.scores)
        plt.savefig(fname + '.png')
        plt.show()
