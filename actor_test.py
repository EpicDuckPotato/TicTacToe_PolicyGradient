from actor_class import Actor
import numpy as np
import tensorflow as tf

def test_prob_selection():
    with tf.Session() as sess:
        actor = Actor()
        sess.run(actor.var_init)
        probs = []

        for i in range(9):
            state = np.zeros((1, 9))
            state[0, i] = 1
            probs.append(actor.get_policy(state, sess)[0, i])

        state_batch = np.identity(9)
        action_batch = np.arange(9)

        autoprobs = actor.get_prob(state_batch, action_batch, sess)

        for i in range(9):
            if probs[i] != autoprobs[i]:
                print(probs[i])
                print(autoprobs[i])
            assert(probs[i] == autoprobs[i])
        print('All tests successful!')

test_prob_selection()