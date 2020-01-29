import tensorflow as tf
from board_class import Board
from memory_class import Memory
from critic_class import Critic
from actor_class import Actor
import numpy as np

gamma = 0.5#discount factor
batch_size = 200

def fix_policy(state, policy):
    for i in range(9):
        if state[i] != 0:
            policy[i] = 0
    probsum = np.sum(policy)
    return policy/probsum

def choose_action(state, actor, sess):
    policy = actor.get_policy(state.reshape((1, 9)), sess)[0]
    policy = fix_policy(state, policy)
    return np.random.choice(9, p=policy)

#we don't want the network to be different for X and O, so we make each player see the board as X would
def state_from_board(board, counter):
    state = np.array(board.board)
    if counter == 1:
        state = -state
    state = state
    return state

memory = Memory(3000)

actor = Actor(0.001)
critic = Critic(0.001)

with tf.Session() as sess:
    sess.run(actor.var_init)
    sess.run(critic.var_init)
    for game in range(4500):
        if game%100 == 0:
            print(game)
        board = Board()
        winner = ''
        counter = 0
        symbols = ['X', 'O']
        #we need to store samples temporarily because we don't get their values till the end of each game
        samples = []#each sample contains state, action, reward, and next state
        while winner == '':
            state = state_from_board(board, counter)

            action = choose_action(state, actor, sess)

            current_sample = []
            current_sample.append(state)
            current_sample.append(action)

            winner = board.setSquare(action, symbols[counter])
            current_sample.append(0.5)#placeholder reward. we change this when we know the winner

            samples.append(current_sample)
            #switch to next player
            counter = (counter + 1)%2

        #lol this is so ugly
        xreward = 0
        if winner == 'X':
            xreward = 0.5
        elif winner == 'O':
            xreward = -0.5

        #add the next state to each sample and set rewards based on winner
        num_samples = len(samples)
        for i in range(num_samples):
            #next state
            if i < num_samples - 2:
                samples[i].append(samples[i + 2][0])
            else:
                samples[i].append(None)

            if i%2 == 0:
                samples[i][2] = samples[i][2] + xreward*(i+1)/num_samples
            else:
                samples[i][2] = samples[i][2] - xreward*(i+1)/num_samples
            memory.add_sample(samples[i])

        sample_batch = memory.sample_samples(batch_size)
        actual_batch_size = len(sample_batch)
        state_batch = np.zeros((actual_batch_size, 9))
        next_state_batch = np.zeros((actual_batch_size, 9))
        action_batch = np.array([sample[1] for sample in sample_batch])

        for i, sample in enumerate(sample_batch):
            state_batch[i] = sample[0]
            if sample[3] is not None:
                next_state_batch[i] = sample[3]

        qsa_batch = critic.predict_batch(state_batch, sess)
        policy_batch = actor.get_policy_batch(state_batch, sess)
        advantage_batch = np.array([0 for i in range(actual_batch_size)])

        next_qsa_batch = critic.predict_batch(next_state_batch, sess)
        next_policy_batch = actor.get_policy_batch(next_state_batch, sess)
        #next_value_batch = critic.predict_value_batch(next_state_batch, sess)

        #fix up the policy so that invalid actions have a probability of zero
        for i in range(actual_batch_size):
            policy_batch[i] = fix_policy(state_batch[i, :], policy_batch[i, :])
            next_policy_batch[i] = fix_policy(next_state_batch[i, :], next_policy_batch[i, :])

        for i in range(actual_batch_size):
            #dot product is used here to compute expected value
            next_value = np.dot(next_qsa_batch[i], next_policy_batch[i])
            #next_value = next_value_batch[i]

            if sample_batch[i][3] is not None:
                qsa_batch[i, action_batch[i]] = sample_batch[i][2] + gamma*next_value
            else:
                qsa_batch[i, action_batch[i]] = sample_batch[i][2]

            advantage_batch[i] = qsa_batch[i, action_batch[i]]

        critic.train_batch(state_batch, qsa_batch, sess)
        actor.train_batch(state_batch, action_batch, advantage_batch, sess)
    critic.save(sess, 'tic_tac_toe_critic_nobad_a2c')
    actor.save(sess, 'tic_tac_toe_actor_nobad_a2c')
    critic.plot_losses('a2c_critic_losses')
    actor.plot_scores('a2c_scores')