from board_class import Board
import numpy as np
import tensorflow as tf

def state_from_board(board, counter):
    state = np.array(board.board)
    if counter == 1:
        state = -state
    state = state.reshape((1,9))
    return state

#restore trained network
sess = tf.InteractiveSession()
meta_graph = tf.train.import_meta_graph('trained_models/tic_tac_toe_actor_nobad_a2c.meta')
meta_graph.restore(sess, tf.train.latest_checkpoint('./trained_models/'))
graph = tf.get_default_graph()
states = graph.get_tensor_by_name('states:0')

#checks if a string can be cast to an int
def is_int(s):
    try:
        int(s)
        return True
    except:
        return False

#returns an index based on current board configuration
def ai_pick(board, counter):
    state = state_from_board(board, counter)
    policy = sess.run('policy:0', feed_dict={states: state})[0]
    #here we make sure invalid moves aren't made
    for i in range(9):
        if state[0, i] != 0:
            policy[i] = 0
    probsum = np.sum(policy)
    policy /= probsum
    return np.random.choice(9, p=policy)

board = Board()

player_symbol = input('Pick X or O (X goes first): ')

while player_symbol != 'X' and player_symbol != 'O':
    player_symbol = input('Invalid symbol. Pick X or O: ')

if player_symbol == 'X':
    player_num = 0
else:
    player_num = 1

winner = ''
counter = 0
symbols = ['X', 'O']
while winner == '':
    board.printBoard()
    if counter == player_num:
        index = input('Choose an index for a square: ')
        while not is_int(index) or int(index) < 0 or int(index) > 8 or board.getSquare(int(index)) != 0:
            index = input('Your entry was invalid. Choose again: ')
        index = int(index)
    else:
        index = ai_pick(board, counter)
        print(symbols[counter] + ' chooses index ' + str(index))
        print()
    winner = board.setSquare(index, symbols[counter])
    counter = (counter + 1)%2

board.printBoard()
if winner == 'D':
    print('Draw!')
else:
    print(winner + ' won!')