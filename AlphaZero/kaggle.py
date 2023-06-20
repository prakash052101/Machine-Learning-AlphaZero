import numpy as np

from nodeNetNeu import *
from tictactoe import *
from conectFour import *
from mcts import *
from mctsParallel import *


class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args['search']:
            self.mcts = MCTS(self.game, self.args, self.model)

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        state[state==2] = -1

        state = self.game.change_perspective(state, player)

        if self.args['search']:
            policy = self.mcts.search(state)

        else:
            policy, _ = self.model.predict(state, augment=self.args['augument'])

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)    

        if self.args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy) 

        return action  