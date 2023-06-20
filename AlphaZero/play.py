from conectFour import *
from tictactoe import *
from kaggle import *

import kaggle_environments
import matplotlib.pyplot as plt

game = ConnectFour()
tictactoe = TicTacToe()

args = {
    'C' : 2,
    'num_searches' : 600,
    'dirichlet_epsilon' : 0.1,
    'dirichlet_alpha' : 0.3,
    'search': True,
    'temperature':0,

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("model_0_ConnectFour.pt", map_location=device))
model.eval()

# model = ResNet(game, 4, 64, device)
# model.load_state_dict(torch.load("model_2.pt", map_location=device))
# model.eval()

env = kaggle_environments.make("connectx")

player1 = KaggleAgent(model, game, args)
player2 = KaggleAgent(model, game, args)

players = [player1.run, player2.run]

env.run(players)

env.render(mode='')

# state = tictactoe.get_initial_state()
# state = tictactoe.get_next_state(state, 2, -1)
# state = tictactoe.get_next_state(state, 4, -1)
# state = tictactoe.get_next_state(state, 6, 1)
# state = tictactoe.get_next_state(state, 8, 1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# encoded_state = tictactoe.get_encoded_state(state)

# tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)


# model = ResNet(tictactoe, 4, 64, device)
# model.load_state_dict(torch.load('model_2.pt', map_location=device))
# model.eval()

# policy, value = model(tensor_state)

# value = value.item()
# policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

# print(value)
# print(state)
# print(tensor_state)

# plt.bar(range(tictactoe.action_size), policy)
# plt.show()