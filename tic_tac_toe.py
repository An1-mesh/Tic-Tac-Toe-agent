# Animesh, 2001CS07

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy as dc

class TicTacToe:
    def __init__(self): # Initialise
        self.board = np.zeros((3, 3))
        self.players = ['X', 'O']
        self.winner = None
        self.curr_player = None
        self.curr_id = None
        self.game_end = False
    
    def display_board(self): # Print the board
        print("-------------")
        for i in range(len(self.board)):
            print("|", end=' ')
            for j in range(len(self.board)):
                print(self.players[int(self.board[i][j] - 1)] if self.board[i][j] != 0 else " ", end=' | ')
            print()
            print("-------------")
    
    def swap_turn(self): # Swapping players turn
        self.curr_id = 1 - self.curr_id
        self.curr_player = self.players[self.curr_id]
    
    def valid_moves(self): # Check for list fo valid moves
        pos = []
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] == 0:
                    pos.append((x, y))
        return pos
    
    def move_it(self, pos): # Execute a move
        moved = False
        if self.board[pos[0]][pos[1]] != 0:
            moved = False
            return moved
        self.board[pos[0]][pos[1]] = self.players.index(self.curr_player) + 1
        self.check_winner()
        self.swap_turn()
        moved = True
        return moved
    
    # Check if there is a winner  
    def check_winner(self):         
        # Checking rows and columns
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.players[int(self.board[i][0] - 1)]
                self.game_end = True
                return
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                self.winner = self.players[int(self.board[0][i] - 1)]
                self.game_end = True
                return
        # Checking diagonals
        if self.board[2][2] == self.board[1][1] == self.board[0][0] != 0:
            self.winner = self.players[int(self.board[0][0] - 1)]
            self.game_end = True
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.players[int(self.board[0][2] - 1)]
            self.game_end = True
            return
        # Match ends in a draw
        if self.valid_moves() == []:
            self.game_end = True
            return

class QLearningAgent: # Q-Learning agent class
    def __init__(self, alpha, epsilon, discount_factor): # Initialize
        self.Q = {}
        self.alpha = alpha * 1.0
        self.epsilon = epsilon * 1.0
        self.discount_factor = discount_factor * 1.0

    def get_hashkey(self, action, state): # Get hashkey to insert into the Q-table
        hash = ''
        r = 0
        while r < 3:
            c = 0
            while c < 3:
                hash += str(state[r][c])
                c += 1
            r += 1
        hash += str(action[0])
        hash += str(action[1])
        return hash

    def get_Q_value(self, state, action): # Fetch Q-values
        hash = self.get_hashkey(action, state)
        if hash not in self.Q:
            self.Q[hash] = 0.00
        return self.Q[hash]

    def choose_action(self, state, valid_moves, is_training=True): # Choose best next action
        if random.uniform(0, 1) < self.epsilon and is_training:
            return random.choice(valid_moves)
        else:
            Q_values = [self.get_Q_value(state, action) for action in valid_moves]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(valid_moves)) if Q_values[i] == max_Q]
                i = 0
                i = random.choice(best_moves)
            else:
                i = 0
                i = Q_values.index(max_Q)
            return valid_moves[i]
            if max_Q < 0.00:
                return 0

    def update_Q_value(self, state, action, reward, next_state, game): # Update values in teh Q-table
        next_Q_values = [self.get_Q_value(next_state, next_action) for next_action in game.valid_moves()]
        max_next_Q = 0.00
        if not (state == next_state).all(): # No next state, i.e. match ends with the last move
            max_next_Q = max(next_Q_values) if next_Q_values else 0.00
        hash = self.get_hashkey(action, state)
        if hash not in self.Q:
            self.Q[hash] = 0.00
        self.Q[hash] += self.alpha * (reward + self.discount_factor * max_next_Q - self.Q[hash])

def plot_graph(total_list, win_list, loss_list, draw_list, plt_title, num_wins, num_draws, num_loss, num_games): # Plot the graph
    plt.ylabel('Game outcomes in %')
    plt.xlabel('Number of games')
    plt.title(plt_title)

    plt.plot(total_list, win_list, 'g-', label='Wins')
    plt.plot(total_list, loss_list, 'r-', label='Losses')
    plt.plot(total_list, draw_list, 'b-', label='Ties')
    plt.legend()
    plt.show()

    print("Win percentage: {:.2f}%".format(num_wins * 100 / num_games))
    print("Tie percentage: {:.2f}%".format(num_draws * 100 / num_games))
    print("Loss percentage: {:.2f}%".format(num_loss * 100 / num_games))
    print()

def train(p_id, num_episodes, alpha, epsilon, discount_factor): # Training function
    print("Training...")
    win_list = []
    draw_list = []
    loss_list = []
    total_list = []
    num_wins = 0
    num_draws = 0
    num_loss = 0

    agent = QLearningAgent(alpha, epsilon, discount_factor)
    for i in tqdm(range(num_episodes)):
        game = TicTacToe()
        game.curr_id = 0
        game.curr_player = game.players[game.curr_id]
        state = dc(game.board)
        while game.game_end != True:
            reward = -0.2
            if game.curr_player == game.players[p_id]:
                action = agent.choose_action(state, game.valid_moves())
                game.move_it(action)
            else:
                game.move_it(random.choice(game.valid_moves()))
                next_state = dc(game.board)
                
                if i >= 1 and not game.game_end:
                    agent.update_Q_value(state, action, reward, next_state, game)
                    state = next_state
            
            if game.game_end:
                if game.winner == game.players[p_id]:
                    reward = 1.0
                    num_wins += 1
                elif game.winner == game.players[1 - p_id]:
                    reward = -1.0
                    num_loss += 1
                else:
                    num_draws += 1
                agent.update_Q_value(state, action, reward, state, game)
                break
    
        win_list.append(num_wins * 100 / (i+1))
        loss_list.append(num_loss * 100 / (i+1))
        draw_list.append(num_draws * 100 / (i+1))
        total_list.append(i + 1)

    plot_graph(total_list, win_list, loss_list, draw_list, "Training against randomised player", num_wins, num_draws, num_loss, (i+1))
    return agent

def eval(p_id, agent, num_games): # Evaluate the trained network
    print("\nTesting...")
    win_list = []
    draw_list = []
    loss_list = []
    total_list = []
    num_wins = 0
    num_draws = 0
    num_loss = 0
    
    for i in tqdm(range(num_games)):
        game = TicTacToe()
        game.curr_id = 0
        game.curr_player = game.players[game.curr_id]
        state = game.board
        while not game.game_end:
            if game.curr_player == game.players[p_id]:
                action = agent.choose_action(state, game.valid_moves(), False) # False for exploration
            else:
                action = random.choice(game.valid_moves())
            game.move_it(action)
            reward = -0.2
            state = game.board
            if game.game_end:
                if game.winner == game.players[p_id]:
                    reward = 1.0
                elif game.winner == game.players[1-p_id]:
                    reward = -1.0
        if reward == 1:
            num_wins += 1
        elif game.winner is None:
            num_draws += 1
        else:
            num_loss += 1

        win_list.append(num_wins * 100 / (i+1))
        loss_list.append(num_loss * 100 / (i+1))
        draw_list.append(num_draws * 100 / (i+1))
        total_list.append(i + 1)

    plot_graph(total_list, win_list, loss_list, draw_list, "Testing against randomised player", num_wins, num_draws, num_loss, num_games)

def play_with_agent(agent, p_id):
    game = TicTacToe()
    game.curr_id = 0
    game.curr_player = game.players[game.curr_id]
    game.display_board()

    while game.game_end != True:
        if game.curr_player == game.players[1-p_id]:
            move = input(f"{game.curr_player}'s turn. Enter the row and column (ex- 0 0): ")
            move = tuple(map(int, move.split()))
            
            while move not in game.valid_moves():
                move = input("Invalid move. Try again: ")
                move = tuple(map(int, move.split()))
        else:
            move = agent.choose_action(game.board, game.valid_moves())
        game.move_it(move)
        game.display_board()

    if game.winner is None:
        print("It's a draw!")
    else:
        print(f"{game.winner} wins!")

def main():
    # Picking sides
    side = 'X'
    p_id = 1
    while True:
        side = input("Please choose a side: O or X\n")
        if side == 'X' or side == 'x' or side =='o' or side =='O':
            break
        print("Invalid side. Choose either O or X.")
    if side =='O' or side == 'o':
        p_id = 0
    else:
        p_id = 1

    # Training the Q-learning agent
    agent = train(p_id, num_episodes=50000, alpha=0.1, epsilon=0.2, discount_factor=0.99)

    # Evaluating the Q-learning agent
    eval(p_id, agent, num_games=2000)

    # Playing against the Q-learning agent
    play_with_agent(agent, p_id)

if __name__ == "__main__":
    main()