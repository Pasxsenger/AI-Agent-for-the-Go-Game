import random
import time
from copy import deepcopy


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
        piece_type = int(lines[0])
        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])
    with open(path, 'w') as f:
        f.write(res)


class HostofGO:
    def __init__(self, piece_type, previous_board, board, n=5):
        self.size = n
        self.komi = n / 2
        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
        self.died_pieces = []

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        return deepcopy(self)

    def update_board(self, new_board):
        self.board = new_board

    def detect_neighbor(self, location, board=None):
        if board is None:
            board = self.board
        neighbors = []
        i = location[0]
        j = location[1]

        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))

        return neighbors

    def detect_neighbor_ally(self, location, board=None):
        if board is None:
            board = self.board
        group_allies = []
        neighbors = self.detect_neighbor(location=location, board=board)
        for piece in neighbors:
            if board[piece[0]][piece[1]] == board[location[0]][location[1]]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, location, board=None):
        ally_members = []
        stack = [location]
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(location=piece, board=board)
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, location, board=None):
        if board is None:
            board = self.board
        ally_members = self.ally_dfs(location=location, board=board)
        for member in ally_members:
            neighbors = self.detect_neighbor(location=member)
            for piece in neighbors:
                if board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def get_liberties(self, location, board=None):
        if board is None:
            board = self.board
        liberties = []
        ally_members = self.ally_dfs(location=location, board=board)
        for ally in ally_members:
            neighbors = self.detect_neighbor(location=ally)
            for piece in neighbors:
                if board[piece[0]][piece[1]] == 0:
                    liberties.append(piece)
        return liberties

    def find_died_pieces(self, piece_type, board=None):
        if board is None:
            board = self.board
        died_pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == piece_type and not self.find_liberty(location=(i, j), board=board):
                    died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type=piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def valid_place_check(self, location, piece_type, board=None):
        if board is None:
            board = self.board

        if not (0 <= location[0] < len(board)):
            return False
        if not (0 <= location[1] < len(board)):
            return False

        if board[location[0]][location[1]] != 0:
            return False

        test_go = self.copy_board()
        test_board = test_go.board

        test_board[location[0]][location[1]] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(location):
            return True

        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(location):
            return False
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def score(self, piece_type):
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt


class MiniMaxGo:
    def __init__(self, piece_type, previous_board, board, size, go_host):
        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
        self.size = size
        self.go_host = go_host
        self.level = 4
        self.opp_level = 1

    def get_piece_locations(self, board, piece_type):
        locations = []
        for i in range(self.go_host.size):
            for j in range(self.go_host.size):
                if board[i][j] == piece_type:
                    locations.append((i, j))
        return locations

    def eval_func(self, board, piece_type):
        black_score = 0
        white_score = self.go_host.komi
        black_endangered_liberty = 0
        white_endangered_liberty = 0
        dead_pieces_black = len(self.go_host.find_died_pieces(piece_type=1, board=board))
        dead_pieces_white = len(self.go_host.find_died_pieces(piece_type=2, board=board))
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 1:
                    black_score = black_score + 1
                    liberties = self.go_host.get_liberties(location=(i, j), board=board)
                    if len(liberties) < 2:
                        black_endangered_liberty += 1
                elif board[i][j] == 2:
                    white_score = white_score + 1
                    liberties = self.go_host.get_liberties(location=(i, j), board=board)
                    if len(liberties) < 2:
                        white_endangered_liberty = white_endangered_liberty + 1

        if piece_type == 1:
            evalu = (white_endangered_liberty - black_endangered_liberty) + (black_score - white_score) + (dead_pieces_white * 10 - dead_pieces_black * 16)
        else:
            evalu = (black_endangered_liberty - white_endangered_liberty) + (white_score - black_score) + (dead_pieces_black * 10 - dead_pieces_white * 16)
        return evalu

    def get_legal_moves(self, piece_type, board=None):
        legal_moves = []
        for i in range(self.go_host.size):
            for j in range(self.go_host.size):
                if self.go_host.valid_place_check(location=(i, j), piece_type=piece_type, board=board):
                    legal_moves.append((i, j))
        random.shuffle(legal_moves)
        return legal_moves

    def get_opp_legal_moves(self, opp_go_host, piece_type):
        legal_moves = []
        for i in range(opp_go_host.size):
            for j in range(opp_go_host.size):
                if opp_go_host.valid_place_check(location=(i, j), piece_type=piece_type, board=None):
                    legal_moves.append((i, j))
        random.shuffle(legal_moves)
        return legal_moves

    def min_node(self, piece_type, level, alpha, beta, start, board):
        end = time.time()
        new_board = deepcopy(board)
        moves = self.get_legal_moves(piece_type=piece_type, board=None)
        beta_min = float('inf')
        if len(moves) == 0 or level == 0 or end - start > 8.5:
            return (-1, -1), self.eval_func(board=new_board, piece_type=piece_type)
        else:
            for move in moves:
                board_to_pass = deepcopy(board)
                if move in self.get_legal_moves(piece_type=piece_type, board=None):
                    self.go_host.previous_board = deepcopy(board_to_pass)
                    board_to_pass[move[0]][move[1]] = piece_type
                    self.go_host.board = board_to_pass
                new_board = board_to_pass
                self.go_host.remove_died_pieces(piece_type=3 - piece_type)
                if piece_type == 1:
                    next_piece_type = 2
                elif piece_type == 2:
                    next_piece_type = 1
                new_move, new_score = self.max_node(piece_type=next_piece_type, level=level - 1, alpha=alpha,
                                                    beta=beta, start=start, board=new_board)

                if new_score < beta_min:
                    beta_min = new_score
                    best_move = move
                beta = min(new_score, beta)
                if beta <= alpha:
                    break
            return best_move, beta_min

    def max_node(self, piece_type, level, alpha, beta, start, board):
        end = time.time()
        new_board = deepcopy(board)
        moves = self.get_legal_moves(piece_type=piece_type, board=None)
        pieces_to_remove = []
        alpha_max = float('-inf')
        for move in moves:
            self.go_host.board[move[0]][move[1]] = piece_type
            opponent_moves = self.get_legal_moves(piece_type=3 - piece_type, board=None)
            for opp_move in opponent_moves:
                self.go_host.board[opp_move[0]][opp_move[1]] = 3 - piece_type
                dead_pieces = self.go_host.find_died_pieces(piece_type=piece_type)
                self.go_host.board[opp_move[0]][opp_move[1]] = 0
                if (move in dead_pieces) and (move not in pieces_to_remove):
                    pieces_to_remove.append(move)
            self.go_host.board[move[0]][move[1]] = 0

        for piece in pieces_to_remove:
            if piece in moves:
                moves.remove(piece)

        if len(moves) == 0 or end - start > 8.5 or level == 0:
            return (-1, -1), self.eval_func(board=new_board, piece_type=piece_type)

        else:
            for move in moves:
                board_to_pass = deepcopy(board)
                if move in self.get_legal_moves(piece_type=piece_type, board=None):
                    self.go_host.previous_board = deepcopy(board_to_pass)
                    board_to_pass[move[0]][move[1]] = piece_type
                    self.go_host.board = board_to_pass
                new_board = board_to_pass

                self.go_host.remove_died_pieces(piece_type=3 - piece_type)
                if piece_type == 1:
                    next_piece_type = 2
                elif piece_type == 2:
                    next_piece_type = 1
                new_move, new_score = self.min_node(piece_type=next_piece_type, level=level - 1, alpha=alpha,
                                                    beta=beta, start=start, board=new_board)

                if alpha_max < new_score:
                    alpha_max = new_score
                    best_move = move
                alpha = max(new_score, alpha)
                if beta <= alpha:
                    break
            return best_move, alpha_max

    def opp_min_node(self, opp_go_host, piece_type, level, alpha, beta, start, board):
        new_board = deepcopy(board)
        beta_min = float('inf')
        moves = self.get_opp_legal_moves(opp_go_host=opp_go_host, piece_type=piece_type)
        end = time.time()

        if len(moves) == 0 or level == 0 or end - start > 8.5:
            return (-1, -1), self.eval_func(board=new_board, piece_type=piece_type)

        else:
            for move in moves:
                board_to_pass = deepcopy(board)
                if move in self.get_opp_legal_moves(opp_go_host=opp_go_host, piece_type=piece_type):
                    opp_go_host.previous_board = deepcopy(board_to_pass)
                    board_to_pass[move[0]][move[1]] = piece_type
                    opp_go_host.board = board_to_pass
                new_board = board_to_pass

                opp_go_host.remove_died_pieces(piece_type=3 - piece_type)
                if piece_type == 1:
                    next_piece_type = 2
                elif piece_type == 2:
                    next_piece_type = 1
                new_move, new_score = self.opp_max_node(opp_go_host=opp_go_host, piece_type=next_piece_type,
                                                        level=level - 1,
                                                        alpha=alpha, beta=beta, start=start, board=new_board)
                if beta_min > new_score:
                    beta_min = new_score
                    best_move = move
                beta = min(new_score, beta)
                if alpha >= beta:
                    break
            return best_move, beta_min

    def opp_max_node(self, opp_go_host, piece_type, level, alpha, beta, start, board):
        end = time.time()
        new_board = deepcopy(board)
        alpha_max = float('-inf')
        moves = self.get_opp_legal_moves(opp_go_host=opp_go_host, piece_type=piece_type)

        if len(moves) == 0 or end - start > 8.5 or level == 0:
            return (-1, -1), self.eval_func(board=new_board, piece_type=piece_type)

        else:
            for move in moves:
                board_to_pass = deepcopy(board)
                if move in self.get_opp_legal_moves(opp_go_host=opp_go_host, piece_type=piece_type):
                    opp_go_host.previous_board = deepcopy(board_to_pass)
                    board_to_pass[move[0]][move[1]] = piece_type
                    opp_go_host.board = board_to_pass
                new_board = board_to_pass

                opp_go_host.remove_died_pieces(piece_type=3 - piece_type)
                if piece_type == 1:
                    next_piece_type = 2
                elif piece_type == 2:
                    next_piece_type = 1
                new_move, new_score = self.opp_min_node(opp_go_host=opp_go_host, piece_type=next_piece_type, level=level - 1,
                                                        alpha=alpha, beta=beta, start=start, board=new_board)
                if alpha_max < new_score:
                    best_move = move
                    alpha_max = new_score
                alpha = max(new_score, alpha)
                if alpha >= beta:
                    break
            return best_move, alpha_max

    def get_next_step(self):
        piece_type = self.piece_type
        free_spaces = []
        for i in range(self.go_host.size):
            for j in range(self.go_host.size):
                if self.go_host.board[i][j] == 0:
                    free_spaces.append((i, j))

        conquests = dict()
        for space in free_spaces:
            self.go_host.board[space[0]][space[1]] = piece_type
            dead_pieces = self.go_host.find_died_pieces(piece_type=3 - piece_type)
            self.go_host.board[space[0]][space[1]] = 0
            if len(dead_pieces) >= 1:
                conquests[space] = len(dead_pieces)

        sorted_conquests = sorted(conquests, key=conquests.get, reverse=True)

        for conquest in sorted_conquests:
            temp_board = deepcopy(self.go_host.board)
            temp_board[conquest[0]][conquest[1]] = piece_type
            dead_pieces = self.go_host.find_died_pieces(piece_type=3 - piece_type, board=temp_board)
            for dp in dead_pieces:
                temp_board[dp[0]][dp[1]] = 0
            if conquest is not None and temp_board != self.go_host.previous_board:
                return conquest

        moves = self.get_legal_moves(piece_type=piece_type, board=None)
        pieces_to_remove = []
        for move in moves:
            self.go_host.board[move[0]][move[1]] = piece_type
            opponent_moves = self.get_legal_moves(piece_type=3 - piece_type, board=self.go_host.board)
            for om in opponent_moves:
                self.go_host.board[om[0]][om[1]] = 3 - piece_type
                dead_pieces = self.go_host.find_died_pieces(piece_type=piece_type)
                self.go_host.board[om[0]][om[1]] = 0
                if move in dead_pieces:
                    pieces_to_remove.append(move)
            self.go_host.board[move[0]][move[1]] = 0

        for piece in pieces_to_remove:
            if piece in moves:
                moves.remove(piece)

        if len(moves) == 0:
            return 'PASS'

        save_moves = dict()
        opponent_moves = []
        for i in range(self.go_host.size):
            for j in range(self.go_host.size):
                if self.go_host.board[i][j] == 0:
                    opponent_moves.append((i, j))

        for om in opponent_moves:
            self.go_host.board[om[0]][om[1]] = 3 - piece_type
            player_dead_pieces = self.go_host.find_died_pieces(piece_type=piece_type)
            self.go_host.board[om[0]][om[1]] = 0
            if len(player_dead_pieces) >= 1:
                save_moves[om] = len(player_dead_pieces)

        sorted_save_moves = sorted(save_moves, key=save_moves.get, reverse=True)

        for sm in sorted_save_moves:
            if sm is not None and sm in moves:
                return sm

        opponent_locations = self.get_piece_locations(board=self.go_host.board, piece_type=3 - piece_type)

        empty_x_opponent = []
        neighbors_list = []
        for i in opponent_locations:
            for operation in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (0 <= i[0] + operation[0] < self.go_host.size) and (0 <= i[1] + operation[1] < self.go_host.size):
                    neighbors = [(i[0] + operation[0], i[1] + operation[1])]

            for neighbor in neighbors:
                neighbors_list.append(neighbor)

        for neighbor in neighbors_list:
            if self.board[neighbor[0]][neighbor[1]] == 0:
                empty_x_opponent.append(neighbor)

        for move in moves:
            temp_board = deepcopy(self.go_host.board)
            temp_board[move[0]][move[1]] = piece_type
            dead_pieces = self.go_host.find_died_pieces(piece_type=3 - piece_type, board=temp_board)
            for dp in dead_pieces:
                temp_board[dp[0]][dp[1]] = 0
            opponent_locations = self.get_piece_locations(board=temp_board, piece_type=3 - piece_type)
            empty_y_opponent = []
            neighbors_list = []
            for i in opponent_locations:
                for operation in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (0 <= i[0] + operation[0] < self.go_host.size) and (0 <= i[1] + operation[1] < self.go_host.size):
                        neighbors = [(i[0] + operation[0], i[1] + operation[1])]

                for neighbor in neighbors:
                    neighbors_list.append(neighbor)

            for neighbor in neighbors_list:
                if self.board[neighbor[0]][neighbor[1]] == 0:
                    empty_y_opponent.append(neighbor)

            if len(empty_x_opponent) - len(empty_y_opponent) >= 1:
                return move

        possible_moves = [(2, 2), (1, 1), (1, 3), (3, 1), (3, 3), (2, 0), (2, 4), (0, 2), (4, 2)]
        if len(moves) >= 15:
            for pm in possible_moves:
                if pm in moves:
                    return pm

        opponent_board = deepcopy(self.go_host.board)
        opponent_previous_board = deepcopy(self.go_host.previous_board)

        opp_go_host = HostofGO(piece_type=3 - piece_type, previous_board=opponent_previous_board, board=opponent_board, n=self.size)

        start = time.time()
        move, score = self.opp_max_node(opp_go_host=opp_go_host, piece_type=3 - piece_type, level=self.opp_level,
                                        alpha=float('-inf'), beta=float('inf'), start=start, board=opponent_board)
        x = move[0]
        y = move[1]
        self.go_host.board[x][y] = 3 - piece_type
        free_spaces = []
        for i in range(self.go_host.size):
            for j in range(self.go_host.size):
                if self.go_host.board[i][j] == 0:
                    free_spaces.append((i, j))

        conquests = dict()
        for space in free_spaces:
            self.go_host.board[space[0]][space[1]] = piece_type
            dead_pieces = self.go_host.find_died_pieces(piece_type=3 - piece_type)
            self.go_host.board[space[0]][space[1]] = 0
            if len(dead_pieces) >= 1:
                conquests[space] = len(dead_pieces)

        sorted_conquests = sorted(conquests, key=conquests.get, reverse=True)
        conquests_remove = []
        self.go_host.board[x][y] = 0
        if len(sorted_conquests) != 0:
            for i in sorted_conquests:
                self.go_host.board[i[0]][i[1]] = piece_type
                opp_moves = self.get_legal_moves(piece_type=3 - piece_type, board=self.go_host.board)
                for j in opp_moves:
                    self.go_host.board[j[0]][j[1]] = 3 - piece_type
                    dead_pieces = self.go_host.find_died_pieces(piece_type=3 - piece_type, board=self.go_host.board)
                    self.go_host.board[j[0]][j[1]] = 0
                    if i in dead_pieces:
                        conquests_remove.append(i)
                self.go_host.board[i[0]][i[1]] = 0

            for x in conquests_remove:
                if x in sorted_conquests:
                    sorted_conquests.remove(x)

            for i in sorted_conquests:
                if i in moves:
                    return i

        start = time.time()
        move, score = self.max_node(piece_type=piece_type, level=self.level,
                                    alpha=float('-inf'), beta=float('inf'), start=start, board=self.go_host.board)
        return move

    def play(self, output_path):
        start = time.time()
        next_step = self.get_next_step()
        if next_step is None:
            next_step = 'PASS'
        end = time.time()
        print("TIME: ", (end - start))
        print("OUTPUT: ", next_step)
        writeOutput(result=next_step, path=output_path)


def main():
    size = 5
    input_path = 'input.txt'
    output_path = 'output.txt'
    piece_type, previous_board, board = readInput(size, input_path)
    go_host = HostofGO(piece_type=piece_type, previous_board=previous_board, board=board, n=size)
    agent = MiniMaxGo(piece_type, previous_board, board, size, go_host)
    agent.play(output_path)


if __name__ == '__main__':
    main()
