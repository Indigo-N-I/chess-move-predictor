from moves import get_moves, get_all_moves, get_win_loss
import pickle
from datetime import datetime

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2023, 3, 7)
    game_num = 500
    print("gathering games")

    games = get_win_loss('whoisis', start, end, game_num)

    with open("win_loss.p", 'wb') as f:
        pickle.dump(games, f)
