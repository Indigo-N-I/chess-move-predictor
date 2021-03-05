import berserk
from datetime import datetime
import berserk.utils

def get_games(user: str, start: datetime, end:datetime, games: int = 100):
    client = berserk.Client()

    start = berserk.utils.to_millis(start)
    end = berserk.utils.to_millis(end)

    return client.games.export_by_player(user, since=start, until=end, max = games, as_pgn = True)

def split_bw(games, user):
    '''
    splits games into games played as white and games played as white
    returns (whitegames, blackgames)
    '''
    white_games = []
    black_games = []
    random_games = 0

    for game in games:
        if game.find(user, game.index('Black')) > 0:
            black_games.append(game)
        elif game.find(user) > 0:
            white_games.append(game)
        else:
            random_games += 1

    if random_games > 0:
        print(f"Found {random_games} number of games not played by {user}")
    return white_games, black_games

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 20

    retreaved_games = get_games('whoisis', start, end, games)
    # print(type(retreaved_games), type(1))
    white, black = split_bw(retreaved_games, 'whoisis')
    print(white)
    print(black)
