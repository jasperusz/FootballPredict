import pandas as pd
import numpy as np

# Read CSV variables
competitions = pd.read_csv('data/competitions.csv')
clubs_infos = pd.read_csv('data/clubs.csv')
club_games = pd.read_csv('data/club_games.csv')
games = pd.read_csv('data/games.csv')
game_events = pd.read_csv('data/game_events.csv')
appearances = pd.read_csv('data/appearances.csv')
players_infos = pd.read_csv('data/players.csv')
player_valuations = pd.read_csv('data/player_valuations.csv')
merged_games_data = pd.merge(games, club_games)

# Games Variables
season_played = games['season']
club_position = games['home_club_position']
opponent_position = games['away_club_position']
club_goals_history = games['home_club_goals']
opponent_goals_history = games['away_club_goals']

# Club Games Variables
own_goals = club_games['own_goals']
opponent_goals = club_games['opponent_goals']

conditions = [
    (own_goals > opponent_goals),
    (own_goals < opponent_goals),
    (own_goals == opponent_goals)
]

choices = ['win', 'loss', 'draw']

game_results = np.select(conditions, choices, default='unknown')
merged_games_data['result'] = game_results

