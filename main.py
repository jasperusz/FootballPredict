import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Machine Learning Variables
X = merged_games_data[['club_id', 'opponent_id', 'hosting', 'home_club_position', 'away_club_position', 'round']].dropna()
y = merged_games_data['result']
y = y.loc[X.index]

# Encode categorical variables
le_hosting = LabelEncoder()
le_round = LabelEncoder()
le_result = LabelEncoder()

X['hosting'] = le_hosting.fit_transform(X['hosting'])
X['round'] = le_round.fit_transform(X['round'])
y = le_result.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
    )

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators = 200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))