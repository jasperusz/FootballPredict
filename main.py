import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from difflib import get_close_matches


# Read CSV variables

clubs_infos = pd.read_csv('data/clubs.csv')
club_games = pd.read_csv('data/club_games.csv')
games = pd.read_csv('data/games.csv')
players_infos = pd.read_csv('data/players.csv')
player_valuations = pd.read_csv('data/player_valuations.csv')
merged_games_data = pd.merge(games, club_games)

# Games Variables
season_played = games['season']
club_position = games['home_club_position']
opponent_position = games['away_club_position']
recent_club_position = games.sort_values('date').groupby('home_club_id')['home_club_position'].last()
recent_opponent_position = games.sort_values('date').groupby('away_club_id')['away_club_position'].last()
club_goals_history = games['home_club_goals']
opponent_goals_history = games['away_club_goals']


# Club info Variables
club_name_to_id = clubs_infos.set_index('name')['club_id'].to_dict()
club_value = player_valuations.groupby('current_club_id')['market_value_in_eur'].sum()
club_value = club_value.reset_index()
club_value.columns = ['club_id', 'club_market_value']
merged_games_data = merged_games_data.merge(club_value, on='club_id', how='left')
recent_club_position = recent_club_position.reset_index()
recent_club_position.columns = ['home_club_id', 'recent_club_position']
merged_games_data = merged_games_data.merge(recent_club_position, on='home_club_id', how='left')
club_win_rate = club_games[club_games['is_win'] == 1].groupby('club_id').size() / \
    club_games.groupby('club_id').size()
club_win_rate = club_win_rate.reset_index()
club_win_rate.columns = ['club_id', 'club_win_rate']
merged_games_data = merged_games_data.merge(club_win_rate, on='club_id', how='left')

# Oponnent info Variables
opponent_value = club_value.copy()
opponent_value.columns = ['opponent_id', 'opponent_market_value']
merged_games_data = merged_games_data.merge(opponent_value, on='opponent_id', how='left')
recent_opponent_position = recent_opponent_position.reset_index()
recent_opponent_position.columns = ['away_club_id', 'recent_opponent_position']
merged_games_data = merged_games_data.merge(recent_opponent_position, on='away_club_id', how='left')

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
merged_games_data = merged_games_data[merged_games_data['season'] >= 2023]

# Machine Learning Variables
X = merged_games_data[['club_id', 'opponent_id', 'hosting', 'club_market_value', \
                       'recent_club_position', 'recent_opponent_position', 'opponent_market_value', 'club_win_rate']]
y = merged_games_data['result']
y = y.loc[X.index]

# Encode categorical variables
le_hosting = LabelEncoder()
le_round = LabelEncoder()
le_result = LabelEncoder()

X['hosting'] = le_hosting.fit_transform(X['hosting'])
y = le_result.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
    )

# Train Random Forest Classifier
if os.path.exists('model.pkl'):
    model = joblib.load('model.pkl')
else:
    model = RandomForestClassifier(n_estimators = 200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = accuracy * 100

club_input = input('Enter the club name: ')
matches = [name for name in club_name_to_id.keys() if club_input.lower() in name.lower()]

if not matches:
    print("No matching club found.")
else:
    for i, name in enumerate(matches):
        print(f'{i + 1}. {name}')

    choice = int(input('Select the club number: '))
    club_name = matches[choice - 1]
    club_id = club_name_to_id[club_name]
    print(f'Selected club: {club_name} (ID: {club_id})')

opponent_input = input('Enter the opponent club name: ')
matches = [name for name in club_name_to_id.keys() if opponent_input.lower() in name.lower()]
if not matches:
    print("No matching club found.")
else:
    for i, name in enumerate(matches):
        print(f'{i + 1}. {name}')

    choice = int(input('Select the opponent club number: '))
    opponent_name = matches[choice - 1]
    opponent_id = club_name_to_id[opponent_name]
    print(f'Selected opponent club: {opponent_name} (ID: {opponent_id})')

def predict_match(club_id, opponent_id, hosting, recent_club_position, recent_opponent_position, club_win_rate):
    club_market_value = club_value.loc[club_value['club_id'] == club_id, 'club_market_value'].values[0]
    opponent_market_value = opponent_value.loc[opponent_value['opponent_id'] == opponent_id, 'opponent_market_value'].values[0]
    recent_club_position = recent_club_position.loc[recent_club_position['home_club_id']\
                                                     == club_id, 'recent_club_position'].values[0]
    recent_opponent_position = recent_opponent_position.loc[recent_opponent_position['away_club_id']\
                                                             == opponent_id, 'recent_opponent_position'].values[0]
    club_win_rate_value = club_win_rate.loc[club_win_rate['club_id'] == club_id, 'club_win_rate'].values[0]

    input_data = pd.DataFrame({
        'club_id': [club_id],
        'opponent_id': [opponent_id],
        'hosting': le_hosting.transform([hosting]),
        'club_market_value': [club_market_value],
        'recent_club_position': [recent_club_position],
        'recent_opponent_position': [recent_opponent_position],
        'opponent_market_value': [opponent_market_value],
        'club_win_rate': [club_win_rate_value]
    })

    prediction = model.predict(input_data)
    predicted_result = le_result.inverse_transform(prediction)[0]
    return predicted_result

hosting = input('Who is hosting the match? (home/away): ').strip().title()
predicted_result = predict_match(club_id, opponent_id, hosting, recent_club_position, recent_opponent_position, club_win_rate)
print(f'Predicted match result: home team {predicted_result}')

print(f'Model accuracy: {accuracy:.2f}')
