
from pathlib import Path
import joblib
import pandas as pd


# File paths

BASE_DIR = Path(__file__).resolve().parent

MATCH_MODEL_PATH = BASE_DIR / "model_artifacts" / "match_winner_pipeline.joblib"
PLAYER_MODEL_PATH = BASE_DIR / "model_artifacts" / "top_player_pipeline.joblib"

MATCH_FEATURES_PATH = BASE_DIR / "afl_match_features_v1.csv"
PLAYER_FEATURES_PATH = BASE_DIR / "afl_player_features_v1.csv"


# Load trained models

match_winner_pipeline = joblib.load(MATCH_MODEL_PATH)
top_player_pipeline = joblib.load(PLAYER_MODEL_PATH)


# Load feature data

match_features = pd.read_csv(MATCH_FEATURES_PATH)
player_features = pd.read_csv(PLAYER_FEATURES_PATH)

match_features["match_date"] = pd.to_datetime(
    match_features["match_date"]
)

player_features["match_date"] = pd.to_datetime(
    player_features["match_date"]
)


# Match model feature columns

MATCH_FEATURE_COLUMNS = [
    "round",
    "home_team",
    "away_team",
    "venue",
    "home_recent_5_win_rate",
    "home_win_streak",
    "home_recent_5_avg_score",
    "home_days_rest",
    "away_recent_5_win_rate",
    "away_win_streak",
    "away_recent_5_avg_score",
    "away_days_rest",
    "h2h_matches",
    "h2h_current_home_wins",
    "h2h_current_away_wins",
    "h2h_draws",
    "h2h_current_home_win_rate",
    "home_pre_match_ladder_rank",
    "home_pre_match_points",
    "home_pre_match_percentage",
    "away_pre_match_ladder_rank",
    "away_pre_match_points",
    "away_pre_match_percentage"
]


# Predict match winner

def predict_match_winner(team_a, team_b, date):

    # Validate team names
    if not isinstance(team_a, str) or not team_a.strip():
        raise ValueError("team_a must be a non-empty string.")

    if not isinstance(team_b, str) or not team_b.strip():
        raise ValueError("team_b must be a non-empty string.")

    # Teams must be different
    if team_a == team_b:
        raise ValueError("team_a and team_b must be different teams.")

    # Validate date
    try:
        match_date = pd.to_datetime(date)
    except Exception:
        raise ValueError(
            "Invalid date. Please provide a valid date such as '2025-08-08'."
        )

    # Known teams
    known_teams = set(
        match_features["home_team"].dropna().unique()
    ) | set(
        match_features["away_team"].dropna().unique()
    )

    if team_a not in known_teams:
        raise ValueError(
            f"Unknown team name: '{team_a}'. "
            "Please use a team name from the dataset."
        )

    if team_b not in known_teams:
        raise ValueError(
            f"Unknown team name: '{team_b}'. "
            "Please use a team name from the dataset."
        )

    # Available date range
    min_date = match_features["match_date"].min()
    max_date = match_features["match_date"].max()

    if match_date < min_date or match_date > max_date:
        raise ValueError(
            f"Date '{match_date.date()}' is outside the available "
            f"data range ({min_date.date()} to {max_date.date()})."
        )

    # Find exact match
    match_row = match_features[
        (match_features["match_date"] == match_date) &
        (match_features["home_team"] == team_a) &
        (match_features["away_team"] == team_b)
    ].copy()

    if match_row.empty:
        raise ValueError(
            f"No match found for {team_a} vs {team_b} "
            f"on {match_date.date()}."
        )

    # Prepare model input
    X = match_row[MATCH_FEATURE_COLUMNS]

    # Prediction
    prediction = match_winner_pipeline.predict(X)[0]
    probabilities = match_winner_pipeline.predict_proba(X)[0]

    class_names = match_winner_pipeline.classes_

    probability_map = {
        class_name: float(probability)
        for class_name, probability
        in zip(class_names, probabilities)
    }

    return {
        "home_team": team_a,
        "away_team": team_b,
        "date": str(match_date.date()),
        "winner": prediction,
        "probability": round(
            probability_map[prediction],
            4
        ),
        "class_probabilities": {
            key: round(value, 4)
            for key, value in probability_map.items()
        }
    }


# Predict top players

def predict_top_player(match_date, team, top_k=5):

    # Validate team
    if not isinstance(team, str) or not team.strip():
        raise ValueError(
            "Team name must be a non-empty string."
        )

    # Validate top_k
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    # Validate team exists
    known_teams = set(
        player_features["team"].dropna().unique()
    )

    if team not in known_teams:
        raise ValueError(
            f"Unknown team name: '{team}'. "
            "Please use a team name from the dataset."
        )

    # Validate date
    try:
        match_date = pd.to_datetime(match_date)
    except Exception:
        raise ValueError(
            "Invalid date. Please provide a valid date such as '2025-08-08'."
        )

    # Available date range
    min_date = player_features["match_date"].min()
    max_date = player_features["match_date"].max()

    if match_date < min_date or match_date > max_date:
        raise ValueError(
            f"Date '{match_date.date()}' is outside the available "
            f"data range ({min_date.date()} to {max_date.date()})."
        )

    # Get players for requested team and date
    match_players = player_features[
        (player_features["match_date"] == match_date) &
        (player_features["team"] == team)
    ].copy()

    if match_players.empty:
        raise ValueError(
            f"No player data found for {team} "
            f"on {match_date.date()}."
        )

    # Check for usable features
    if match_players[
        "player_recent_5_avg_disposals"
    ].isna().all():

        raise ValueError(
            f"No usable player features found for {team} "
            f"on {match_date.date()}."
        )

    # Model input
    X_players = match_players[
        ["player_recent_5_avg_disposals"]
    ]

    # Predict
    match_players["predicted_disposals"] = (
        top_player_pipeline.predict(X_players)
    )

    # Rank
    ranked_players = (
        match_players
        .sort_values(
            "predicted_disposals",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # Build result
    results = []

    for rank, (_, row) in enumerate(
        ranked_players.iterrows(),
        start=1
    ):

        results.append({
            "rank": rank,
            "player_id": int(row["player_id"]),
            "predicted_disposals": round(
                float(row["predicted_disposals"]),
                2
            )
        })

    return results
