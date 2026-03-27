# 🚀 IPL AI Predictor FINAL PRO VERSION

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import random

# ================= LOAD DATA =================
data = pd.read_csv("ipl_matches.csv")

# ================= PITCH CONDITIONS =================
pitch_map = {
    "Chennai": "spin",
    "Mumbai": "batting",
    "Kolkata": "spin",
    "Bengaluru": "batting",
    "Jaipur": "balanced",
    "Ahmedabad": "balanced",
    "Hyderabad": "bowling",
    "Lucknow": "spin",
    "Delhi": "batting",
    "Mohali": "batting",
    "Dharamsala": "fast",
    "Guwahati": "balanced",
    "Raipur": "balanced"
}

def pitch_score(pitch):
    if pitch == "batting":
        return 0.2
    elif pitch == "bowling":
        return -0.2
    elif pitch == "spin":
        return -0.1
    elif pitch == "fast":
        return -0.1
    else:
        return 0

# ================= TEAM STATS =================
def calculate_team_stats(df):
    teams = list(set(df['team1']).union(set(df['team2'])))
    stats = {}

    for team in teams:
        matches = df[(df['team1'] == team) | (df['team2'] == team)]
        wins = matches[matches['winner'] == team].shape[0]
        total = matches.shape[0]

        win_rate = wins / total if total > 0 else 0.5

        stats[team] = {"win_rate": win_rate}

    return stats

team_stats = calculate_team_stats(data)

# ================= ADD FEATURES =================
data['team1_strength'] = data['team1'].map(lambda x: team_stats[x]['win_rate'])
data['team2_strength'] = data['team2'].map(lambda x: team_stats[x]['win_rate'])

data['pitch'] = data['venue'].map(lambda x: pitch_map.get(x, "balanced"))
data['pitch_score'] = data['pitch'].map(pitch_score)

# ================= MODEL =================
features = data[[
    "team1", "team2", "toss_winner", "venue",
    "team1_strength", "team2_strength",
    "pitch_score"
]]

target = data["winner"]

features = pd.get_dummies(features)

model = RandomForestClassifier(n_estimators=200)
model.fit(features, target)

teams = sorted(team_stats.keys())
venues = sorted(data['venue'].unique())

app = FastAPI()

# ================= HOME =================
@app.get("/", response_class=HTMLResponse)
def home():
    team_options = "".join([f"<option>{t}</option>" for t in teams])
    venue_options = "".join([f"<option>{v}</option>" for v in venues])

    return f"""
    <html>
    <body style="text-align:center;">
        <h1>🏏 IPL AI Predictor PRO MAX</h1>

        <form action="/predict">
            Team1:<select name="team1">{team_options}</select><br><br>
            Team2:<select name="team2">{team_options}</select><br><br>
            Toss Winner:<select name="toss_winner">{team_options}</select><br><br>
            Venue:<select name="venue">{venue_options}</select><br><br>
            <button type="submit">Predict</button>
        </form>

        <hr>

        <form action="/live">
            <input name="score" placeholder="Score"><br><br>
            <input name="wickets" placeholder="Wickets"><br><br>
            <input name="overs" placeholder="Overs"><br><br>
            <input name="target" placeholder="Target"><br><br>
            <button type="submit">Live Predict</button>
        </form>

        <br><br>
        <a href="/live-auto">Auto Live</a>
    </body>
    </html>
    """

# ================= PREDICT =================
@app.get("/predict", response_class=HTMLResponse)
def predict(team1: str, team2: str, toss_winner: str, venue: str):

    t1 = team_stats[team1]['win_rate']
    t2 = team_stats[team2]['win_rate']

    pitch = pitch_map.get(venue, "balanced")
    p_score = pitch_score(pitch)

    df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "venue": venue,
        "team1_strength": t1,
        "team2_strength": t2,
        "pitch_score": p_score
    }])

    df = pd.get_dummies(df)
    df = df.reindex(columns=features.columns, fill_value=0)

    pred = model.predict(df)[0]
    prob = max(model.predict_proba(df)[0])

    prob = max(0, min(1, prob + p_score))

    return f"""
    <html>
    <body style='text-align:center;'>
        <h2>🏆 Winner: {pred}</h2>
        <h3>Win Probability: {round(prob*100,2)}%</h3>
        <h3>Pitch: {pitch}</h3>
        <a href='/'>Back</a>
    </body>
    </html>
    """

# ================= LIVE =================
@app.get("/live", response_class=HTMLResponse)
def live(score: int, wickets: int, overs: float, target: int):

    balls_left = int((20 - overs) * 6)
    runs_needed = target - score

    if balls_left <= 0:
        prob = 0
    else:
        required_rr = runs_needed / (balls_left / 6)
        prob = max(0, min(1, 1 - (required_rr / 12)))

    return f"<h2>Win Probability: {round(prob*100,2)}%</h2><a href='/'>Back</a>"

# ================= AUTO LIVE =================
@app.get("/live-auto", response_class=HTMLResponse)
def live_auto():
    score = random.randint(100, 180)
    target = 180
    overs = random.randint(10, 19)
    wickets = random.randint(2, 8)

    balls_left = int((20 - overs) * 6)
    runs_needed = target - score

    if balls_left <= 0:
        prob = 0
    else:
        required_rr = runs_needed / (balls_left / 6)
        prob = max(0, min(1, 1 - (required_rr / 12)))

    return f"<h2>Auto Match: {score}/{wickets} ({overs} ov)</h2><h3>{round(prob*100,2)}%</h3><a href='/'>Back</a>"

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
