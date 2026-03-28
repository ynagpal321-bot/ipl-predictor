 # 🚀 IPL AI Predictor ULTRA PRO (FINAL FIXED)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import requests

# 🔴 ADD YOUR API KEY HERE
API_KEY = "d80d4fc2-4579-436c-8b6b-2d1d16213234"

# ================= LOAD DATA =================
data = pd.read_csv("ipl_matches.csv")

# ================= PITCH =================
pitch_map = {
    "Chennai": "spin", "Mumbai": "batting", "Kolkata": "spin",
    "Bengaluru": "batting", "Jaipur": "balanced", "Ahmedabad": "balanced",
    "Hyderabad": "bowling", "Lucknow": "spin", "Delhi": "batting",
    "Mohali": "batting", "Dharamsala": "fast", "Guwahati": "balanced", "Raipur": "balanced"
}

def pitch_score(p):
    return {"batting": 0.2, "bowling": -0.2, "spin": -0.1, "fast": -0.1}.get(p, 0)

# ================= PLAYER =================
players = {
    "MI": {"bat": 0.8, "bowl": 0.75},
    "CSK": {"bat": 0.78, "bowl": 0.80},
    "RCB": {"bat": 0.85, "bowl": 0.65},
    "KKR": {"bat": 0.75, "bowl": 0.78},
    "SRH": {"bat": 0.70, "bowl": 0.82},
    "RR": {"bat": 0.77, "bowl": 0.74},
    "GT": {"bat": 0.79, "bowl": 0.81},
    "LSG": {"bat": 0.74, "bowl": 0.76},
    "DC": {"bat": 0.72, "bowl": 0.73},
    "PBKS": {"bat": 0.76, "bowl": 0.70}
}

def team_strength(t):
    p = players.get(t, {"bat": 0.7, "bowl": 0.7})
    return (p["bat"] + p["bowl"]) / 2

# ================= MATCHUPS =================
def matchup_adv(t1, t2):
    return (team_strength(t1) - team_strength(t2)) * 0.3

# ================= HOME =================
home_map = {
    "CSK": "Chennai","MI": "Mumbai","KKR": "Kolkata","RCB": "Bengaluru",
    "RR": "Jaipur","GT": "Ahmedabad","SRH": "Hyderabad","LSG": "Lucknow",
    "DC": "Delhi","PBKS": "Mohali"
}

def home_adv(team, venue):
    return 0.1 if home_map.get(team) == venue else 0

# ================= TEAM STATS =================
def calc_stats(df):
    teams = list(set(df['team1']).union(set(df['team2'])))
    stats = {}
    for t in teams:
        m = df[(df['team1']==t)|(df['team2']==t)]
        w = m[m['winner']==t].shape[0]
        stats[t] = {"win_rate": w/len(m) if len(m)>0 else 0.5}
    return stats

team_stats = calc_stats(data)

# ================= FEATURE ENGINEERING =================
def prepare_features(df, ref_columns=None):
    df = df.copy()

    df['team1_strength'] = df['team1'].map(lambda x: team_stats.get(x, {"win_rate":0.5})['win_rate'])
    df['team2_strength'] = df['team2'].map(lambda x: team_stats.get(x, {"win_rate":0.5})['win_rate'])
    df['pitch_score'] = df['venue'].map(lambda x: pitch_score(pitch_map.get(x,"balanced")))

    X = df[[
        "team1","team2","toss_winner","venue",
        "team1_strength","team2_strength","pitch_score"
    ]]

    X = pd.get_dummies(X)

    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)

    return X

# ================= TRAIN MODEL =================
X = prepare_features(data)
y = data["winner"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

feature_columns = X.columns

teams = sorted(team_stats.keys())
venues = sorted(data['venue'].unique())

app = FastAPI()

# ================= HOME =================
@app.get("/", response_class=HTMLResponse)
def home():
    t_opts = "".join([f"<option>{t}</option>" for t in teams])
    v_opts = "".join([f"<option>{v}</option>" for v in venues])
    return f"""
    <html><body style='text-align:center;'>
    <h1>🏏 IPL ULTRA AI</h1>

    <form action="/predict">
    Team1:<select name="team1">{t_opts}</select><br><br>
    Team2:<select name="team2">{t_opts}</select><br><br>
    Toss:<select name="toss_winner">{t_opts}</select><br><br>
    Venue:<select name="venue">{v_opts}</select><br><br>
    <button>Predict</button>
    </form>

    <hr>

    <form action="/live">
    <input name="score" placeholder="Score"><br><br>
    <input name="wickets" placeholder="Wickets"><br><br>
    <input name="overs" placeholder="Overs"><br><br>
    <input name="target" placeholder="Target"><br><br>
    <button>Live</button>
    </form>

    <br><br>
    <a href="/live-auto">🔴 Live API</a>
    <br><br>
    <a href="/retrain">♻️ Retrain Model</a>
    </body></html>
    """

# ================= PREDICT =================
@app.get("/predict", response_class=HTMLResponse)
def predict(team1:str, team2:str, toss_winner:str, venue:str):

    df = pd.DataFrame([{
        "team1":team1,
        "team2":team2,
        "toss_winner":toss_winner,
        "venue":venue
    }])

    X_new = prepare_features(df, feature_columns)

    base = max(model.predict_proba(X_new)[0])

    prob = base
    prob += matchup_adv(team1,team2)
    prob += home_adv(team1,venue)

    pitch = pitch_map.get(venue,"balanced")
    prob += pitch_score(pitch)

    prob = max(0,min(1,prob))

    return f"<h2>{team1} vs {team2}</h2><h3>Winner: {model.predict(X_new)[0]}</h3><h3>{round(prob*100,2)}%</h3><a href='/'>Back</a>"

# ================= LIVE =================
@app.get("/live", response_class=HTMLResponse)
def live(score:int,wickets:int,overs:float,target:int):

    balls = int((20-overs)*6)
    need = target-score

    rr = need/(balls/6) if balls>0 else 12
    prob = 1-(rr/12)
    prob -= wickets*0.02

    prob = max(0,min(1,prob))

    return f"<h2>Live: {score}/{wickets}</h2><h3>{round(prob*100,2)}%</h3><a href='/'>Back</a>"

# ================= LIVE API =================
@app.get("/live-auto", response_class=HTMLResponse)
def live_auto():
    try:
        url=f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}"
        res=requests.get(url).json()
        m=res['data'][0]

        t1,t2=m['teams']
        s=m['score'][0]

        score=s['r']
        overs=s['o']
        wickets=s['w']
        target=score+20

    except:
        t1,t2="TeamA","TeamB"
        score,overs,wickets,target=120,15,3,180

    balls=int((20-overs)*6)
    need=target-score
    rr=need/(balls/6) if balls>0 else 12
    prob=max(0,min(1,1-rr/12))

    return f"<h2>{t1} vs {t2}</h2><h3>{score}/{wickets}</h3><h3>{round(prob*100,2)}%</h3><a href='/'>Back</a>"

# ================= RETRAIN =================
@app.get("/retrain")
def retrain():
    global model, feature_columns, team_stats

    new = pd.read_csv("ipl_matches.csv")

    # update stats
    team_stats = calc_stats(new)

    X_new = prepare_features(new)
    y_new = new["winner"]

    model.fit(X_new, y_new)
    feature_columns = X_new.columns

    return {"status": "retrained successfully"}

# ================= RUN =================
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
