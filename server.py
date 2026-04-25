import flask
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = flask.Flask(__name__)

# ─────────────────────────────────────────────
# تحميل نماذج ML و RL
# ─────────────────────────────────────────────
print("⏳ Chargement des modèles...")

ml_model     = joblib.load('model.pkl')
ml_encoder   = joblib.load('label_encoder.pkl')
ml_features  = joblib.load('feature_cols.pkl')

with open("RL_model/q_table.pkl", "rb") as f:
    Q = pickle.load(f)
with open("RL_model/rl_config.pkl", "rb") as f:
    rl_config = pickle.load(f)

ACTIONS    = rl_config['actions']
N_DAYS     = rl_config['n_days']
N_WEEKEND  = rl_config['n_weekend']
N_POSITION = rl_config.get('n_position', 3)

print(f"✅ ML chargé | Features: {ml_features}")
print(f"✅ RL chargé | Actions: {ACTIONS}")

# ─────────────────────────────────────────────
# أسماء الأجهزة للعرض
# ─────────────────────────────────────────────
DEVICE_LABELS = {
    'none'            : '🌙 Aucun appareil',
    'coffee_machine'  : '☕ Machine à Café',
    'ac'              : '❄️ Climatisation',
    'tv'              : '📺 Télévision',
    'lights'          : '💡 Éclairage',
    'washing_machine' : '🧺 Machine à Laver',
}

DEVICE_ACTIONS = {
    'none'            : 'Aucune action requise',
    'coffee_machine'  : 'Allumer la machine à café',
    'ac'              : 'Allumer la climatisation',
    'tv'              : 'Allumer la télévision',
    'lights'          : 'Allumer les lumières',
    'washing_machine' : 'Lancer la machine à laver',
}

# ─────────────────────────────────────────────
# Helper RL
# ─────────────────────────────────────────────
def get_rl_state(hour, day, is_weekend, position):
    return (hour * (N_DAYS * N_WEEKEND * N_POSITION) +
            day  * (N_WEEKEND * N_POSITION) +
            is_weekend * N_POSITION +
            position)

# ─────────────────────────────────────────────
# Route: Test
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return flask.jsonify({
        "message"       : "✅ Server is running",
        "main_model"    : "Q-Learning RL ← النموذج الرئيسي",
        "compare_model" : "Random Forest ML ← للمقارنة فقط",
    })

# ─────────────────────────────────────────────
# Route: /predict — RL هو النموذج الرئيسي
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = flask.request.json

        hour        = int(data.get("hour", datetime.now().hour))
        position    = int(data.get("position", 2))
        day_of_week = int(data.get("day_of_week", 0))
        is_weekend  = int(data.get("is_weekend", 0))

        # ✅ RL — Q-Table (النموذج الرئيسي)
        state       = get_rl_state(hour, day_of_week, is_weekend, position)
        best_idx    = int(np.argmax(Q[state]))
        pred_device = ACTIONS[best_idx]
        confidence  = round(float(Q[state][best_idx]) * 10, 1)

        return flask.jsonify({
            "success"          : True,
            "predicted_device" : pred_device,
            "deviceLabel"      : DEVICE_LABELS.get(pred_device, pred_device),
            "action"           : DEVICE_ACTIONS.get(pred_device, ''),
            "confidence_pct"   : min(confidence, 99),
            "position"         : position,
            "hour"             : hour,
            "model"            : "Q-Learning RL",
        })

    except Exception as e:
        return flask.jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# Route: /predict_ml — ML للمقارنة فقط
# ─────────────────────────────────────────────
@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    try:
        data = flask.request.json

        hour        = int(data.get("hour", datetime.now().hour))
        position    = int(data.get("position", 2))
        consumption = float(data.get("consumption", 2.0))
        day_of_week = int(data.get("day_of_week", 0))
        month       = int(data.get("month", 6))
        is_weekend  = int(data.get("is_weekend", 0))

        input_dict = {}
        if 'hour'        in ml_features: input_dict['hour']        = hour
        if 'day_of_week' in ml_features: input_dict['day_of_week'] = day_of_week
        if 'month'       in ml_features: input_dict['month']       = month
        if 'is_weekend'  in ml_features: input_dict['is_weekend']  = is_weekend
        if 'consumption' in ml_features: input_dict['consumption'] = consumption
        if 'position'    in ml_features: input_dict['position']    = position

        input_df       = pd.DataFrame([input_dict])
        pred_encoded   = ml_model.predict(input_df)[0]
        pred_device    = ml_encoder.inverse_transform([pred_encoded])[0]
        probas         = ml_model.predict_proba(input_df)[0]
        confidence_pct = round(float(probas.max()) * 100, 1)

        return flask.jsonify({
            "success"          : True,
            "predicted_device" : pred_device,
            "deviceLabel"      : DEVICE_LABELS.get(pred_device, pred_device),
            "action"           : DEVICE_ACTIONS.get(pred_device, ''),
            "confidence_pct"   : confidence_pct,
            "position"         : position,
            "hour"             : hour,
            "model"            : "Random Forest ML",
        })

    except Exception as e:
        return flask.jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# Route: /predict_rl — نفس /predict (للتوافق)
# ─────────────────────────────────────────────
@app.route("/predict_rl", methods=["POST"])
def predict_rl():
    return predict()

# ─────────────────────────────────────────────
# تشغيل
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Server démarré sur http://localhost:5000")
    print("🧠 Modèle principal : Q-Learning RL")
    print("📊 Comparaison     : Random Forest ML (/predict_ml)")
    app.run(host="0.0.0.0", port=5000, debug=True)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)