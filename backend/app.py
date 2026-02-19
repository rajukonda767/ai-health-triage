from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
import warnings
from google import genai

# ===============================
# Flask Setup
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# Load ML Model
# ===============================
model = joblib.load("rf_model.pkl")


client = genai.Client(
    api_key="AIzaSyCWxuogL0F_5Tx38oTpBtvzQCj6_6Jx7Js",
    http_options={
        "api_version": "v1"
    }
)


# ===============================
# High Risk Rule Engine
# ===============================
def high_risk_rule(data):
    if data["chest_pain"] == 1:
        return True
    if data["breathlessness"] == 1:
        return True
    if data["heart_disease"] == 1 and (data["fever"] == 1 or data["cough"] == 1):
        return True
    if data["diabetes"] == 1 and data["fever"] == 1 and data["days"] > 2:
        return True
    if data["age"] > 75 and data["fever"] == 1:
        return True

    return False


# ===============================
# Gemini Recommendation Generator
# ===============================
def generate_llm_recommendation(data, risk):

    prompt = f"""
You are a medical triage assistant.

Risk Level: {risk}

Symptoms:
Fever: {data['fever']}
Cough: {data['cough']}
Breathlessness: {data['breathlessness']}
Chest Pain: {data['chest_pain']}
Duration: {data['days']} days

Medical History:
Asthma: {data['asthma']}
Diabetes: {data['diabetes']}
Heart Disease: {data['heart_disease']}

Instructions:
- Give exactly 3 short home-care recommendations.
- Each line must start with a dash (-).
- Do NOT number the recommendations.
- Do NOT write any headings.
- Do NOT mention medicine names.
- Do NOT give diagnosis.

After the 3 lines, write ONE single line exactly in this format:
When to see a doctor: <short warning sentence>

Do not add any extra text.
Keep total under 100 words.

"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print("LLM Error:", e)
        return (
            "Rest well and stay hydrated.\n"
            "Monitor your symptoms closely.\n"
            "Avoid physical strain.\n\n"
            "**If symptoms worsen, consult a doctor immediately.**"
        )




# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return "Backend Running"


@app.route("/health")
def health():
    return {"status": "running"}, 200


@app.route("/triage", methods=["POST"])
def triage():
    try:
        data = request.json

        # High Risk Rule Override
        if high_risk_rule(data):
            return jsonify({
                "risk": "High",
                "confidence": 99,
                "recommendation_text":
                    "- Visit the nearest emergency hospital immediately.\n"
                    "- Do not delay seeking medical care.\n"
                    "- Avoid self-medication.\n"
                    "When to see a doctor: Seek emergency medical attention immediately.",
                "note": "This system provides decision support only and is not a medical diagnosis."
            })


        # ML Prediction
        input_features = np.array([[
            data["age"],
            data["fever"],
            data["cough"],
            data["breathlessness"],
            data["chest_pain"],
            data["days"],
            data["asthma"],
            data["diabetes"],
            data["heart_disease"]
        ]])

        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        risk = prediction

        # Gemini only for Low / Medium
        if risk in ["Low", "Medium"]:
            recommendation_text = generate_llm_recommendation(data, risk)
        else:
            recommendation_text = "Monitor symptoms carefully."

        return jsonify({
            "risk": risk,
            "confidence": confidence,
            "recommendation_text": recommendation_text,
            "note": "This system provides decision support only and is not a medical diagnosis."
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)