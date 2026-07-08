"""
app.py
-------
Flask backend for the CO2 Emissions Chatbot hackathon project.

Endpoints:
  GET  /api/health              -> health check
  POST /api/chat                -> {"message": "..."}  -> chatbot reply
  GET  /api/stats               -> overall dataset stats
  GET  /api/stats/<year>        -> stats for a specific year
  GET  /api/forecast?days=30    -> forecast next N days from today
  GET  /api/forecast?date=YYYY-MM-DD -> forecast for a specific date
  GET  /api/trend                -> yearly average trend
  GET  /api/seasonality           -> monthly average seasonality

Run:
    pip install -r ../requirements.txt
    python app.py
Then open frontend/index.html in your browser (or serve it, see README).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

import data_utils
import predictor
import chatbot

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin/port) to call this API


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    message = body.get("message", "")
    if not message.strip():
        return jsonify({"error": "message field is required"}), 400
    result = chatbot.handle_message(message)
    return jsonify(result)


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(data_utils.overall_stats())


@app.route("/api/stats/<int:year>", methods=["GET"])
def stats_year(year):
    result = data_utils.year_stats(year)
    if result is None:
        return jsonify({"error": f"No data for year {year}"}), 404
    return jsonify(result)


@app.route("/api/forecast", methods=["GET"])
def forecast():
    date_param = request.args.get("date")
    days_param = request.args.get("days", default=30, type=int)

    if date_param:
        result = predictor.forecast_single_date(date_param)
        return jsonify(result)

    from datetime import datetime, timedelta
    start = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = predictor.forecast(start, days=days_param)
    return jsonify({"start_date": start, "days": days_param, "forecast": result})


@app.route("/api/trend", methods=["GET"])
def trend():
    return jsonify(data_utils.yearly_trend())


@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    return jsonify(data_utils.monthly_seasonality())


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable if you demo from another device on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
