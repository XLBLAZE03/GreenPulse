"""
app.py
-------
Flask backend for the CO2 Emissions Chatbot + Equipment Emissions Bot project.

Endpoints:
  GET  /                              -> serves frontend/index.html

  GET  /api/health                    -> health check

  POST /api/chat                      -> chatbot reply
                                          Handles:
                                          - normal CO2 dataset questions
                                          - equipment-wise CO2/energy/root-cause questions

  GET  /api/stats                     -> overall dataset stats
  GET  /api/stats/<year>              -> stats for a specific year
  GET  /api/forecast?days=30          -> forecast next N days from today
  GET  /api/forecast?date=YYYY-MM-DD  -> forecast for a specific date
  GET  /api/trend                     -> yearly average trend
  GET  /api/seasonality               -> monthly average seasonality

  GET  /api/equipment/summary         -> equipment-wise CO2 + energy summary
  GET  /api/equipment/sections        -> section-wise emissions summary
  POST /api/equipment/upload          -> upload new INEOS Excel file dynamically
  GET  /api/equipment/report          -> download PDF root-cause report
"""

import os

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

import data_utils
import predictor
import chatbot
import equipment_bot


app = Flask(__name__)
CORS(app)


# Absolute path to frontend folder
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)


# -------------------------------------------------
# Frontend Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def serve_home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_frontend_files(path):
    """
    Serves files from the frontend folder.
    If the requested file does not exist, fallback to index.html.
    """
    requested_file = os.path.join(FRONTEND_DIR, path)

    if os.path.exists(requested_file):
        return send_from_directory(FRONTEND_DIR, path)

    return send_from_directory(FRONTEND_DIR, "index.html")


# -------------------------------------------------
# Health Route
# -------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------------------------------------
# Combined Chatbot Route
# -------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    message = body.get("message", "")

    if not message.strip():
        return jsonify({"error": "message field is required"}), 400

    # First try equipment-wise bot
    # Handles prompts like:
    # - equipment-wise CO2 generation
    # - energy consumption
    # - section-wise emissions
    # - root cause
    # - generate report
    try:
        equipment_result = equipment_bot.answer_equipment_prompt(message)

        if equipment_result.get("handled"):
            return jsonify({
                "reply": equipment_result.get("reply", ""),
                "intent": equipment_result.get("intent", "equipment_query")
            })

    except Exception as e:
        # Do not break original chatbot if equipment bot fails
        return jsonify({
            "reply": f"Equipment bot error: {str(e)}",
            "intent": "equipment_error"
        }), 500

    # Fallback to original CO2 emissions chatbot
    result = chatbot.handle_message(message)
    return jsonify(result)


# -------------------------------------------------
# Original CO2 Dataset API Routes
# -------------------------------------------------
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

    return jsonify({
        "start_date": start,
        "days": days_param,
        "forecast": result
    })


@app.route("/api/trend", methods=["GET"])
def trend():
    return jsonify(data_utils.yearly_trend())


@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    return jsonify(data_utils.monthly_seasonality())


# -------------------------------------------------
# Equipment Emissions API Routes
# -------------------------------------------------
@app.route("/api/equipment/summary", methods=["GET"])
def equipment_summary_route():
    """
    Returns equipment-wise CO2 generation and energy consumption.

    Example response includes:
    - Boiler emissions + energy
    - Furnace emissions + energy
    - Reformer emissions + energy
    - Other equipment emissions + energy
    - Root cause equipment
    """
    try:
        return jsonify(equipment_bot.equipment_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/equipment/sections", methods=["GET"])
def equipment_sections_route():
    """
    Returns section-wise emissions summary.

    Example sections:
    - Utilities / Steam Generation
    - Thermal Processing
    - Process Conversion
    - Auxiliary Equipment
    """
    try:
        return jsonify(equipment_bot.section_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/equipment/upload", methods=["POST"])
def equipment_upload_route():
    """
    Upload a new INEOS Excel file dynamically.
    The uploaded file becomes the active background dataset for:
    - equipment summary
    - section summary
    - root cause
    - report generation
    - equipment chatbot responses
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Uploaded file has no filename"}), 400

    if not file.filename.lower().endswith((".xlsx", ".xls")):
        return jsonify({"error": "Please upload an Excel file with .xlsx or .xls extension"}), 400

    try:
        result = equipment_bot.set_uploaded_dataset(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/equipment/report", methods=["GET"])
def equipment_report_route():
    """
    Generates and downloads PDF report containing:
    - equipment-wise CO2 generation
    - equipment-wise energy consumption
    - section-wise emissions
    - root-cause analysis showing highest emitting equipment
    """
    try:
        pdf_buffer = equipment_bot.generate_pdf_report()

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="equipment_emissions_root_cause_report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# App Runner
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)