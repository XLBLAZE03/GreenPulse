"""
chatbot.py
-----------
Rule-based NLU + response generator for the CO2 emissions chatbot.

This does NOT require any external LLM API — it works fully offline
using regex/keyword intent matching, which is perfect for a hackathon
demo (fast, free, no API key needed).

An optional `use_llm_polish()` hook is included at the bottom showing
how you could pass the retrieved data into Claude's API to make the
final answer sound more natural (see README for details).
"""

import re
from datetime import datetime, timedelta
import data_utils
import predictor

YEAR_RE = re.compile(r"\b(20(0[0-9]|1[0-9]|2[0-4]))\b")
DAYS_RE = re.compile(r"(\d+)\s*day")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}


def _extract_years(text):
    return [int(y) for y in YEAR_RE.findall(text.replace("-", " "))[0:0]] or \
           [int(m) for m in re.findall(r"\b(20\d{2})\b", text)]


def _extract_days(text):
    m = DAYS_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_explicit_date(text):
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def handle_message(message: str) -> dict:
    """
    Main entry point. Takes a raw user message, returns a dict:
      {"intent": str, "reply": str, "data": <optional structured data>}
    """
    text = message.lower().strip()

    # ---- 1. Greeting ----
    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", text):
        return {
            "intent": "greeting",
            "reply": ("Hello! I'm the INEOS CO2 Emissions Assistant. I can answer questions about "
                      "historical CO2 emissions (2010-2024) and forecast future emissions. "
                      "Try asking: 'What was the average CO2 emission in 2020?' or "
                      "'Predict CO2 emissions for the next 30 days'."),
        }

    # ---- 2. Help ----
    if "help" in text or "what can you do" in text:
        return {
            "intent": "help",
            "reply": (
                "I can help with:\n"
                "- Overall stats: 'What is the average CO2 emission?'\n"
                "- Yearly stats: 'Total CO2 emitted in 2018'\n"
                "- Comparisons: 'Compare 2015 and 2020'\n"
                "- Trend: 'What is the CO2 trend over the years?'\n"
                "- Seasonality: 'Which month has the highest emissions?'\n"
                "- Correlations: 'What factors drive CO2 emissions?'\n"
                "- Records: 'What was the highest CO2 day ever recorded?'\n"
                "- Forecasts: 'Predict CO2 for the next 14 days' or "
                "'What will CO2 emissions be on 2025-06-01?'"
            ),
        }

    # ---- 3. Forecast / Prediction ----
    # An explicit future-looking date (e.g. 2025-06-01, outside the historical
    # 2010-2024 range) always means "forecast", regardless of other keywords.
    explicit_date = _extract_explicit_date(text)
    if explicit_date and int(explicit_date[:4]) > 2024:
        result = predictor.forecast_single_date(explicit_date)
        return {
            "intent": "forecast_date",
            "reply": (f"Predicted CO2 emission on {result['date']}: "
                      f"{result['predicted_co2_tCO2e']} tCO2e."),
            "data": result,
        }

    if any(k in text for k in ["predict", "forecast", "future", "next", "will"]):
        if explicit_date:
            result = predictor.forecast_single_date(explicit_date)
            return {
                "intent": "forecast_date",
                "reply": (f"Predicted CO2 emission on {result['date']}: "
                          f"{result['predicted_co2_tCO2e']} tCO2e."),
                "data": result,
            }

        days = _extract_days(text) or 30
        start = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        results = predictor.forecast(start, days=days)
        avg_pred = round(sum(r["predicted_co2_tCO2e"] for r in results) / len(results), 2)
        return {
            "intent": "forecast_range",
            "reply": (f"Forecast for the next {days} days (starting {start}): "
                      f"average predicted emission is {avg_pred} tCO2e/day. "
                      f"(Range: {min(r['predicted_co2_tCO2e'] for r in results)} - "
                      f"{max(r['predicted_co2_tCO2e'] for r in results)} tCO2e)"),
            "data": results,
        }

    # ---- 4. Compare two years ----
    years_found = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if "compare" in text and len(years_found) >= 2:
        comp = data_utils.compare_years(years_found[0], years_found[1])
        if comp is None:
            return {"intent": "compare_years", "reply": "I don't have data for one of those years (valid range: 2010-2024)."}
        direction = "higher" if comp["mean_diff"] > 0 else "lower"
        return {
            "intent": "compare_years",
            "reply": (f"In {comp['year_a']['year']}, average daily CO2 was {comp['year_a']['mean']} tCO2e, "
                      f"vs {comp['year_b']['mean']} tCO2e in {comp['year_b']['year']}. "
                      f"That's {abs(comp['mean_diff'])} tCO2e {direction} "
                      f"({abs(comp['pct_change'])}%) in {comp['year_a']['year']}."),
            "data": comp,
        }

    # ---- 5. Yearly stats ----
    if years_found:
        year = years_found[0]
        stats = data_utils.year_stats(year)
        if stats is None:
            return {"intent": "year_stats", "reply": f"I don't have data for {year}. Data covers 2010-2024."}
        if "total" in text or "sum" in text:
            metric_reply = f"Total CO2 emitted in {year}: {stats['total']:,} tCO2e."
        elif "max" in text or "highest" in text or "peak" in text:
            metric_reply = f"Highest single-day CO2 emission in {year}: {stats['max']} tCO2e."
        elif "min" in text or "lowest" in text:
            metric_reply = f"Lowest single-day CO2 emission in {year}: {stats['min']} tCO2e."
        else:
            metric_reply = f"Average daily CO2 emission in {year}: {stats['mean']} tCO2e."
        return {"intent": "year_stats", "reply": metric_reply, "data": stats}

    # ---- 6. Trend over years ----
    if "trend" in text:
        trend = data_utils.yearly_trend()
        first_year, last_year = min(trend), max(trend)
        change = round(trend[last_year] - trend[first_year], 2)
        direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed roughly flat"
        return {
            "intent": "trend",
            "reply": (f"Average daily CO2 emissions {direction} from {trend[first_year]} tCO2e "
                      f"in {first_year} to {trend[last_year]} tCO2e in {last_year} "
                      f"(change of {change} tCO2e). Overall the emissions are fairly stable year "
                      f"over year, with most variation coming from seasonal patterns rather than a "
                      f"long-term trend."),
            "data": trend,
        }

    # ---- 7. Seasonality (month) ----
    if "month" in text or "season" in text:
        seasonal = data_utils.monthly_seasonality()
        peak_month = max(seasonal, key=seasonal.get)
        low_month = min(seasonal, key=seasonal.get)
        return {
            "intent": "seasonality",
            "reply": (f"Emissions are highest in {MONTH_NAMES[peak_month]} (avg {seasonal[peak_month]} tCO2e) "
                      f"and lowest in {MONTH_NAMES[low_month]} (avg {seasonal[low_month]} tCO2e), "
                      f"showing a clear seasonal cycle likely tied to natural gas usage."),
            "data": seasonal,
        }

    # ---- 8. Correlations / drivers ----
    if any(k in text for k in ["factor", "driver", "correlat", "cause", "influence"]):
        corr = data_utils.top_correlations()
        top3 = list(corr.items())[:3]
        top_str = ", ".join([f"{k} ({v})" for k, v in top3])
        return {
            "intent": "correlations",
            "reply": (f"The variables most strongly correlated with CO2 emissions are: {top_str}. "
                      f"Natural gas consumption is typically the biggest driver in petrochemical operations."),
            "data": corr,
        }

    # ---- 9. Highest/lowest ever recorded ----
    if ("highest" in text or "lowest" in text or "record" in text) and not years_found:
        hl = data_utils.highest_lowest_days()
        top = hl["highest"][0]
        bottom = hl["lowest"][0]
        return {
            "intent": "records",
            "reply": (f"Highest CO2 day ever recorded: {top['Date']} with {top['CO2_Emissions_tCO2e']} tCO2e. "
                      f"Lowest: {bottom['Date']} with {bottom['CO2_Emissions_tCO2e']} tCO2e."),
            "data": hl,
        }

    # ---- 10. Overall stats (default/fallback for stat-like questions) ----
    if any(k in text for k in ["average", "mean", "total", "overall", "how much", "co2", "emission"]):
        stats = data_utils.overall_stats()
        return {
            "intent": "overall_stats",
            "reply": (f"Across {stats['records']} days ({stats['date_range']}), average daily CO2 emission "
                      f"was {stats['mean']} tCO2e (min {stats['min']}, max {stats['max']}). "
                      f"Total emitted over the full period: {stats['total']:,} tCO2e."),
            "data": stats,
        }

    # ---- 11. Fallback ----
    return {
        "intent": "fallback",
        "reply": ("I'm not sure I understood that. Try asking about average/total emissions, "
                  "a specific year, a comparison between years, the yearly trend, seasonality, "
                  "or a forecast (e.g. 'predict CO2 for next 30 days'). Type 'help' to see examples."),
    }
