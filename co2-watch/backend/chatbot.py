"""
chatbot.py
-----------
Rule-based NLU + response generator for the CO2 emissions chatbot.

Fixed version:
- Handles future month/year prompts like "Provide data for February month 2030"
- Handles historical month/year prompts like "Provide data for February 2020"
- Handles percentage increase after a forecast query
- Avoids returning the same generic "next 60 days" response for unrelated prompts
"""

import re
import calendar
from datetime import datetime, timedelta

import data_utils
import predictor


# -------------------------------------------------
# Regex patterns
# -------------------------------------------------
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
DAYS_RE = re.compile(r"\b(\d+)\s*(day|days)\b")


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


# Simple process-level memory for follow-up questions like:
# User: "Provide data for February month 2030"
# User: "How much percentage increase?"
LAST_CONTEXT = {}


# -------------------------------------------------
# Extraction helpers
# -------------------------------------------------
def _extract_years(text: str):
    return [int(y) for y in YEAR_RE.findall(text)]


def _extract_days(text: str):
    match = DAYS_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _extract_explicit_date(text: str):
    match = DATE_RE.search(text)
    return match.group(1) if match else None


def _extract_month(text: str):
    text = text.lower()

    for name, num in MONTH_ALIASES.items():
        if re.search(rf"\b{name}\b", text):
            return num

    return None


def _is_percentage_question(text: str):
    keywords = [
        "percentage",
        "percent",
        "%",
        "increase",
        "decrease",
        "change",
        "growth",
        "higher",
        "lower",
    ]
    return any(k in text for k in keywords)


def _is_forecast_question(text: str):
    forecast_keywords = [
        "predict",
        "forecast",
        "future",
        "next",
        "will",
        "expected",
        "estimate",
        "estimated",
        "projection",
        "projected",
    ]

    return any(k in text for k in forecast_keywords)


def _is_data_request(text: str):
    data_keywords = [
        "provide",
        "give",
        "show",
        "data",
        "emission",
        "emissions",
        "co2",
        "month",
        "monthly",
        "year",
        "average",
        "total",
    ]

    return any(k in text for k in data_keywords)


def _historical_max_year():
    df = data_utils.load_data()
    return int(df["Date"].dt.year.max())


def _historical_min_year():
    df = data_utils.load_data()
    return int(df["Date"].dt.year.min())


def _date_range_for_month(year: int, month: int):
    days = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{days:02d}"
    return start, end, days


def _percentage_change(new_value, old_value):
    if old_value is None or old_value == 0:
        return None

    return round(((new_value - old_value) / old_value) * 100, 2)


# -------------------------------------------------
# Historical stats helpers
# -------------------------------------------------
def _historical_month_stats(year: int, month: int):
    df = data_utils.load_data().copy()
    df["Date"] = df["Date"].dt.tz_localize(None) if hasattr(df["Date"].dt, "tz") else df["Date"]

    sub = df[
        (df["Date"].dt.year == year)
        & (df["Date"].dt.month == month)
    ]

    if sub.empty:
        return None

    col = "CO2_Emissions_tCO2e"

    return {
        "type": "historical_month",
        "year": year,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "days": int(len(sub)),
        "mean": round(float(sub[col].mean()), 2),
        "min": round(float(sub[col].min()), 2),
        "max": round(float(sub[col].max()), 2),
        "total": round(float(sub[col].sum()), 2),
    }


def _historical_month_average_across_years(month: int):
    df = data_utils.load_data().copy()
    sub = df[df["Date"].dt.month == month]

    if sub.empty:
        return None

    col = "CO2_Emissions_tCO2e"

    return {
        "month": month,
        "month_name": MONTH_NAMES[month],
        "mean": round(float(sub[col].mean()), 2),
        "min": round(float(sub[col].min()), 2),
        "max": round(float(sub[col].max()), 2),
        "total": round(float(sub[col].sum()), 2),
        "days": int(len(sub)),
    }


def _historical_year_average(year: int):
    stats = data_utils.year_stats(year)
    return stats


# -------------------------------------------------
# Forecast helpers
# -------------------------------------------------
def _forecast_month_stats(year: int, month: int):
    start, end, days = _date_range_for_month(year, month)
    results = predictor.forecast(start, days=days)

    values = [float(r["predicted_co2_tCO2e"]) for r in results]

    return {
        "type": "forecast_month",
        "year": year,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "start_date": start,
        "end_date": end,
        "days": days,
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "total": round(sum(values), 2),
        "daily_forecast": results,
    }


def _forecast_year_stats(year: int):
    start = f"{year}-01-01"
    days = 366 if calendar.isleap(year) else 365
    results = predictor.forecast(start, days=days)

    values = [float(r["predicted_co2_tCO2e"]) for r in results]

    return {
        "type": "forecast_year",
        "year": year,
        "start_date": start,
        "end_date": f"{year}-12-31",
        "days": days,
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "total": round(sum(values), 2),
        "daily_forecast": results,
    }


def _forecast_next_days(days: int):
    start = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    results = predictor.forecast(start, days=days)

    values = [float(r["predicted_co2_tCO2e"]) for r in results]

    return {
        "type": "forecast_range",
        "start_date": start,
        "days": days,
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "total": round(sum(values), 2),
        "daily_forecast": results,
    }


def _forecast_single_date(date_str: str):
    result = predictor.forecast_single_date(date_str)

    return {
        "type": "forecast_date",
        "date": result["date"],
        "predicted_co2_tCO2e": float(result["predicted_co2_tCO2e"]),
        "raw": result,
    }


# -------------------------------------------------
# Reply builders
# -------------------------------------------------
def _save_context(context: dict):
    global LAST_CONTEXT
    LAST_CONTEXT = context.copy()


def _reply_for_future_month(year: int, month: int):
    forecast_stats = _forecast_month_stats(year, month)
    baseline = _historical_month_average_across_years(month)

    pct_change = None
    direction = "different from"

    if baseline:
        pct_change = _percentage_change(forecast_stats["mean"], baseline["mean"])

        if pct_change is not None:
            if pct_change > 0:
                direction = "higher than"
            elif pct_change < 0:
                direction = "lower than"
            else:
                direction = "equal to"

    context = {
        "type": "forecast_month",
        "label": f"{forecast_stats['month_name']} {year}",
        "forecast_mean": forecast_stats["mean"],
        "forecast_total": forecast_stats["total"],
        "forecast_min": forecast_stats["min"],
        "forecast_max": forecast_stats["max"],
        "baseline_label": f"historical {forecast_stats['month_name']} average from {_historical_min_year()}-{_historical_max_year()}",
        "baseline_mean": baseline["mean"] if baseline else None,
        "pct_change": pct_change,
    }

    _save_context(context)

    if baseline and pct_change is not None:
        return {
            "intent": "forecast_month",
            "reply": (
                f"For {forecast_stats['month_name']} {year}, the forecasted average daily CO2 emission is "
                f"{forecast_stats['mean']} tCO2e/day.\n\n"
                f"Total predicted emission for the month: {forecast_stats['total']} tCO2e "
                f"over {forecast_stats['days']} days.\n"
                f"Predicted range: {forecast_stats['min']} - {forecast_stats['max']} tCO2e/day.\n\n"
                f"Compared with the historical {forecast_stats['month_name']} average "
                f"({baseline['mean']} tCO2e/day), this is {abs(pct_change)}% {direction} historical levels."
            ),
            "data": {
                "forecast": forecast_stats,
                "baseline": baseline,
                "pct_change": pct_change,
            },
        }

    return {
        "intent": "forecast_month",
        "reply": (
            f"For {forecast_stats['month_name']} {year}, the forecasted average daily CO2 emission is "
            f"{forecast_stats['mean']} tCO2e/day.\n"
            f"Total predicted emission for the month: {forecast_stats['total']} tCO2e "
            f"over {forecast_stats['days']} days.\n"
            f"Predicted range: {forecast_stats['min']} - {forecast_stats['max']} tCO2e/day."
        ),
        "data": forecast_stats,
    }


def _reply_for_historical_month(year: int, month: int):
    stats = _historical_month_stats(year, month)

    if stats is None:
        return {
            "intent": "historical_month_missing",
            "reply": (
                f"I do not have historical data for {MONTH_NAMES[month]} {year}. "
                f"The available historical dataset covers {_historical_min_year()}-{_historical_max_year()}."
            ),
        }

    _save_context({
        "type": "historical_month",
        "label": f"{MONTH_NAMES[month]} {year}",
        "mean": stats["mean"],
        "total": stats["total"],
    })

    return {
        "intent": "historical_month",
        "reply": (
            f"For {MONTH_NAMES[month]} {year}, average daily CO2 emission was "
            f"{stats['mean']} tCO2e/day.\n\n"
            f"Total CO2 emitted: {stats['total']} tCO2e over {stats['days']} days.\n"
            f"Recorded range: {stats['min']} - {stats['max']} tCO2e/day."
        ),
        "data": stats,
    }


def _reply_for_future_year(year: int):
    forecast_stats = _forecast_year_stats(year)
    overall = data_utils.overall_stats()

    pct_change = _percentage_change(forecast_stats["mean"], overall["mean"])

    direction = "higher than" if pct_change and pct_change > 0 else "lower than" if pct_change and pct_change < 0 else "equal to"

    _save_context({
        "type": "forecast_year",
        "label": str(year),
        "forecast_mean": forecast_stats["mean"],
        "forecast_total": forecast_stats["total"],
        "baseline_label": f"historical average from {_historical_min_year()}-{_historical_max_year()}",
        "baseline_mean": overall["mean"],
        "pct_change": pct_change,
    })

    return {
        "intent": "forecast_year",
        "reply": (
            f"For {year}, the forecasted average daily CO2 emission is "
            f"{forecast_stats['mean']} tCO2e/day.\n\n"
            f"Total predicted annual emission: {forecast_stats['total']} tCO2e.\n"
            f"Predicted range: {forecast_stats['min']} - {forecast_stats['max']} tCO2e/day.\n\n"
            f"Compared with the historical average ({overall['mean']} tCO2e/day), "
            f"this is {abs(pct_change)}% {direction} historical levels."
        ),
        "data": {
            "forecast": forecast_stats,
            "baseline": overall,
            "pct_change": pct_change,
        },
    }


def _reply_for_percentage_followup():
    if not LAST_CONTEXT:
        return {
            "intent": "percentage_followup_missing_context",
            "reply": (
                "Please specify what to compare. For example:\n"
                "- percentage increase for February 2030\n"
                "- compare February 2030 with historical February average\n"
                "- percentage increase from 2020 to 2030"
            ),
        }

    pct_change = LAST_CONTEXT.get("pct_change")

    if pct_change is None:
        return {
            "intent": "percentage_followup_no_baseline",
            "reply": (
                f"I have the previous result for {LAST_CONTEXT.get('label', 'the last query')}, "
                "but no baseline was available to calculate percentage change."
            ),
        }

    direction = "increase" if pct_change > 0 else "decrease" if pct_change < 0 else "change"

    return {
        "intent": "percentage_followup",
        "reply": (
            f"For {LAST_CONTEXT.get('label')}, the percentage {direction} is "
            f"{abs(pct_change)}%.\n\n"
            f"Forecast average: {LAST_CONTEXT.get('forecast_mean')} tCO2e/day.\n"
            f"Baseline: {LAST_CONTEXT.get('baseline_mean')} tCO2e/day "
            f"({LAST_CONTEXT.get('baseline_label')})."
        ),
        "data": LAST_CONTEXT,
    }


# -------------------------------------------------
# Main chatbot handler
# -------------------------------------------------
def handle_message(message: str) -> dict:
    """
    Main entry point.
    Returns:
      {"intent": str, "reply": str, "data": optional}
    """

    text = message.lower().strip()

    if not text:
        return {
            "intent": "empty",
            "reply": "Please enter a question about CO2 emissions, equipment emissions, forecasts, or trends.",
        }

    # -------------------------------------------------
    # 1. Greeting
    # -------------------------------------------------
    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", text):
        return {
            "intent": "greeting",
            "reply": (
                "Hello! I'm the INEOS CO2 Emissions Assistant. "
                "I can answer historical CO2 questions, future forecasts, month-wise forecasts, "
                "equipment-wise emissions, energy consumption, root cause, and reports."
            ),
        }

    # -------------------------------------------------
    # 2. Help
    # -------------------------------------------------
    if "help" in text or "what can you do" in text:
        return {
            "intent": "help",
            "reply": (
                "I can help with:\n"
                "- Overall stats: 'What is the average CO2 emission?'\n"
                "- Yearly stats: 'Total CO2 emitted in 2020'\n"
                "- Month data: 'Provide data for February 2020'\n"
                "- Future month forecast: 'Provide data for February month 2030'\n"
                "- Percentage change: 'How much percentage increase?'\n"
                "- Comparisons: 'Compare 2015 and 2020'\n"
                "- Trend: 'What is the CO2 trend over the years?'\n"
                "- Seasonality: 'Which month has the highest emissions?'\n"
                "- Forecasts: 'Predict CO2 for next 30 days' or 'What will CO2 be on 2030-02-15?'"
            ),
        }

    years_found = _extract_years(text)
    month_found = _extract_month(text)
    explicit_date = _extract_explicit_date(text)

    hist_min_year = _historical_min_year()
    hist_max_year = _historical_max_year()

    # -------------------------------------------------
    # 3. Percentage follow-up
    # Must come before generic forecast/stats logic.
    # -------------------------------------------------
    if _is_percentage_question(text) and not years_found and not month_found:
        return _reply_for_percentage_followup()

    # -------------------------------------------------
    # 4. Explicit date forecast / historical date
    # -------------------------------------------------
    if explicit_date:
        date_year = int(explicit_date[:4])

        if date_year > hist_max_year:
            result = _forecast_single_date(explicit_date)

            overall = data_utils.overall_stats()
            pct_change = _percentage_change(result["predicted_co2_tCO2e"], overall["mean"])

            _save_context({
                "type": "forecast_date",
                "label": explicit_date,
                "forecast_mean": round(result["predicted_co2_tCO2e"], 2),
                "baseline_label": f"historical average from {hist_min_year}-{hist_max_year}",
                "baseline_mean": overall["mean"],
                "pct_change": pct_change,
            })

            direction = "higher than" if pct_change and pct_change > 0 else "lower than" if pct_change and pct_change < 0 else "equal to"

            return {
                "intent": "forecast_date",
                "reply": (
                    f"Predicted CO2 emission on {result['date']}: "
                    f"{round(result['predicted_co2_tCO2e'], 2)} tCO2e.\n\n"
                    f"Compared with the historical average ({overall['mean']} tCO2e/day), "
                    f"this is {abs(pct_change)}% {direction} historical levels."
                ),
                "data": {
                    "forecast": result,
                    "baseline": overall,
                    "pct_change": pct_change,
                },
            }

        # Historical specific date
        df = data_utils.load_data().copy()
        df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
        row = df[df["date_str"] == explicit_date]

        if not row.empty:
            value = round(float(row.iloc[0]["CO2_Emissions_tCO2e"]), 2)

            _save_context({
                "type": "historical_date",
                "label": explicit_date,
                "mean": value,
            })

            return {
                "intent": "historical_date",
                "reply": f"On {explicit_date}, recorded CO2 emission was {value} tCO2e.",
                "data": {
                    "date": explicit_date,
                    "co2_tCO2e": value,
                },
            }

        return {
            "intent": "date_not_found",
            "reply": f"I could not find historical data for {explicit_date}.",
        }

    # -------------------------------------------------
    # 5. Month + year request
    # Example: "Provide data for February month 2030"
    # -------------------------------------------------
    if month_found and years_found and _is_data_request(text):
        year = years_found[0]

        if year > hist_max_year:
            return _reply_for_future_month(year, month_found)

        if hist_min_year <= year <= hist_max_year:
            return _reply_for_historical_month(year, month_found)

        return {
            "intent": "year_out_of_range",
            "reply": (
                f"I do not have data for {year}. Historical data covers "
                f"{hist_min_year}-{hist_max_year}. For future years after {hist_max_year}, I can forecast."
            ),
        }

    # -------------------------------------------------
    # 6. Future year request
    # Example: "Provide data for 2030"
    # -------------------------------------------------
    if years_found and years_found[0] > hist_max_year and _is_data_request(text):
        return _reply_for_future_year(years_found[0])

    # -------------------------------------------------
    # 7. Forecast / prediction by keyword
    # -------------------------------------------------
    if _is_forecast_question(text):
        if month_found and years_found:
            year = years_found[0]
            return _reply_for_future_month(year, month_found)

        if years_found and years_found[0] > hist_max_year:
            return _reply_for_future_year(years_found[0])

        days = _extract_days(text) or 30
        forecast_stats = _forecast_next_days(days)

        overall = data_utils.overall_stats()
        pct_change = _percentage_change(forecast_stats["mean"], overall["mean"])

        _save_context({
            "type": "forecast_range",
            "label": f"next {days} days",
            "forecast_mean": forecast_stats["mean"],
            "forecast_total": forecast_stats["total"],
            "baseline_label": f"historical average from {hist_min_year}-{hist_max_year}",
            "baseline_mean": overall["mean"],
            "pct_change": pct_change,
        })

        direction = "higher than" if pct_change and pct_change > 0 else "lower than" if pct_change and pct_change < 0 else "equal to"

        return {
            "intent": "forecast_range",
            "reply": (
                f"Forecast for the next {days} days starting {forecast_stats['start_date']}:\n"
                f"- Average predicted emission: {forecast_stats['mean']} tCO2e/day\n"
                f"- Total predicted emission: {forecast_stats['total']} tCO2e\n"
                f"- Range: {forecast_stats['min']} - {forecast_stats['max']} tCO2e/day\n\n"
                f"Compared with the historical average ({overall['mean']} tCO2e/day), "
                f"this is {abs(pct_change)}% {direction} historical levels."
            ),
            "data": {
                "forecast": forecast_stats,
                "baseline": overall,
                "pct_change": pct_change,
            },
        }

    # -------------------------------------------------
    # 8. Compare two years
    # -------------------------------------------------
    if "compare" in text and len(years_found) >= 2:
        year_a, year_b = years_found[0], years_found[1]

        comp = data_utils.compare_years(year_a, year_b)

        if comp is None:
            return {
                "intent": "compare_years_missing",
                "reply": f"I do not have historical data for one of those years. Valid historical range: {hist_min_year}-{hist_max_year}.",
            }

        direction = "higher" if comp["mean_diff"] > 0 else "lower"

        _save_context({
            "type": "compare_years",
            "label": f"{year_a} vs {year_b}",
            "pct_change": comp["pct_change"],
            "mean_diff": comp["mean_diff"],
        })

        return {
            "intent": "compare_years",
            "reply": (
                f"In {comp['year_a']['year']}, average daily CO2 was "
                f"{comp['year_a']['mean']} tCO2e/day, compared with "
                f"{comp['year_b']['mean']} tCO2e/day in {comp['year_b']['year']}.\n\n"
                f"Difference: {abs(comp['mean_diff'])} tCO2e/day {direction}, "
                f"which is {abs(comp['pct_change'])}%."
            ),
            "data": comp,
        }

    # -------------------------------------------------
    # 9. Historical year stats
    # -------------------------------------------------
    if years_found:
        year = years_found[0]

        if year > hist_max_year:
            return _reply_for_future_year(year)

        stats = data_utils.year_stats(year)

        if stats is None:
            return {
                "intent": "year_stats_missing",
                "reply": f"I do not have historical data for {year}. Data covers {hist_min_year}-{hist_max_year}.",
            }

        _save_context({
            "type": "historical_year",
            "label": str(year),
            "mean": stats["mean"],
            "total": stats["total"],
        })

        if "total" in text or "sum" in text:
            reply = f"Total CO2 emitted in {year}: {stats['total']:,} tCO2e."
        elif "max" in text or "highest" in text or "peak" in text:
            reply = f"Highest single-day CO2 emission in {year}: {stats['max']} tCO2e."
        elif "min" in text or "lowest" in text:
            reply = f"Lowest single-day CO2 emission in {year}: {stats['min']} tCO2e."
        else:
            reply = (
                f"For {year}, average daily CO2 emission was {stats['mean']} tCO2e/day.\n"
                f"Total CO2 emitted: {stats['total']:,} tCO2e.\n"
                f"Range: {stats['min']} - {stats['max']} tCO2e/day."
            )

        return {
            "intent": "year_stats",
            "reply": reply,
            "data": stats,
        }

    # -------------------------------------------------
    # 10. Trend
    # -------------------------------------------------
    if "trend" in text:
        trend = data_utils.yearly_trend()
        first_year, last_year = min(trend), max(trend)
        change = round(trend[last_year] - trend[first_year], 2)
        pct_change = _percentage_change(trend[last_year], trend[first_year])

        direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed roughly flat"

        _save_context({
            "type": "trend",
            "label": f"{first_year}-{last_year}",
            "pct_change": pct_change,
            "mean_diff": change,
        })

        return {
            "intent": "trend",
            "reply": (
                f"Average daily CO2 emissions {direction} from "
                f"{trend[first_year]} tCO2e/day in {first_year} to "
                f"{trend[last_year]} tCO2e/day in {last_year}.\n\n"
                f"Change: {change} tCO2e/day "
                f"({abs(pct_change)}% {'increase' if pct_change > 0 else 'decrease' if pct_change < 0 else 'change'})."
            ),
            "data": trend,
        }

    # -------------------------------------------------
    # 11. Month / seasonality without year
    # -------------------------------------------------
    if month_found:
        baseline = _historical_month_average_across_years(month_found)

        if baseline:
            _save_context({
                "type": "historical_month_average",
                "label": f"historical {MONTH_NAMES[month_found]} average",
                "mean": baseline["mean"],
                "total": baseline["total"],
            })

            return {
                "intent": "historical_month_average",
                "reply": (
                    f"Across the historical dataset ({hist_min_year}-{hist_max_year}), "
                    f"{MONTH_NAMES[month_found]} average daily CO2 emission is "
                    f"{baseline['mean']} tCO2e/day.\n"
                    f"Range across all {MONTH_NAMES[month_found]} records: "
                    f"{baseline['min']} - {baseline['max']} tCO2e/day."
                ),
                "data": baseline,
            }

    if "month" in text or "season" in text:
        seasonal = data_utils.monthly_seasonality()
        peak_month = max(seasonal, key=seasonal.get)
        low_month = min(seasonal, key=seasonal.get)

        return {
            "intent": "seasonality",
            "reply": (
                f"Emissions are highest in {MONTH_NAMES[peak_month]} "
                f"(average {seasonal[peak_month]} tCO2e/day) and lowest in "
                f"{MONTH_NAMES[low_month]} (average {seasonal[low_month]} tCO2e/day)."
            ),
            "data": seasonal,
        }

    # -------------------------------------------------
    # 12. Correlations / drivers
    # -------------------------------------------------
    if any(k in text for k in ["factor", "driver", "correlat", "influence"]):
        corr = data_utils.top_correlations()
        top3 = list(corr.items())[:3]
        top_str = ", ".join([f"{k} ({v})" for k, v in top3])

        return {
            "intent": "correlations",
            "reply": (
                f"The variables most strongly correlated with CO2 emissions are: {top_str}. "
                f"Natural gas consumption is typically the biggest operational driver."
            ),
            "data": corr,
        }

    # -------------------------------------------------
    # 13. Highest / lowest record days
    # -------------------------------------------------
    if ("highest" in text or "lowest" in text or "record" in text) and not years_found:
        hl = data_utils.highest_lowest_days()
        top = hl["highest"][0]
        bottom = hl["lowest"][0]

        return {
            "intent": "records",
            "reply": (
                f"Highest CO2 day ever recorded: {top['Date']} with "
                f"{top['CO2_Emissions_tCO2e']} tCO2e.\n"
                f"Lowest CO2 day ever recorded: {bottom['Date']} with "
                f"{bottom['CO2_Emissions_tCO2e']} tCO2e."
            ),
            "data": hl,
        }

    # -------------------------------------------------
    # 14. Overall stats
    # -------------------------------------------------
    if any(k in text for k in ["average", "mean", "total", "overall", "how much", "co2", "emission"]):
        stats = data_utils.overall_stats()

        _save_context({
            "type": "overall_stats",
            "label": "overall historical dataset",
            "mean": stats["mean"],
            "total": stats["total"],
        })

        return {
            "intent": "overall_stats",
            "reply": (
                f"Across {stats['records']} days ({stats['date_range']}), "
                f"average daily CO2 emission was {stats['mean']} tCO2e/day.\n"
                f"Range: {stats['min']} - {stats['max']} tCO2e/day.\n"
                f"Total emitted over the full period: {stats['total']:,} tCO2e."
            ),
            "data": stats,
        }

    # -------------------------------------------------
    # 15. Fallback
    # -------------------------------------------------
    return {
        "intent": "fallback",
        "reply": (
            "I could not map that to a specific calculation. Try asking:\n"
            "- Provide data for February month 2030\n"
            "- How much percentage increase?\n"
            "- Provide data for February 2020\n"
            "- Compare 2015 and 2020\n"
            "- Predict CO2 for next 30 days\n"
            "- Show equipment-wise CO2 generation"
        ),
    }