# CO2 Watch — INEOS Emissions Chatbot & Forecaster

A hackathon-ready project built on the **INEOS Daily CO2 Dataset (2010–2024)**.
It has three parts:

1. **A forecasting model** trained on daily CO2 emissions (tCO2e), able to predict
   future emissions for any date.
2. **A chatbot backend** (Flask API) that answers natural-language questions about
   the historical data AND calls the forecasting model for "predict/forecast" questions.
3. **A chat frontend** (single HTML file) that talks to the backend.

```
co2_chatbot_project/
├── data/
│   └── INEOS_Daily_CO2_Dataset_2010_2024.xlsx
├── model/
│   ├── train_model.py          <- trains + saves the forecasting model
│   ├── co2_forecast_model.pkl  <- generated after training
│   └── co2_metadata.pkl        <- generated after training
├── backend/
│   ├── data_utils.py           <- historical stats / queries
│   ├── predictor.py            <- loads model, forecasts future dates
│   ├── chatbot.py              <- NLU (intent detection) + reply generation
│   └── app.py                  <- Flask API that ties it all together
├── frontend/
│   └── index.html              <- chat UI (vanilla HTML/CSS/JS)
└── requirements.txt
```

---

## 1. Setup

```bash
cd co2_chatbot_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the forecasting model

```bash
cd model
python train_model.py
```

This reads the Excel dataset, engineers date-based features (year trend +
cyclical day-of-year), trains a `GradientBoostingRegressor`, and saves:
- `co2_forecast_model.pkl`
- `co2_metadata.pkl`

Console output on this dataset (for reference):
```
Test MAE : 57.05 tCO2e
Test R^2 : 0.774
```
The model only uses the **date** as input (not future production/energy values),
so it can genuinely forecast any future day — perfect for a live demo.

> Why is R² "only" 0.77 and not 0.99? Because day-to-day CO2 in this dataset has
> real random noise (operational variance) on top of the seasonal pattern — a
> model that fits that noise perfectly would be **overfit**. 0.77 R² with a ~2.4%
> average error means the model has cleanly captured the genuine seasonal +
> trend signal, which is the honest, presentable number for a hackathon demo.

## 3. Run the backend

```bash
cd backend
python app.py
```

This starts a Flask server at `http://localhost:5000` with these endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | health check |
| POST | `/api/chat` | `{"message": "..."}` → chatbot reply |
| GET | `/api/stats` | overall dataset stats |
| GET | `/api/stats/<year>` | stats for one year |
| GET | `/api/forecast?days=30` | forecast next N days from tomorrow |
| GET | `/api/forecast?date=2025-06-01` | forecast a specific date |
| GET | `/api/trend` | yearly average trend |
| GET | `/api/seasonality` | monthly average seasonality |

Test it quickly with curl:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"predict CO2 for the next 14 days"}'
```

## 4. Run the frontend

Just open `frontend/index.html` in your browser (double-click it, or drag it in).
It calls `http://localhost:5000/api/...`, so **keep the backend running** in a
separate terminal.

If you want to serve it properly instead of opening the file directly (avoids
some browsers' file:// restrictions):
```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

---

## How the chatbot understands questions (no external API needed)

`chatbot.py` uses **rule-based intent detection** — regex + keyword matching —
to figure out what the user wants, then pulls the exact numbers from
`data_utils.py` (historical) or `predictor.py` (forecast) and formats a reply.
This is intentionally dependency-free (no API key, no cost, no network needed),
which matters a lot for a live hackathon demo where you can't risk a flaky
internet connection or rate limits.

Supported question types:
- Overall stats — *"What is the average CO2 emission?"*
- Yearly stats — *"Total CO2 emitted in 2018"*
- Year comparison — *"Compare 2015 and 2020"*
- Long-term trend — *"What is the CO2 trend over the years?"*
- Seasonality — *"Which month has the highest emissions?"*
- Correlation/drivers — *"What factors drive CO2 emissions?"*
- Record days — *"What was the highest CO2 day ever recorded?"*
- Forecasts — *"Predict CO2 for the next 30 days"* / *"What will CO2 emissions be on 2025-06-01?"*

---

## Optional upgrade: make replies sound more natural with an LLM

The rule-based chatbot above is fast and demo-safe, but you can optionally pipe
its structured data into an LLM (e.g. the Claude API) to phrase the final
answer more conversationally. The key idea is **retrieval before generation**:
you already have the exact numbers from `data_utils`/`predictor` — you're only
asking the LLM to phrase them nicely, not to invent numbers.

```python
# backend/llm_polish.py  (optional add-on)
import requests

def polish_with_claude(user_question, structured_data, api_key):
    prompt = f"""A user asked: "{user_question}"
Here is the exact data retrieved from our CO2 database: {structured_data}
Write a short, natural, one-paragraph answer using ONLY these numbers. Do not invent any figures."""

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return resp.json()["content"][0]["text"]
```

Then in `app.py`'s `/api/chat` route, after calling `chatbot.handle_message()`,
optionally call `polish_with_claude(message, result["data"], api_key)` and use
that as the reply instead. Keep the rule-based path as a fallback if the API
call fails or there's no key configured — that way your demo never breaks even
if the venue wifi drops.

---

## Linking the chatbot to your backend (the wiring, explained)

This is the core pattern for **any** frontend-to-chatbot-backend connection,
not just this project:

1. **Backend exposes an HTTP endpoint** that accepts a user message and returns
   a reply. Here that's `POST /api/chat` in `app.py`.
2. **CORS must be enabled** on the backend if your frontend is served from a
   different origin/port than the backend (e.g. frontend on `:8080`, backend on
   `:5000`). This project uses `flask-cors`'s `CORS(app)` for that — without it,
   the browser will block the request with a CORS error in the console.
3. **Frontend calls the endpoint with `fetch()`**, sends the user's message as
   JSON, and renders `response.reply` in the chat window. See the `sendMessage()`
   function in `index.html`.
4. **State lives in the browser**, not the backend — each request is independent
   (the backend has no memory of prior messages in this simple version). If you
   want multi-turn memory, you'd pass the whole conversation history in the
   request body and adapt `chatbot.py` to use it.

### Deploying so others can use it (post-hackathon)
- Backend: any host that runs Python (Render, Railway, Fly.io, an EC2 box, etc).
  Just make sure `co2_forecast_model.pkl`/`co2_metadata.pkl` are generated
  (run `train_model.py`) and committed/uploaded alongside the code, and the
  Excel file is present at `data/INEOS_Daily_CO2_Dataset_2010_2024.xlsx`.
- Frontend: any static host (GitHub Pages, Netlify, Vercel) — just update
  `API_BASE` in `index.html` to your deployed backend's URL instead of
  `http://localhost:5000`.

---

## Ideas to extend for extra hackathon points
- Swap `GradientBoostingRegressor` for an ensemble (average of GBR + RandomForest)
  and show the confidence interval on the frontend.
- Add a `/api/whatif` endpoint that lets users adjust `Production_Index` or
  `Natural_Gas_GJ` and see the modeled CO2 impact — great for a "what if we cut
  gas usage by 10%?" demo moment.
- Plot a chart (recharts/Chart.js) of historical + forecasted emissions on the
  frontend for a strong visual "wow" moment during judging.
- Add voice input (Web Speech API) to the chat box for a flashier demo.
