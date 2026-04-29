import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables (This will automatically grab GOOGLE_APPLICATION_CREDENTIALS)
load_dotenv()

# --- 1. FastAPI Setup ---
app = FastAPI()

# Enable CORS so your Next.js frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update this to your Vercel URL later for strict security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LangChain & Database Setup ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model="x-ai/grok-4.1-fast", 
    temperature=0,
)

# --- BIGQUERY CLOUD CONNECTION ---
# Replace these strings with your actual GCP details or put them in your .env file
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
BQ_DATASET_ID = os.environ.get("BQ_DATASET_ID", "your_dataset_name")

# The SQLAlchemy connection string for BigQuery
BQ_URI = f"bigquery://{GCP_PROJECT_ID}/{BQ_DATASET_ID}"

# Lightweight engine for direct API queries.
dashboard_engine = create_engine(BQ_URI)

custom_prefix = """
You are an expert Marketing Data Analyst with deep knowledge of Google Ads.
You are analyzing data from a live Google Cloud BigQuery dataset.
You must ONLY query the `ai_campaign_performance_mart` table. Use BigQuery Standard SQL syntax.

Here is the schema and definition of the metrics:
* `date`: The date of the performance record (DATE). 
* `campaign_name`: The human-readable name of the campaign.
* `campaign_status`: The status of the campaign (e.g., 'ENABLED', 'PAUSED').
* `cost_inr`: Total spend for the day, in Indian Rupees.
* `impressions`: Number of times the ad was shown.
* `clicks`: Number of times the ad was clicked.
* `ctr`: Click-through rate.
* `conversions`: Total number of conversions.
* `cost_per_conversion`: Cost per acquisition (CPA).
* `conversion_value`: Total value/revenue generated from conversions.
* `roas`: Return on Ad Spend for that specific day.

YOUR ANALYTICAL GUIDELINES & MATHEMATICAL RULES:
1. STRICT ROAS CALCULATION: When aggregating data, you MUST calculate ROAS as `SUM(conversion_value) / SUM(cost_inr)`. NEVER use `AVG(roas)` or `MAX(roas)`. Always filter for `SUM(cost_inr) > 100`.
2. ACTIVE CAMPAIGNS ONLY: Always filter your queries using `WHERE campaign_status = 'ENABLED'` unless the user explicitly asks for paused campaigns.
3. AGGREGATE AT THE RIGHT LEVEL: If the user asks for a high-level metric, aggregate ALL matching campaigns into ONE single overarching number.
4. SELECT WHAT YOU GROUP: If you ever use a `GROUP BY` clause, you MUST include that exact column in your `SELECT` statement.
5. BIGQUERY DATES: Use BigQuery functions like `CURRENT_DATE()` for today. To look back X days, use `DATE_SUB(CURRENT_DATE(), INTERVAL X DAY)`.
6. When you provide your final answer, state the actual numbers (Spend, CPA, ROAS, etc.) to back up your claims.

CRITICAL FORMATTING RULES:
1. Always aggregate data (SUM, AVG) when querying across multiple days.
2. NEVER execute DML commands (INSERT, UPDATE, DELETE, DROP).
"""

chat_db = None
agent_executor = None


def get_agent_executor():
    global chat_db, agent_executor

    if agent_executor is not None:
        return agent_executor

    chat_db = SQLDatabase.from_uri(BQ_URI)
    toolkit = SQLDatabaseToolkit(db=chat_db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=custom_prefix,
        agent_type="openai-tools",
        handle_parsing_errors=True,
        top_k=10,
    )
    return agent_executor

# --- 3. API Endpoints & State ---
chat_history = []

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global chat_history
    user_query = request.message
    
    try:
        # Build context string
        context_string = ""
        if chat_history:
            context_string = "PREVIOUS CONVERSATION CONTEXT:\n"
            for msg in chat_history:
                role = "User" if isinstance(msg, HumanMessage) else "AI Analyst"
                context_string += f"{role}: {msg.content}\n"
            context_string += "\n"
        
        full_prompt = f"{context_string}NEW QUESTION: {user_query}"

        executor = get_agent_executor()
        response = executor.invoke({"input": full_prompt})
        output = response["output"]
        
        # Save memory
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=output))
        chat_history = chat_history[-4:]
        
        return {"reply": output}
        
    except Exception as e:
        return {"reply": f"🚨 **BigQuery Connection Error:** {str(e)}"}


@app.get("/api/dashboard-metrics")
async def dashboard_metrics_endpoint(range: str = "7d"):
    from datetime import date, datetime, timedelta
    from sqlalchemy import text

    range_days_lookup = {
        "7d": 7,
        "14d": 14,
        "30d": 30,
    }
    range_days = range_days_lookup.get(range, 7)

    def normalize_number(value):
        if value is None:
            return 0.0
        return float(value)

    def calculate_percent_change(current_value, previous_value):
        if previous_value in (None, 0):
            return None
        return ((current_value - previous_value) / abs(previous_value)) * 100

    def build_metric_payload(current_value, previous_value):
        current_number = normalize_number(current_value)
        previous_number = normalize_number(previous_value)
        return {
            "value": current_number,
            "previousValue": previous_number,
            "deltaPercent": calculate_percent_change(current_number, previous_number),
        }

    latest_date_sql = text(
        """
        SELECT MAX(date) AS max_date
        FROM ai_campaign_performance_mart
        WHERE campaign_status = 'ENABLED'
        """
    )

    summary_sql = text(
        """
        SELECT
          SUM(cost_inr) AS total_cost,
          SUM(conversions) AS total_conversions,
          SUM(conversion_value) AS total_revenue,
          SAFE_DIVIDE(SUM(cost_inr), NULLIF(SUM(conversions), 0)) AS average_cpa,
          SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_inr), 0)) AS overall_roas
        FROM ai_campaign_performance_mart
        WHERE campaign_status = 'ENABLED'
          AND date BETWEEN :start_date AND :end_date
        """
    )

    try:
        with dashboard_engine.connect() as connection:
            latest_row = connection.execute(latest_date_sql).mappings().one()
            max_date_value = latest_row["max_date"]

            if max_date_value is None:
                return {
                    "asOfDate": "",
                    "selectedRange": range,
                    "rangeDays": range_days,
                    "periodLabel": f"Last {range_days} days",
                    "comparisonLabel": f"Previous {range_days} days",
                    "metrics": {
                        "totalCost": build_metric_payload(0, 0),
                        "totalConversions": build_metric_payload(0, 0),
                        "averageCpa": build_metric_payload(0, 0),
                        "overallRoas": build_metric_payload(0, 0),
                        "totalRevenue": build_metric_payload(0, 0),
                    },
                }

            if isinstance(max_date_value, datetime):
                as_of_date = max_date_value.date()
            elif isinstance(max_date_value, date):
                as_of_date = max_date_value
            else:
                as_of_date = datetime.fromisoformat(str(max_date_value)).date()

            current_start_date = as_of_date - timedelta(days=range_days - 1)
            previous_end_date = current_start_date - timedelta(days=1)
            previous_start_date = previous_end_date - timedelta(days=range_days - 1)

            current_summary = connection.execute(
                summary_sql,
                {
                    "start_date": current_start_date.isoformat(),
                    "end_date": as_of_date.isoformat(),
                },
            ).mappings().one()

            previous_summary = connection.execute(
                summary_sql,
                {
                    "start_date": previous_start_date.isoformat(),
                    "end_date": previous_end_date.isoformat(),
                },
            ).mappings().one()

            return {
                "asOfDate": as_of_date.isoformat(),
                "selectedRange": range,
                "rangeDays": range_days,
                "periodLabel": f"Last {range_days} days",
                "comparisonLabel": f"Previous {range_days} days",
                "metrics": {
                    "totalCost": build_metric_payload(
                        current_summary["total_cost"],
                        previous_summary["total_cost"],
                    ),
                    "totalConversions": build_metric_payload(
                        current_summary["total_conversions"],
                        previous_summary["total_conversions"],
                    ),
                    "averageCpa": build_metric_payload(
                        current_summary["average_cpa"],
                        previous_summary["average_cpa"],
                    ),
                    "overallRoas": build_metric_payload(
                        current_summary["overall_roas"],
                        previous_summary["overall_roas"],
                    ),
                    "totalRevenue": build_metric_payload(
                        current_summary["total_revenue"],
                        previous_summary["total_revenue"],
                    ),
                },
            }
    except Exception as e:
        return {
            "error": f"Unable to load dashboard metrics from BigQuery: {str(e)}"
        }

# --- 4. Server Execution ---
if __name__ == "__main__":
    print("Starting Insight Engine BigQuery API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
