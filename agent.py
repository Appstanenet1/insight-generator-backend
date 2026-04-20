import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# --- 1. FastAPI Setup ---
app = FastAPI()

# Enable CORS so your Next.js frontend can talk to this API on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
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

SQL_URI = "sqlite:///mock_database.db"
db = SQLDatabase.from_uri(SQL_URI, include_tables=["ai_campaign_performance_mart"])

custom_prefix = """
You are an expert Marketing Data Analyst with deep knowledge of Google Ads.
You are analyzing data from a local SQLite database.
You must ONLY query the `ai_campaign_performance_mart` table. Use standard SQLite syntax.

Here is the schema and definition of the metrics:
* `date`: The date of the performance record (YYYY-MM-DD). 
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
5. NO LIVE DATES (CRITICAL): NEVER use `date('now')`, `CURRENT_DATE`, or `today`. This database is a static historical export. To look back X days, you MUST anchor to the most recent data point. Example for yesterday: `date((SELECT MAX(date) FROM ai_campaign_performance_mart), '-1 day')`.
6. When you provide your final answer, state the actual numbers (Spend, CPA, ROAS, etc.) to back up your claims.

CRITICAL FORMATTING RULES:
1. Always aggregate data (SUM, AVG) when querying across multiple days.
2. NEVER execute DML commands (INSERT, UPDATE, DELETE, DROP).
"""

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit, 
    verbose=True, 
    prefix=custom_prefix,
    agent_type="openai-tools", 
    handle_parsing_errors=True,
    top_k=10 
)

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
        
        response = agent_executor.invoke({"input": full_prompt})
        output = response["output"]
        
        # Save memory
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=output))
        chat_history = chat_history[-4:]
        
        return {"reply": output}
        
    except Exception as e:
        return {"reply": f"Error processing request: {str(e)}"}

# --- 4. Server Execution ---
if __name__ == "__main__":
    print("Starting Insight Generator API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)