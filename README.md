🌱 AgriBot – AI-Powered Farming Assistant

AgriBot is an intelligent agricultural assistant that helps farmers make better decisions using AI, Knowledge Graphs, and Real-Time Data.

It provides:

🌾 Smart answers to farming queries
🌐 Automatic web-based knowledge updates
📊 Knowledge Graph (Neo4j) powered insights
⚡ Fast similarity search using FAISS
🌦️ Weather-based crop risk alerts


🚀 Features
🤖 AI Question Answering
Farmers can ask queries like:
“Best time to grow wheat?”
System first checks Knowledge Graph
If not found → fetches data from web → updates KG automatically

🧠 Knowledge Graph (Neo4j)
Stores farming entities and relationships
Continuously grows with new information
Improves answer quality over time

⚡ Vector Search (FAISS)
Fast similarity search for relevant nodes
Uses sentence embeddings

🌐 Web Scraping + AI Extraction
Uses Tavily API for web search
Uses LLM (Groq) to extract structured data
Automatically updates Knowledge Graph

🌦️ Crop Risk Alerts
Location-based weather alerts
Detects:
High humidity → fungal disease risk
Heat stress
Heavy rainfall
Provides actionable suggestions

🏗️ Tech Stack
Backend
FastAPI
Neo4j (Knowledge Graph)
MySQL (User & Query storage)
FAISS (Vector Search)
AI / ML
SentenceTransformers (all-MiniLM-L6-v2)
Groq LLM (Llama 3)
External APIs
Tavily (Web Search)
Weather API (for alerts)
Frontend
HTML, CSS, JavaScript
