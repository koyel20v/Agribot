# app.py — AgriGenius FastAPI application
# ══════════════════════════════════════════════════════════════════════════════
#
#  KEY CHANGE (Alerts endpoint):
#    GET /alerts?location=<city>&crop=<optional>
#    - Fetches live weather from Open-Meteo (no API key required)
#    - Runs rule-based alert generation (_generate_crop_alerts)
#    - If `crop` param is provided → filters alerts where alert["crop"] == crop
#      (case-insensitive prefix match so "Rice" matches "Rice / Paddy" etc.)
#    - Returns: { location, weather, fetched_at, alerts }
#
#  All other code (auth, /ask pipeline, officer routes) is unchanged.
# ══════════════════════════════════════════════════════════════════════════════

import os, json, re, logging, hashlib, datetime
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import numpy as np
import faiss
import httpx
import mysql.connector
from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agrigenius")

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS     = os.getenv("NEO4J_PASS",     "1234llmk")

DB_HOST        = os.getenv("DB_HOST",        "localhost")
DB_PORT        = int(os.getenv("DB_PORT",    3306))
DB_USER        = os.getenv("DB_USER",        "root")
DB_PASSWORD    = os.getenv("DB_PASSWORD",    "koye20@%")
DB_NAME        = os.getenv("DB_NAME",        "agrigenius_db")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

JWT_SECRET     = os.getenv("JWT_SECRET",     "change_this_in_production")
JWT_ALGORITHM  = "HS256"
JWT_EXPIRY_HRS = int(os.getenv("JWT_EXPIRY_HOURS", 24))

FAISS_DIR      = os.getenv("FAISS_DIR",      "faiss_index")
FAISS_INDEX    = os.path.join(FAISS_DIR, "index.faiss")
FAISS_META     = os.path.join(FAISS_DIR, "meta.json")
THRESHOLD      = float(os.getenv("FAISS_THRESHOLD", "0.55"))
EMBEDDING_DIM  = 384

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ══════════════════════════════════════════════════════════════════
# GLOBAL SINGLETONS
# ══════════════════════════════════════════════════════════════════
neo4j_driver  = None
faiss_index   = None
metadata      = []
embedder      = None
llm           = None
search_tool   = None

# ══════════════════════════════════════════════════════════════════
# MYSQL HELPERS
# ══════════════════════════════════════════════════════════════════

def get_db():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME
    )

def init_db():
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INT AUTO_INCREMENT PRIMARY KEY,
            name      VARCHAR(120) NOT NULL,
            email     VARCHAR(120) UNIQUE NOT NULL,
            password  VARCHAR(256) NOT NULL,
            role      ENUM('farmer','officer') NOT NULL DEFAULT 'farmer',
            location  VARCHAR(100),
            language  VARCHAR(10) DEFAULT 'en',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id    INT AUTO_INCREMENT PRIMARY KEY,
            user_id     INT NOT NULL,
            question    TEXT NOT NULL,
            ai_response TEXT,
            source      VARCHAR(20) DEFAULT 'none',
            status      ENUM('pending','approved','rejected') DEFAULT 'pending',
            validated_answer TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    log.info("[DB] Tables verified/created")

# ══════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed

def create_token(user_id: int, email: str, role: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HRS)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "role": role, "exp": expire},
        JWT_SECRET, algorithm=JWT_ALGORITHM
    )

security = HTTPBearer()

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "email": payload["email"], "role": payload["role"]}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")

def require_farmer(user=Depends(get_current_user)):
    if user["role"] != "farmer":
        raise HTTPException(status_code=403, detail="Farmers only.")
    return user

def require_officer(user=Depends(get_current_user)):
    if user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Officers only.")
    return user

# ══════════════════════════════════════════════════════════════════
# FAISS / NEO4J STARTUP
# ══════════════════════════════════════════════════════════════════

def _init_globals():
    global neo4j_driver, faiss_index, metadata, embedder, llm, search_tool

    log.info("[INIT] Connecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    log.info("[INIT] Loading SentenceTransformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    log.info("[INIT] Loading Groq LLM...")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)

    log.info("[INIT] Loading Tavily search tool...")
    search_tool = TavilySearchResults(k=3)

    if os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_META):
        log.info("[INIT] Loading FAISS from disk...")
        faiss_index = faiss.read_index(FAISS_INDEX)
        with open(FAISS_META, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        log.info(f"[INIT] FAISS loaded — {faiss_index.ntotal} vectors")
    else:
        log.info("[INIT] Building FAISS from Neo4j embeddings...")
        _build_faiss_from_kg()

    log.info("[INIT] All globals ready ✅")


def _build_faiss_from_kg():
    global faiss_index, metadata
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (n) WHERE n.embedding IS NOT NULL
            RETURN id(n) AS id, n.name AS name, n.embedding AS embedding
        """).data()

    if not records:
        log.warning("[INIT] No KG embeddings found — empty FAISS index created")
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata    = []
        _persist_faiss()
        return

    dim           = len(records[0]["embedding"])
    embeddings_np = np.array([r["embedding"] for r in records], dtype="float32")
    faiss_index   = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings_np)
    metadata = [{"id": r["id"], "name": r["name"]} for r in records]
    os.makedirs(FAISS_DIR, exist_ok=True)
    _persist_faiss()
    log.info(f"[INIT] FAISS built — {faiss_index.ntotal} vectors")


def _persist_faiss():
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(faiss_index, FAISS_INDEX)
    with open(FAISS_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _init_globals()
    yield
    if neo4j_driver:
        neo4j_driver.close()

# ══════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════

def _embed(text: str) -> np.ndarray:
    return embedder.encode(text).astype("float32")


def _search_faiss(query_vec: np.ndarray, k: int = 5):
    if faiss_index.ntotal == 0:
        return np.array([float("inf")]), []
    vec = query_vec.reshape(1, -1)
    distances, indices = faiss_index.search(vec, min(k, faiss_index.ntotal))
    top_nodes = [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]
    return distances[0], top_nodes


def _get_kg_context(node_ids: list) -> str:
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (a)-[r]->(b) WHERE id(a) IN $ids
            RETURN a.name AS source, type(r) AS relation, b.name AS target
        """, ids=node_ids).data()
    if not records:
        return ""
    return "\n".join(f"{r['source']} -[{r['relation']}]-> {r['target']}" for r in records)


# Web pipeline constants
_WEB_MIN_SENT_LEN  = 20
_WEB_CLEAN_MAX_CHARS = 3000
_KG_FIELD_MAX_LEN  = 120
_RELATION_MIN_LEN  = 3
_RELATION_BLACKLIST = {"is", "has", "are", "was", "be", "and", "or", "the"}

_EXTRACTION_SYSTEM_PROMPT = """\
You are a structured knowledge extractor for an agricultural advisory system.

RULES (non-negotiable):
1. Output ONLY a valid JSON array — no explanation, no markdown, no code fences.
2. Every element must have exactly these three string keys:
   "source"   — the agricultural entity or concept
   "relation" — a concise relationship verb/phrase (e.g. requires, causes, treated_by)
   "target"   — the related entity or value
3. Extract ONLY farming / agriculture related triples.
4. If no relevant knowledge exists in the text, return exactly: []
5. Do NOT invent facts. Only extract what is explicitly stated.
"""


def _clean_web_text(raw) -> str:
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(item.get("content", "") or item.get("snippet", ""))
            else:
                parts.append(str(item))
        raw = " ".join(parts)

    text = str(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences  = re.split(r"[.\n;]+", text)
    meaningful = [s.strip() for s in sentences if len(s.strip()) >= _WEB_MIN_SENT_LEN]
    cleaned    = ". ".join(meaningful)

    if len(cleaned) > _WEB_CLEAN_MAX_CHARS:
        cleaned = cleaned[:_WEB_CLEAN_MAX_CHARS]
        last_space = cleaned.rfind(" ")
        if last_space > _WEB_CLEAN_MAX_CHARS * 0.8:
            cleaned = cleaned[:last_space]

    return cleaned


def _llm_extract_triples(cleaned_text: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract farming knowledge triples from this text:\n\n{cleaned_text}"),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _extract_json_array(text: str):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _validate_triple(fact: dict, idx: int) -> bool:
    if not isinstance(fact, dict):
        return False
    for field in ("source", "relation", "target"):
        val = fact.get(field)
        if val is None or not isinstance(val, str) or not val.strip():
            return False
        if len(val.strip()) > _KG_FIELD_MAX_LEN:
            return False
    relation = fact["relation"].strip().lower()
    if len(relation) < _RELATION_MIN_LEN or relation in _RELATION_BLACKLIST:
        return False
    return True


def _sanitise_triple(fact: dict) -> dict:
    source   = fact["source"].strip()
    target   = fact["target"].strip()
    relation = re.sub(r"\s+", "_", fact["relation"].strip().lower())
    return {"source": source, "relation": relation, "target": target}


def _insert_triples_into_kg(valid_triples: list) -> int:
    inserted = 0
    with neo4j_driver.session() as session:
        for fact in valid_triples:
            try:
                session.run(
                    """
                    MERGE (a:Entity {name: $source})
                    MERGE (b:Entity {name: $target})
                    MERGE (a)-[:RELATION {type: $relation, source: 'web'}]->(b)
                    """,
                    fact,
                )
                inserted += 1
            except Exception as e:
                log.error(f"[KG] Neo4j error: {e}")
    return inserted


def _update_faiss_with_new_nodes(valid_triples: list):
    global metadata
    names_seen: set = set()
    for fact in valid_triples:
        for key in ("source", "target"):
            name = fact.get(key, "").strip()
            if name:
                names_seen.add(name)

    new_vectors, new_meta = [], []
    with neo4j_driver.session() as session:
        for name in names_seen:
            try:
                result   = session.run(
                    "MATCH (n:Entity {name: $name}) RETURN id(n) AS id", name=name
                ).single()
                neo4j_id = result["id"] if result else -1
                new_vectors.append(_embed(name))
                new_meta.append({"id": neo4j_id, "name": name})
            except Exception as e:
                log.error(f"[FAISS] Failed to embed '{name}': {e}")

    if new_vectors:
        faiss_index.add(np.array(new_vectors, dtype="float32"))
        metadata.extend(new_meta)
        _persist_faiss()


def _run_web_pipeline(question: str) -> str:
    log.info(f"[WEB] Tavily search for: {question!r}")
    try:
        web_results = search_tool.run(question)
    except Exception as e:
        log.error(f"[WEB] Tavily failed: {e}")
        return ""

    if not web_results:
        return ""

    cleaned_text = _clean_web_text(web_results)
    if not cleaned_text:
        return str(web_results)

    try:
        raw_llm_output = _llm_extract_triples(cleaned_text)
    except Exception as e:
        log.error(f"[WEB] LLM failed: {e}")
        return str(web_results)

    parsed_triples = _extract_json_array(raw_llm_output)
    if not parsed_triples:
        return str(web_results)

    valid_triples = [
        _sanitise_triple(f) for i, f in enumerate(parsed_triples)
        if isinstance(f, dict) and _validate_triple(f, i)
    ]

    if valid_triples:
        _insert_triples_into_kg(valid_triples)
        _update_faiss_with_new_nodes(valid_triples)

    return str(web_results)


def _generate_answer(question: str, context: str) -> str:
    if context:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Give a clear, practical farming answer using the context above."
        )
    else:
        prompt = (
            f"Question:\n{question}\n\n"
            "Give a clear, practical farming answer based on your knowledge."
        )
    return llm.invoke(prompt).content.strip()

# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AgriGenius API",
    description="AI-powered farming advisory backend",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name:     str
    email:    str
    password: str
    role:     str = "farmer"
    location: str = "India"
    language: str = "en"

class LoginRequest(BaseModel):
    email:    str
    password: str

class AskRequest(BaseModel):
    question: str

class EditAnswerRequest(BaseModel):
    answer: str

# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.post("/auth/register", tags=["Auth"])
def register(data: RegisterRequest):
    if data.role not in ("farmer", "officer"):
        raise HTTPException(400, "Role must be 'farmer' or 'officer'.")
    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (data.email,))
        if cursor.fetchone():
            raise HTTPException(409, "An account with this email already exists.")
        cursor.execute(
            "INSERT INTO users (name, email, password, role, location, language) VALUES (%s,%s,%s,%s,%s,%s)",
            (data.name, data.email, hash_password(data.password), data.role, data.location, data.language)
        )
        conn.commit()
        return {"message": f"Account created. Welcome, {data.name}!"}
    finally:
        cursor.close(); conn.close()


@app.post("/auth/login", tags=["Auth"])
def login(data: LoginRequest):
    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT user_id, name, email, password, role, location, language FROM users WHERE email=%s",
            (data.email,)
        )
        user = cursor.fetchone()
        if not user or not verify_password(data.password, user["password"]):
            raise HTTPException(401, "Invalid email or password.")
        token = create_token(user["user_id"], user["email"], user["role"])
        return {
            "access_token": token,
            "token_type":   "bearer",
            "user_id":      user["user_id"],
            "name":         user["name"],
            "email":        user["email"],
            "role":         user["role"],
            "location":     user["location"],
            "language":     user["language"],
        }
    finally:
        cursor.close(); conn.close()

# ══════════════════════════════════════════════════════════════════
# ASK ROUTE — full pipeline
# ══════════════════════════════════════════════════════════════════

@app.post("/ask", tags=["Query"])
def ask(data: AskRequest, user: dict = Depends(require_farmer)):
    question = data.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    log.info(f"[PIPELINE] user_id={user['user_id']} | Q: {question}")

    query_vec    = _embed(question)
    distances, top_nodes = _search_faiss(query_vec)
    top_distance = float(distances[0]) if len(distances) > 0 else float("inf")
    log.info(f"[PIPELINE] FAISS top distance: {top_distance:.4f} (threshold: {THRESHOLD})")

    context = ""
    source  = "none"

    if top_distance > THRESHOLD:
        log.info("[PIPELINE] No KG match — running web pipeline")
        context = _run_web_pipeline(question)
        source  = "web"
    else:
        log.info("[PIPELINE] KG match found — retrieving relationships")
        node_ids = [n["id"] for n in top_nodes]
        context  = _get_kg_context(node_ids)
        source   = "kg"

    answer = _generate_answer(question, context)

    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "INSERT INTO queries (user_id, question, ai_response, source, status) "
            "VALUES (%s,%s,%s,%s,'pending')",
            (user["user_id"], question, answer, source)
        )
        conn.commit()
        query_id = cursor.lastrowid
        cursor.execute(
            "SELECT query_id, question, ai_response, source, status, created_at "
            "FROM queries WHERE query_id=%s",
            (query_id,)
        )
        record = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

    return {
        "query_id":   record["query_id"],
        "question":   record["question"],
        "answer":     record["ai_response"],
        "source":     record["source"],
        "status":     record["status"],
        "created_at": str(record["created_at"]),
    }

# ══════════════════════════════════════════════════════════════════
# FARMER ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/queries/my", tags=["Query"])
def my_queries(user: dict = Depends(require_farmer)):
    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT query_id, question, ai_response, validated_answer,
                   source, status, created_at
            FROM queries
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 50
        """, (user["user_id"],))
        rows = cursor.fetchall()
        for row in rows:
            row["created_at"] = str(row["created_at"])
        return rows
    finally:
        cursor.close()
        conn.close()

# ══════════════════════════════════════════════════════════════════
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LIVE WEATHER-BASED CROP ALERT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Uses Open-Meteo (free, no API key):
#    1. Geocoding API  → city name → (lat, lon)
#    2. Forecast API   → (lat, lon) → live weather
#
#  Alert rules (unchanged from original):
#    Rice    → flooding risk (rain > 50 mm), moderate rain (>20), fungal (humidity > 80%)
#    Wheat   → humidity stress (>75%), heat stress (>35°C)
#    Maize   → extreme heat (>38°C), drought (humidity < 30%)
#    Cotton  → extreme heat (>40°C)
#    Mustard → frost risk (<5°C)
#    General → wind lodging (>50 km/h)
#
#  NEW: optional `crop` query param for client-side pre-filtering
# ══════════════════════════════════════════════════════════════════

async def _geocode_city(city: str) -> dict:
    """Open-Meteo geocoding: city name → lat, lon, country."""
    url    = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.RequestError as e:
            log.error(f"[GEOCODE] Network error: {e}")
            raise HTTPException(503, f"Geocoding service unavailable: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"[GEOCODE] HTTP error: {e}")
            raise HTTPException(502, "Geocoding API returned an error.")

    results = data.get("results")
    if not results:
        raise HTTPException(404, f"Location '{city}' not found. Please try a different city name.")

    r = results[0]
    return {
        "lat":     r["latitude"],
        "lon":     r["longitude"],
        "name":    r.get("name", city),
        "country": r.get("country", ""),
    }


async def _fetch_weather(lat: float, lon: float) -> dict:
    """Open-Meteo weather: (lat, lon) → current conditions."""
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "current":       "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,weather_code",
        "timezone":      "auto",
        "forecast_days": 1,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.RequestError as e:
            log.error(f"[WEATHER] Network error: {e}")
            raise HTTPException(503, f"Weather service unavailable: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"[WEATHER] HTTP error: {e}")
            raise HTTPException(502, "Weather API returned an error.")

    cur = data.get("current", {})
    return {
        "temp_c":       cur.get("temperature_2m",        25.0),
        "humidity_pct": cur.get("relative_humidity_2m",  60.0),
        "rain_mm":      cur.get("rain",                   0.0),
        "wind_kmh":     cur.get("wind_speed_10m",         0.0),
        "condition":    _wmo_code_to_label(cur.get("weather_code", 0)),
        "weather_code": cur.get("weather_code", 0),
    }


def _wmo_code_to_label(code: int) -> str:
    if code == 0:           return "Clear sky"
    if code in (1, 2, 3):  return "Partly cloudy"
    if code in (45, 48):   return "Foggy"
    if code in (51,53,55): return "Drizzle"
    if code in (61,63,65): return "Rain"
    if code in (71,73,75): return "Snow"
    if code in (80,81,82): return "Rain showers"
    if code in (95,96,99): return "Thunderstorm"
    return "Variable"


# ── _ALERT_RULES — DO NOT MODIFY ──────────────────────────────────
def _generate_crop_alerts(weather: dict) -> List[Dict[str, str]]:
    """
    Rule-based alert engine. Applies weather thresholds to generate
    structured crop risk alerts. Rules are unchanged from original.
    """
    temp     = weather["temp_c"]
    humidity = weather["humidity_pct"]
    rain     = weather["rain_mm"]
    wind     = weather["wind_kmh"]

    alerts = []

    # ── RICE ────────────────────────────────────────────────────────
    if rain > 50:
        alerts.append({
            "risk":       "Heavy Rainfall / Flooding",
            "crop":       "Rice",
            "alert":      f"Rainfall of {rain:.1f} mm detected — high flood risk for low-lying paddy fields.",
            "suggestion": "Open irrigation outlets immediately. Check bunds and drainage channels. Avoid transplanting for 48 hours.",
            "severity":   "high",
            "icon":       "🌧️",
        })
    elif rain > 20:
        alerts.append({
            "risk":       "Moderate Rainfall",
            "crop":       "Rice",
            "alert":      f"Rainfall of {rain:.1f} mm — waterlogging may occur in poorly drained fields.",
            "suggestion": "Monitor water level in fields. Ensure side drainage is clear.",
            "severity":   "medium",
            "icon":       "🌦️",
        })

    if humidity > 80:
        alerts.append({
            "risk":       "High Humidity — Fungal Disease",
            "crop":       "Rice",
            "alert":      f"Humidity at {humidity:.0f}% — conditions favour blast disease and sheath blight.",
            "suggestion": "Apply fungicide (e.g., Tricyclazole) as a preventive measure. Increase plant spacing for better air circulation.",
            "severity":   "high",
            "icon":       "🍄",
        })

    # ── WHEAT ───────────────────────────────────────────────────────
    if humidity > 75:
        alerts.append({
            "risk":       "Humidity Stress — Rust Risk",
            "crop":       "Wheat",
            "alert":      f"Humidity at {humidity:.0f}% — risk of yellow rust and powdery mildew on wheat.",
            "suggestion": "Scout fields for early rust symptoms. Use Propiconazole spray if symptoms appear.",
            "severity":   "medium",
            "icon":       "🌾",
        })

    if temp > 35:
        alerts.append({
            "risk":       "Heat Stress",
            "crop":       "Wheat",
            "alert":      f"Temperature of {temp:.1f}°C — heat stress can cause grain shrivelling in wheat.",
            "suggestion": "Irrigate during cooler morning hours. Avoid waterlogging. Harvest early-maturing varieties promptly.",
            "severity":   "high" if temp > 38 else "medium",
            "icon":       "🌡️",
        })

    # ── MAIZE ───────────────────────────────────────────────────────
    if temp > 38:
        alerts.append({
            "risk":       "Extreme Heat Stress",
            "crop":       "Maize",
            "alert":      f"Temperature of {temp:.1f}°C — pollen viability drops sharply above 38°C, reducing yield.",
            "suggestion": "Increase irrigation frequency. Apply mulch to reduce soil temperature. Avoid planting in the next 5 days.",
            "severity":   "high",
            "icon":       "🌽",
        })

    if humidity < 30:
        alerts.append({
            "risk":       "Drought / Dry Stress",
            "crop":       "Maize",
            "alert":      f"Humidity only {humidity:.0f}% — dry conditions increase water stress on maize.",
            "suggestion": "Schedule drip or furrow irrigation. Apply anti-transpirant spray to reduce leaf water loss.",
            "severity":   "medium",
            "icon":       "🏜️",
        })

    # ── COTTON ──────────────────────────────────────────────────────
    if temp > 40:
        alerts.append({
            "risk":       "Extreme Heat — Boll Drop",
            "crop":       "Cotton",
            "alert":      f"Temperature of {temp:.1f}°C — severe heat can cause boll shedding in cotton.",
            "suggestion": "Apply light irrigation. Avoid heavy nitrogen doses during heat wave. Spray 2% KNO₃ for heat tolerance.",
            "severity":   "high",
            "icon":       "☀️",
        })

    # ── MUSTARD ─────────────────────────────────────────────────────
    if temp < 5:
        alerts.append({
            "risk":       "Frost Risk",
            "crop":       "Mustard",
            "alert":      f"Temperature of {temp:.1f}°C — frost conditions can damage mustard flowers and pods.",
            "suggestion": "Apply light irrigation before predicted frost night (water releases latent heat). Cover young plants with cloth.",
            "severity":   "high",
            "icon":       "❄️",
        })

    # ── GENERAL WIND ────────────────────────────────────────────────
    if wind > 50:
        alerts.append({
            "risk":       "Strong Wind — Lodging Risk",
            "crop":       "All Standing Crops",
            "alert":      f"Wind speed {wind:.0f} km/h — risk of lodging (crop falling over) for tall crops.",
            "suggestion": "Install temporary windbreaks. Postpone spraying operations. Check support structures for climbing crops.",
            "severity":   "medium",
            "icon":       "💨",
        })

    # ── No alerts ────────────────────────────────────────────────────
    if not alerts:
        alerts.append({
            "risk":       "No Immediate Risk",
            "crop":       "All Crops",
            "alert":      f"Conditions normal — Temp {temp:.1f}°C, Humidity {humidity:.0f}%, Rain {rain:.1f} mm.",
            "suggestion": "Continue regular field monitoring. Good time for scouting and routine maintenance.",
            "severity":   "low",
            "icon":       "✅",
        })

    return alerts


def _filter_alerts_by_crop(alerts: List[Dict], crop: str) -> List[Dict]:
    """
    Filter alerts to only those matching the requested crop.
    Matching is case-insensitive and checks if the alert's crop
    starts with the requested crop name (handles "All Standing Crops" etc.)

    If no alerts match, returns an empty list so the frontend can
    display the "No alerts for selected crop" UI.
    """
    crop_lower = crop.strip().lower()
    matched = [
        a for a in alerts
        if a.get("crop", "").lower().startswith(crop_lower)
        or crop_lower == "all"
        or crop_lower in a.get("crop", "").lower()
    ]
    return matched


# ── /alerts endpoint ────────────────────────────────────────────────
@app.get("/alerts", tags=["Alerts"])
async def get_alerts(
    location: str = Query(default="Durgapur", description="City name for weather lookup"),
    crop: Optional[str] = Query(default=None, description="Filter alerts by crop name (e.g. Rice, Wheat, Maize, Cotton, Mustard). Omit for all crops."),
    user: dict = Depends(require_farmer),
):
    """
    Fetch live weather for the given city via Open-Meteo (free, no API key),
    apply crop risk rules, and return filtered alert cards.

    Query params:
      - location (required): city name, e.g. "Durgapur"
      - crop     (optional): crop name filter, e.g. "Rice"
                             If provided, only alerts for that crop are returned.
                             If omitted or empty, all crop alerts are returned.

    Response:
      {
        "location": { "name", "country", "lat", "lon" },
        "weather":  { "temperature_c", "humidity_pct", "rainfall_mm", "wind_kmh", "condition" },
        "fetched_at": "2025-03-26 10:30 UTC",
        "crop_filter": "Rice" | null,
        "alerts": [ { "risk", "crop", "alert", "suggestion", "severity", "icon" }, ... ]
      }
    """
    log.info(f"[ALERTS] location='{location}' crop='{crop}' user_id={user['user_id']}")

    # Step 1: Geocode city → coordinates
    geo = await _geocode_city(location)
    log.info(f"[ALERTS] Geocoded '{location}' → lat={geo['lat']}, lon={geo['lon']}")

    # Step 2: Fetch live weather from Open-Meteo
    weather = await _fetch_weather(geo["lat"], geo["lon"])
    log.info(
        f"[ALERTS] Weather: temp={weather['temp_c']}°C, "
        f"humidity={weather['humidity_pct']}%, rain={weather['rain_mm']}mm, "
        f"wind={weather['wind_kmh']}km/h, condition={weather['condition']}"
    )

    # Step 3: Generate all alerts from rule engine (_ALERT_RULES unchanged)
    all_alerts = _generate_crop_alerts(weather)

    # Step 4: If crop filter provided → filter alerts
    crop_clean = (crop or "").strip()
    if crop_clean and crop_clean.lower() != "all":
        filtered_alerts = _filter_alerts_by_crop(all_alerts, crop_clean)
        log.info(f"[ALERTS] Crop filter='{crop_clean}' → {len(filtered_alerts)}/{len(all_alerts)} alerts returned")
    else:
        filtered_alerts = all_alerts
        crop_clean      = None  # normalise to None when not filtering

    return {
        "location": {
            "name":    geo["name"],
            "country": geo["country"],
            "lat":     geo["lat"],
            "lon":     geo["lon"],
        },
        "weather": {
            "temperature_c": weather["temp_c"],
            "humidity_pct":  weather["humidity_pct"],
            "rainfall_mm":   weather["rain_mm"],
            "wind_kmh":      weather["wind_kmh"],
            "condition":     weather["condition"],
        },
        "fetched_at":  datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "crop_filter": crop_clean,
        "alerts":      filtered_alerts,
    }


# ══════════════════════════════════════════════════════════════════
# OFFICER ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/queries", tags=["Officer"])
def get_queries(user: dict = Depends(require_officer)):
    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT q.query_id, q.question, q.ai_response, q.source,
                   q.status, q.validated_answer, q.created_at,
                   u.name     AS farmer_name,
                   u.email    AS farmer_email,
                   u.location, u.language
            FROM queries q
            JOIN users u ON q.user_id = u.user_id
            ORDER BY q.created_at DESC
        """)
        rows = cursor.fetchall()
        for row in rows:
            row["created_at"] = str(row["created_at"])
        return rows
    finally:
        cursor.close()
        conn.close()


@app.post("/queries/{query_id}/approve", tags=["Officer"])
def approve_query(query_id: int, user: dict = Depends(require_officer)):
    conn   = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE queries SET status='approved' WHERE query_id=%s", (query_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(404, "Query not found.")
        return {"message": "Query approved."}
    finally:
        cursor.close(); conn.close()


@app.post("/queries/{query_id}/reject", tags=["Officer"])
def reject_query(query_id: int, user: dict = Depends(require_officer)):
    conn   = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE queries SET status='rejected' WHERE query_id=%s", (query_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(404, "Query not found.")
        return {"message": "Query rejected."}
    finally:
        cursor.close(); conn.close()


@app.put("/queries/{query_id}/answer", tags=["Officer"])
def edit_answer(query_id: int, data: EditAnswerRequest,
                user: dict = Depends(require_officer)):
    conn   = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE queries SET validated_answer=%s, status='approved' WHERE query_id=%s",
            (data.answer, query_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(404, "Query not found.")
        return {"message": "Answer updated and query approved."}
    finally:
        cursor.close(); conn.close()


@app.get("/queries/{query_id}", tags=["Officer"])
def get_query(query_id: int, user: dict = Depends(require_officer)):
    conn   = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM queries WHERE query_id=%s", (query_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Query not found.")
        row["created_at"] = str(row["created_at"])
        return row
    finally:
        cursor.close(); conn.close()


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "AgriGenius API is running 🌾"}


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true",
    )