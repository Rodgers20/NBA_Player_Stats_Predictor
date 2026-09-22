"""
FastAPI backend — exposes all existing Python logic as REST endpoints.
The Next.js frontend calls these; the Dash app can still run in parallel.

Run:
  uvicorn api.main:app --reload --port 8000
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import props, games, players

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm caches on startup (same warmup as the Dash app)
    import threading
    def _warm():
        try:
            from utils.props_cache import refresh_props_cache
            from utils.kaggle_loader import load_engineered_data, load_player_positions, load_defense_vs_position
            DF = load_engineered_data()
            POS = load_player_positions()
            DEF = load_defense_vs_position()
            PLAYERS = DF["PLAYER_NAME"].unique().tolist() if not DF.empty else []
            refresh_props_cache(DF, POS, DEF, PLAYERS)
        except Exception as e:
            print(f"[API] Cache warmup failed: {e}")
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="NBA Props API",
    description="REST API for the NBA Player Props predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(props.router,   prefix="/api")
app.include_router(games.router,   prefix="/api")
app.include_router(players.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
