from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from environment import RestaurantEnv
from models import Task1Action, Task2Action, Task3Action

envs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    for task in ["task_1", "task_2", "task_3"]:
        envs[task] = RestaurantEnv(task_name=task)
    yield

app = FastAPI(title="Restaurant Optimization OpenEnv", version="0.1.0", lifespan=lifespan)

# ── 1. health ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    # validator expects exactly: {"status": "healthy"}
    return {"status": "healthy"}

# ── 2. openenv.yaml ───────────────────────────────────────────────
@app.get("/openenv.yaml")
def serve_openenv_yaml():
    with open("openenv.yaml", "r") as f:
        content = f.read()
    return PlainTextResponse(content=content, media_type="application/x-yaml")

# ── 3. metadata ───────────────────────────────────────────────────
@app.get("/metadata")
def metadata():
    return {
        "name": "restaurant-openenv-final",
        "description": "A restaurant management OpenEnv simulating Andhra/Telangana cuisine.",
        "version": "0.1.0",
        "tasks": ["task_1", "task_2", "task_3"]
    }

# ── 4. schema ─────────────────────────────────────────────────────
@app.get("/schema")
def schema(task: str = Query(default="task_3")):
    if task == "task_1":
        return {
            "action": {
                "type": "object",
                "properties": {
                    "deploy_staff": {"type": "integer", "minimum": 1, "maximum": 15}
                },
                "required": ["deploy_staff"]
            },
            "observation": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "queue_size": {"type": "integer"},
                    "staff_count": {"type": "integer"}
                }
            },
            "state": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "queue_size": {"type": "integer"},
                    "staff_count": {"type": "integer"}
                }
            }
        }
    elif task == "task_2":
        return {
            "action": {
                "type": "object",
                "properties": {
                    "promote_dish": {"type": "string"},
                    "restock_dishes": {"type": "object"}
                },
                "required": ["promote_dish", "restock_dishes"]
            },
            "observation": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "dish_popularity": {"type": "object"},
                    "inventory": {"type": "object"},
                    "waste_yesterday": {"type": "number"}
                }
            },
            "state": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "dish_popularity": {"type": "object"},
                    "inventory": {"type": "object"},
                    "waste_yesterday": {"type": "number"}
                }
            }
        }
    else:
        return {
            "action": {
                "type": "object",
                "properties": {
                    "deploy_staff": {"type": "integer"},
                    "promote_dish": {"type": "string"},
                    "restock_dishes": {"type": "object"}
                },
                "required": ["deploy_staff", "promote_dish", "restock_dishes"]
            },
            "observation": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "queue_size": {"type": "integer"},
                    "staff_count": {"type": "integer"},
                    "dish_popularity": {"type": "object"},
                    "inventory": {"type": "object"},
                    "waste_yesterday": {"type": "number"},
                    "satisfaction": {"type": "number"}
                }
            },
            "state": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "queue_size": {"type": "integer"},
                    "staff_count": {"type": "integer"},
                    "dish_popularity": {"type": "object"},
                    "inventory": {"type": "object"},
                    "waste_yesterday": {"type": "number"},
                    "satisfaction": {"type": "number"}
                }
            }
        }

# ── 5. mcp ────────────────────────────────────────────────────────
@app.post("/mcp")
def mcp(request: Dict[str, Any] = {}):
    method = request.get("method", "")
    req_id = request.get("id", 1)
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "restaurant-openenv-final",
            "description": "Restaurant management RL environment",
            "methods": ["reset", "step", "state", "grade"]
        }
    }

# ── 6. root ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "environment": "restaurant-optimization",
            "tasks": ["task_1", "task_2", "task_3"]}

# ── 7. reset ──────────────────────────────────────────────────────
@app.post("/reset")
def reset_env(task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].reset().model_dump()

# ── 8. state ──────────────────────────────────────────────────────
@app.get("/state")
def get_state(task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].state().model_dump()

# ── 9. step ───────────────────────────────────────────────────────
@app.post("/step")
async def step(request: Dict[str, Any], task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        if task == "task_1":
            action = Task1Action(**request)
        elif task == "task_2":
            action = Task2Action(**request)
        else:
            action = Task3Action(**request)
        obs, reward, done, info = envs[task].step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ── 10. grade ─────────────────────────────────────────────────────
@app.api_route("/grade", methods=["GET", "POST"])
def grade(task: str = Query(default="task_3")):
    if task not in ["task_1", "task_2", "task_3"]:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        env = RestaurantEnv(task_name=task)
        score = env.grade()
        return {"score": float(score)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
