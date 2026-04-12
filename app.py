from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
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

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/openenv.yaml")
def serve_openenv_yaml():
    with open("openenv.yaml", "r") as f:
        content = f.read()
    return PlainTextResponse(content=content, media_type="application/x-yaml")

@app.get("/metadata")
def metadata():
    return {
        "name": "restaurant-openenv-final",
        "description": "A restaurant management OpenEnv simulating Andhra/Telangana cuisine.",
        "version": "0.1.0",
        "tasks": ["task_1", "task_2", "task_3"]
    }

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_1",
                "name": "Staff Scheduling Optimization",
                "description": "Task 1 (Easy): Staff Scheduling Optimization.",
                "grader": "graders:Task1Grader"
            },
            {
                "id": "task_2",
                "name": "Menu and Inventory Optimization",
                "description": "Task 2 (Medium): Menu and Inventory Optimization.",
                "grader": "graders:Task2Grader"
            },
            {
                "id": "task_3",
                "name": "End-to-End Restaurant Optimization",
                "description": "Task 3 (Hard): End-to-End Restaurant Optimization.",
                "grader": "graders:Task3Grader"
            }
        ]
    }

@app.get("/schema")
def schema(task: str = Query(default="task_3")):
    if task == "task_1":
        return {
            "action": {"type": "object", "properties": {"deploy_staff": {"type": "integer", "minimum": 1, "maximum": 15}}, "required": ["deploy_staff"]},
            "observation": {"type": "object", "properties": {"hour": {"type": "integer"}, "day_of_week": {"type": "integer"}, "queue_size": {"type": "integer"}, "staff_count": {"type": "integer"}}},
            "state": {"type": "object", "properties": {"hour": {"type": "integer"}, "day_of_week": {"type": "integer"}, "queue_size": {"type": "integer"}, "staff_count": {"type": "integer"}}}
        }
    elif task == "task_2":
        return {
            "action": {"type": "object", "properties": {"promote_dish": {"type": "string"}, "restock_dishes": {"type": "object"}}, "required": ["promote_dish", "restock_dishes"]},
            "observation": {"type": "object", "properties": {"hour": {"type": "integer"}, "dish_popularity": {"type": "object"}, "inventory": {"type": "object"}, "waste_yesterday": {"type": "number"}}},
            "state": {"type": "object", "properties": {"hour": {"type": "integer"}, "dish_popularity": {"type": "object"}, "inventory": {"type": "object"}, "waste_yesterday": {"type": "number"}}}
        }
    else:
        return {
            "action": {"type": "object", "properties": {"deploy_staff": {"type": "integer"}, "promote_dish": {"type": "string"}, "restock_dishes": {"type": "object"}}, "required": ["deploy_staff", "promote_dish", "restock_dishes"]},
            "observation": {"type": "object", "properties": {"hour": {"type": "integer"}, "day_of_week": {"type": "integer"}, "queue_size": {"type": "integer"}, "staff_count": {"type": "integer"}, "dish_popularity": {"type": "object"}, "inventory": {"type": "object"}, "waste_yesterday": {"type": "number"}, "satisfaction": {"type": "number"}}},
            "state": {"type": "object", "properties": {"hour": {"type": "integer"}, "day_of_week": {"type": "integer"}, "queue_size": {"type": "integer"}, "staff_count": {"type": "integer"}, "dish_popularity": {"type": "object"}, "inventory": {"type": "object"}, "waste_yesterday": {"type": "number"}, "satisfaction": {"type": "number"}}}
        }

@app.post("/mcp")
async def mcp(request: Request):
    try:
        body = await request.json()
    except:
        body = {}
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "restaurant-openenv-final",
            "description": "Restaurant management RL environment",
            "methods": ["reset", "step", "state", "grade"]
        }
    }

@app.get("/")
def root():
    return {"status": "ok", "environment": "restaurant-optimization",
            "tasks": ["task_1", "task_2", "task_3"]}

@app.post("/reset")
async def reset_env(request: Request, task: str = Query(default="task_3")):
    try:
        body = await request.json()
        task = body.get("task", task)
    except:
        pass
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].reset().model_dump()

@app.get("/state")
async def get_state(request: Request, task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].state().model_dump()

@app.post("/step")
async def step(request: Request, task: str = Query(default="task_3")):
    try:
        body = await request.json()
        task = body.pop("task", task)
    except:
        body = {}
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        if task == "task_1":
            action = Task1Action(**body)
        elif task == "task_2":
            action = Task2Action(**body)
        else:
            action = Task3Action(**body)
        obs, reward, done, info = envs[task].step(action)
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.api_route("/grade", methods=["GET", "POST"])
async def grade(request: Request, task: str = Query(default=None)):
    # Accept task from query param OR json body
    if task is None:
        try:
            body = await request.json()
            task = body.get("task", "task_3")
        except:
            task = "task_3"
    if task not in ["task_1", "task_2", "task_3"]:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        env = RestaurantEnv(task_name=task)
        score = env.grade()
        return {"task": task, "score": float(score)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Per-task grade routes
@app.api_route("/grade/{task_id}", methods=["GET", "POST"])
def grade_by_path(task_id: str):
    if task_id not in ["task_1", "task_2", "task_3"]:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task_id}"})
    try:
        env = RestaurantEnv(task_name=task_id)
        score = env.grade()
        return {"task": task_id, "score": float(score)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
