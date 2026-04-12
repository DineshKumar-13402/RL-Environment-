from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
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

@app.get("/")
def root():
    return {"status": "ok", "environment": "restaurant-optimization", "tasks": ["task_1", "task_2", "task_3"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "restaurant-openenv-final",
        "description": "A restaurant management OpenEnv simulating Andhra/Telangana cuisine.",
        "version": "1.0.0",
        "tasks": ["task_1", "task_2", "task_3"]
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "deploy_staff": {"type": "integer"},
                "promote_dish": {"type": "string"},
                "restock_dishes": {"type": "object"}
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer"},
                "queue_size": {"type": "integer"},
                "staff_count": {"type": "integer"},
                "revenue": {"type": "number"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "done": {"type": "boolean"},
                "step": {"type": "integer"}
            }
        }
    }

@app.post("/mcp")
def mcp(request: dict = {}):
    return {"jsonrpc": "2.0", "id": None, "result": {"tools": []}}

@app.post("/reset")
def reset_env(task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].reset().model_dump()

@app.get("/state")
def get_state(task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return envs[task].state().model_dump()

@app.post("/step")
def step_env(task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    return {"status": "ok"}

@app.api_route("/grade", methods=["GET", "POST"])
def grade(task: str = Query(default="task_3")):
    if task not in ["task_1", "task_2", "task_3"]:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        from graders import Task1Grader, Task2Grader, Task3Grader
        grader_map = {
            "task_1": Task1Grader(),
            "task_2": Task2Grader(),
            "task_3": Task3Grader(),
        }
        score = grader_map[task].grade()
        return {"score": float(score)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
