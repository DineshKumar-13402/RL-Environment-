import sys
import os
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# Ensure the parent directory is in sys.path for local module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import RestaurantEnv
from models import Task1Action, Task2Action, Task3Action

envs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    for task in ["task_1", "task_2", "task_3"]:
        envs[task] = RestaurantEnv(task_name=task)
    yield

app = FastAPI(title="Restaurant Optimization OpenEnv", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "environment": "restaurant-optimization",
            "tasks": ["task_1", "task_2", "task_3"]}

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
def step_env(action: dict, task: str = Query(default="task_3")):
    if task not in envs:
        return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})
    try:
        if task == "task_1":
            act = Task1Action(**action)
        elif task == "task_2":
            act = Task2Action(**action)
        else:
            act = Task3Action(**action)
        obs, reward, done, info = envs[task].step(act)
        return {"observation": obs.model_dump(), "reward": reward.model_dump(),
                "done": done, "info": info}
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
