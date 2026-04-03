import os
import json
import textwrap
from typing import Optional, List, Any
from openai import OpenAI
from environment import RestaurantEnv
from models import Task1Action, Task2Action, Task3Action

IMAGE_NAME = os.getenv("IMAGE_NAME", "restaurant_env")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("OPENENV_TASK", "task_1")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "restaurant_opt")
MAX_STEPS = 91   # ← FIXED: 7 days × 13 hours = full episode
TEMPERATURE = 0.3
MAX_TOKENS = 300

SYSTEM_PROMPTS = {
    "task_1": """You are an expert restaurant operations manager.
Your job is to deploy the right number of staff each hour to handle customer demand.
- Queue size tells you how many customers are waiting
- Staff capacity is 8 customers per staff per hour
- Understaffing causes long waits and unhappy customers
- Overstaffing wastes money
- Respond with ONLY a JSON object: {"deploy_staff": <integer between 1 and 15>}""",

    "task_2": """You are an expert restaurant inventory manager.
Your job is to promote the right dishes and restock inventory to maximize revenue and minimize waste.
- Promote dishes that are popular and likely to sell well at this hour
- Restock dishes that are running low before they cause stockouts
- Do not restock dishes that already have high inventory (causes waste)
- Respond with ONLY a JSON object: {"promote_dish": "<exact dish name>", "restock_dishes": {"<dish name>": <amount>}}""",

    "task_3": """You are an expert restaurant general manager.
Your job is to simultaneously optimize staffing, menu promotions, and inventory.
- Balance staff deployment with current queue size (8 customers per staff per hour)
- Promote dishes that match the current time (lunch vs dinner preferences)
- Restock only low-inventory popular dishes to avoid waste
- Respond with ONLY a JSON object: {"deploy_staff": <int 1-15>, "promote_dish": "<dish name>", "restock_dishes": {"<dish name>": <amount>}}"""
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def parse_action(task_name: str, text: str):
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            data = json.loads(json_str)
        else:
            return None, "No JSON found in response"

        if task_name == "task_1":
            return Task1Action(**data), None
        elif task_name == "task_2":
            return Task2Action(**data), None
        else:
            return Task3Action(**data), None
    except Exception as e:
        return None, str(e)

def build_user_prompt(step: int, last_obs: Any, history: List[str], task_name: str) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Current observation:
        {last_obs.model_dump_json(indent=2)}

        Recent history:
        {history_block}

        Respond with exactly ONE valid JSON action now.
    """).strip()

def run_inference():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = RestaurantEnv(task_name=TASK_NAME)
    system_prompt = SYSTEM_PROMPTS.get(TASK_NAME, SYSTEM_PROMPTS["task_3"])

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    obs = env.reset()

    history = []
    rewards_list = []
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        user_prompt = build_user_prompt(step, obs, history, TASK_NAME)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as e:
            log_step(step, "api_error", 0.0, True, str(e))
            break

        action, parse_error = parse_action(TASK_NAME, text)

        if parse_error or action is None:
            if TASK_NAME == "task_1":
                action = Task1Action(deploy_staff=5)
            elif TASK_NAME == "task_2":
                action = Task2Action(promote_dish="Meals", restock_dishes={})
            else:
                action = Task3Action(deploy_staff=5, promote_dish="Meals", restock_dishes={})

        obs, reward_obj, done, info = env.step(action)
        r_val = reward_obj.score
        rewards_list.append(r_val)

        log_step(step, action.model_dump_json().replace('"', "'"), r_val, done,
                 f"Parse error: {parse_error}" if parse_error else None)
        history.append(f"Step {step}: Action={action.model_dump_json()} Score={r_val:.2f}")

    final_score = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    success = final_score > 0.5
    log_end(success, step, final_score, rewards_list)

if __name__ == "__main__":
    run_inference()
