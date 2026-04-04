import os
import json
import textwrap
from typing import Optional, List, Any
from openai import OpenAI
from environment import RestaurantEnv
from models import Task1Action, Task2Action, Task3Action

# ── Exactly as required by the checklist ──────────────────────────
API_BASE_URL    = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME      = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN        = os.getenv("HF_TOKEN")          # NO default — required by checklist
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional
# ──────────────────────────────────────────────────────────────────

TASK_NAME  = os.getenv("OPENENV_TASK", "task_1")
BENCHMARK  = os.getenv("OPENENV_BENCHMARK", "restaurant_opt")
MAX_STEPS  = 91        # 7 days × 13 hours = full episode
TEMPERATURE = 0.3
MAX_TOKENS  = 300

SYSTEM_PROMPTS = {
    "task_1": (
        "You are an expert restaurant operations manager. "
        "Deploy the right number of staff each hour to handle customer demand. "
        "Staff capacity is 8 customers per staff per hour. "
        "Respond with ONLY a JSON object: {\"deploy_staff\": <integer 1-15>}"
    ),
    "task_2": (
        "You are an expert restaurant inventory manager. "
        "Promote dishes that are popular and restock only low-inventory dishes. "
        "Respond with ONLY a JSON object: "
        "{\"promote_dish\": \"<exact dish name>\", "
        "\"restock_dishes\": {\"<dish name>\": <amount>}}"
    ),
    "task_3": (
        "You are an expert restaurant general manager. "
        "Simultaneously optimize staffing, menu promotions, and inventory. "
        "Respond with ONLY a JSON object: "
        "{\"deploy_staff\": <int 1-15>, "
        "\"promote_dish\": \"<dish name>\", "
        "\"restock_dishes\": {\"<dish name>\": <amount>}}"
    ),
}


# ── Log functions — exact START/STEP/END format ───────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )
# ──────────────────────────────────────────────────────────────────


def parse_action(task_name: str, text: str):
    try:
        start_idx = text.find("{")
        end_idx   = text.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            data = json.loads(text[start_idx:end_idx])
        else:
            return None, "no_json"

        if task_name == "task_1":
            return Task1Action(**data), None
        elif task_name == "task_2":
            return Task2Action(**data), None
        else:
            return Task3Action(**data), None
    except Exception as e:
        return None, "parse_failed"


def fallback_action(task_name: str):
    if task_name == "task_1":
        return Task1Action(deploy_staff=5)
    elif task_name == "task_2":
        return Task2Action(promote_dish="Meals", restock_dishes={})
    else:
        return Task3Action(deploy_staff=5, promote_dish="Meals", restock_dishes={})


def build_user_prompt(step: int, obs: Any, history: List[str], task_name: str) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    if task_name == "task_1":
        action_format = '{"deploy_staff": <integer between 1 and 15>}'
    elif task_name == "task_2":
        action_format = '{"promote_dish": "<dish name>", "restock_dishes": {"<dish name>": <amount>}}'
    else:
        action_format = '{"deploy_staff": <int>, "promote_dish": "<dish name>", "restock_dishes": {"<dish name>": <amount>}}'

    return textwrap.dedent(f"""
        Step: {step}
        Current observation:
        {obs.model_dump_json(indent=2)}

        Recent history:
        {history_block}

        Respond with exactly ONE valid JSON object:
        {action_format}
    """).strip()


def run_inference():
    # ── OpenAI client using the required variables ─────────────────
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    # ──────────────────────────────────────────────────────────────

    env = RestaurantEnv(task_name=TASK_NAME)
    system_prompt = SYSTEM_PROMPTS.get(TASK_NAME, SYSTEM_PROMPTS["task_3"])

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    obs = env.reset()

    history      = []
    rewards_list = []
    done         = False
    step         = 0

    while not done and step < MAX_STEPS:
        step += 1
        user_prompt = build_user_prompt(step, obs, history, TASK_NAME)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as e:
            log_step(step, "api_error", 0.0, True, "api_error")
            break

        action, parse_error = parse_action(TASK_NAME, text)

        if action is None:
            action = fallback_action(TASK_NAME)

        obs, reward_obj, done, info = env.step(action)
        r_val = reward_obj.score
        rewards_list.append(r_val)

        # Clean action string for log — no quotes that break parsing
        action_str = json.dumps(action.model_dump(), separators=(',', ':'))
        log_step(step, action_str, r_val, done, parse_error)

        history.append(f"Step {step}: {action_str} -> Score={r_val:.2f}")

    final_score = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    success = final_score > 0.5
    log_end(success, step, rewards_list)


if __name__ == "__main__":
    run_inference()
