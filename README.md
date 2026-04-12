---
title: Restaurant Opt
emoji: 🍛
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - restaurant
  - rl-environment
---

# 🍛 Restaurant Optimization — OpenEnv RL Environment

**Team TensorForge** | Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Space-yellow)](https://huggingface.co/spaces/Dinesh2314/restaurant-openenv-final)
[![GitHub](https://img.shields.io/badge/GitHub-Source-black)](https://github.com/DineshKumar-13402/RL-Environment-)

---

## 🎯 Motivation & Real-World Problem

Most small restaurant owners in India operate on instinct — staffing too many people during slow hours, running out of popular dishes during dinner rush, over-purchasing perishables that go to waste. These decisions cost them thousands of rupees daily.

This environment simulates a real Andhra/Telangana-style restaurant with authentic menu items. An RL agent learns to make better operational decisions than a human owner can intuitively — optimizing staffing, inventory, and promotions simultaneously across a full week of operations.

**The environment fills a real gap**: existing RL restaurant work focuses on food delivery logistics (routing, couriers). This is the first OpenEnv environment focused on in-restaurant operational optimization from the owner's perspective.

---

## 🍽️ Restaurant Domain

Based on a real South Indian restaurant menu (Andhra/Telangana cuisine):

| Category | Dishes |
|----------|--------|
| Non-Veg Starters | Chicken 65, Chilli Chicken, Chicken Lollipop, Chicken Wings, Chicken 555 |
| Veg Starters | Chilli Paneer, Paneer 65, Paneer Manchuria, Chilli Mushroom, Mushroom 65, Mushroom Manchuria |
| Non-Veg Curries | Chicken Curry, Chicken Butter Masala, Kadai Chicken |
| Veg Curries | Paneer Butter Masala, Kaju Tomato Curry, Mushroom Curry, Kadai Paneer |
| Staples | Meals (₹180), Rumali Roti, Pulka |

**Simulation parameters:**
- 150–200 customers per day (230–260 on weekends)
- Lunch peak: 12:00 PM – 2:00 PM
- Dinner peak: 7:00 PM – 11:30 PM
- 40% repeat customers, 60% new customers
- Episode length: 7 days × 13 hours = 91 steps

---

## 📋 Three Tasks (Easy → Medium → Hard)

### Task 1 — Staff Scheduling (Easy)
**Goal:** Deploy the right number of staff each hour to handle customer demand without overstaffing.

| | |
|--|--|
| **Observation** | `hour`, `day_of_week`, `queue_size`, `staff_count` |
| **Action** | `deploy_staff` (integer 1–15) |
| **Reward** | Penalizes understaffing (long queues) and overstaffing (wasted cost) |
| **Done** | After 7 simulated days (91 steps) |

### Task 2 — Menu & Inventory Optimization (Medium)
**Goal:** Promote dishes strategically and restock inventory to maximize revenue while minimizing food waste.

| | |
|--|--|
| **Observation** | `hour`, `dish_popularity`, `inventory`, `waste_yesterday` |
| **Action** | `promote_dish` (dish name), `restock_dishes` (dict of dish → amount) |
| **Reward** | Revenue generated minus waste penalty minus stockout penalty |
| **Done** | After 7 simulated days (91 steps) |

### Task 3 — End-to-End Optimization (Hard)
**Goal:** Simultaneously optimize staffing, menu promotions, and inventory to maximize overall restaurant performance.

| | |
|--|--|
| **Observation** | All of Task 1 + Task 2 observations + `satisfaction` score |
| **Action** | `deploy_staff` + `promote_dish` + `restock_dishes` combined |
| **Reward** | Weighted combination of revenue (35%), efficiency (35%), satisfaction (30%) |
| **Done** | After 7 simulated days (91 steps) |

---

## 🎁 Reward Function

All scores are normalized to [0.0, 1.0].

**Task 1:**
wait_penalty    = min(0.5, queue_size / 80.0)
overstaff_penalty = min(0.3, max(0, staff - optimal) × 0.03)
score = 1.0 - wait_penalty - overstaff_penalty

**Task 2:**
revenue_reward  = min(1.0, revenue_this_step / 5000.0)
waste_penalty   = min(0.3, waste_yesterday / 300.0)
stockout_penalty = 0.1 if satisfaction dropped else 0.0
score = revenue_reward × 0.6 - waste_penalty × 0.3 - stockout_penalty × 0.1 + 0.4

**Task 3:**
score = revenue_reward × 0.35 + efficiency × 0.35 + satisfaction × 0.30

---

## 🏗️ Project Structure
├── environment.py          # Core OpenEnv interface (step/reset/state)
├── adapter.py              # Bridges RestaurantSimulator to OpenEnv API
├── models.py               # Pydantic models for Observation/Action/Reward
├── restaurant_simulator.py # Synthetic data generator (150-200 customers/day)
├── inference.py            # Baseline inference script (OpenAI client)
├── app.py                  # FastAPI server exposing /reset /step /state
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container definition for HF Spaces
└── pyproject.toml          # Dependencies

---

## 🔌 API Endpoints

The environment is served as a REST API on port 7860:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — returns `{"status": "ok"}` |
| `/reset?task=task_1` | POST | Reset environment, returns initial observation |
| `/state?task=task_1` | GET | Get current state without stepping |
| `/step?task=task_1` | POST | Send action, get observation + reward + done |

**Example `/step` request body for Task 1:**
```json
{"deploy_staff": 7}
```

**Example `/step` request body for Task 3:**
```json
{
  "deploy_staff": 8,
  "promote_dish": "Chicken Butter Masala",
  "restock_dishes": {"Chicken 65": 10, "Meals": 15}
}
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Docker
- Hugging Face CLI

### Install dependencies
```bash
pip install openenv-core pydantic openai fastapi uvicorn pandas numpy
```

### Run locally
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run inference baseline
```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENENV_TASK="task_1"
python inference.py
```

### Validate OpenEnv spec
```bash
openenv validate
```

### Docker
```bash
docker build -t restaurant-env .
docker run -p 7860:7860 restaurant-env
```

---

## 📊 Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Score | Notes |
|------|-------|-------|
| Task 1 — Staff Scheduling | 0.87 | Grader verified score |
| Task 2 — Menu & Inventory | 0.8176 | Grader verified score |
| Task 3 — End-to-End | 0.7036 | Grader verified score |



---

## 👥 Team

**TensorForge**
- Dinesh Kumar Dasari
- Afthab Hussain Shaik
- V. Deepak
# Restaurant OpenEnv
