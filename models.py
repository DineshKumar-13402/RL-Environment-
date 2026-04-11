from pydantic import BaseModel, Field
from typing import Dict, Optional

# --- Task 1 Models ---
class Task1Observation(BaseModel):
    hour: int = Field(..., description="Current hour of the day (8 to 23)")
    day_of_week: int = Field(..., description="Day of the week (0 to 6)")
    queue_size: int = Field(..., description="Current number of customers waiting")
    staff_count: int = Field(..., description="Current number of staff on the floor")

class Task1Action(BaseModel):
    deploy_staff: int = Field(..., description="Number of staff to deploy for the next hour (1 to 15)")

class Task1Reward(BaseModel):
    score: float = Field(..., description="Final 0.0 - 1.0 grader score")
    wait_time_penalty: float = Field(..., description="Penalty for having too many customers in queue")
    overstaffing_penalty: float = Field(..., description="Penalty for deploying more staff than necessary")


# --- Task 2 Models ---
class Task2Observation(BaseModel):
    hour: int = Field(..., description="Current hour of the day (8 to 23)")
    dish_popularity: Dict[str, float] = Field(..., description="Popularity of each dish (0.0 to 1.0)")
    inventory: Dict[str, int] = Field(..., description="Current stock of each dish")
    waste_yesterday: float = Field(..., description="Total waste from the previous day")

class Task2Action(BaseModel):
    promote_dish: str = Field(..., description="Name of the dish to promote (increases popularity)")
    restock_dishes: Dict[str, int] = Field(..., description="Dictionary of dishes to restock")

class Task2Reward(BaseModel):
    score: float = Field(..., description="Final 0.0 - 1.0 grader score")
    revenue_reward: float = Field(..., description="Reward for generating revenue")
    waste_penalty: float = Field(..., description="Penalty for wasting inventory")
    stockout_penalty: float = Field(..., description="Penalty for running out of popular dishes")


# --- Task 3 Models ---
class Task3Observation(BaseModel):
    hour: int = Field(..., description="Current hour of the day")
    day_of_week: int = Field(..., description="Day of the week")
    queue_size: int = Field(..., description="Current queue")
    staff_count: int = Field(..., description="Current staff")
    dish_popularity: Dict[str, float] = Field(..., description="Dish popularity")
    inventory: Dict[str, int] = Field(..., description="Dish inventory")
    waste_yesterday: float = Field(..., description="Waste from yesterday")
    satisfaction: float = Field(..., description="Overall customer satisfaction")

class Task3Action(BaseModel):
    deploy_staff: int = Field(..., description="Number of staff to deploy")
    promote_dish: str = Field(..., description="Dish to promote")
    restock_dishes: Dict[str, int] = Field(..., description="Dishes to restock and amounts")

class Task3Reward(BaseModel):
    score: float = Field(..., description="Final 0.0 - 1.0 grader score")
    revenue_reward: float = Field(..., description="Reward correlated with revenue")
    operational_efficiency: float = Field(..., description="Efficiency derived from wait times and staffing")
    satisfaction_reward: float = Field(..., description="Reward for maintaining high satisfaction")
