from typing import Dict, Any, Tuple
try:
    from server.adapter import OpenEnvAdapter as RestaurantSimulator
    from server.models import (
    Task1Observation, Task1Action, Task1Reward,
    Task2Observation, Task2Action, Task2Reward,
    Task3Observation, Task3Action, Task3Reward
)

except ImportError:
    from adapter import OpenEnvAdapter as RestaurantSimulator
    from models import (
    Task1Observation, Task1Action, Task1Reward,
    Task2Observation, Task2Action, Task2Reward,
    Task3Observation, Task3Action, Task3Reward
)

class RestaurantEnv:
    def __init__(self, task_name: str = "task_1"):
        self.task_name = task_name
        self.simulator = RestaurantSimulator()

    def reset(self) -> Any:
        self.simulator.reset()
        self.start_revenue = self.simulator.revenue
        return self._get_observation()

    def state(self) -> Any:
        return self._get_observation()

    def _get_observation(self):
        s = self.simulator.get_state()
        if self.task_name == "task_1":
            return Task1Observation(
                hour=s["hour"],
                day_of_week=s["day_of_week"],
                queue_size=s["queue_size"],
                staff_count=s["staff_count"],
            )
        elif self.task_name == "task_2":
            return Task2Observation(
                hour=s["hour"],
                dish_popularity=s["dish_popularity"],
                inventory=s["inventory"],
                waste_yesterday=s["waste_yesterday"],
            )
        else:
            return Task3Observation(
                hour=s["hour"],
                day_of_week=s["day_of_week"],
                queue_size=s["queue_size"],
                staff_count=s["staff_count"],
                dish_popularity=s["dish_popularity"],
                inventory=s["inventory"],
                waste_yesterday=s["waste_yesterday"],
                satisfaction=s["satisfaction"],
            )

    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict]:
        deploy_staff = self.simulator.staff_count
        promote_dish = ""
        restock_dishes = {}

        if self.task_name == "task_1":
            deploy_staff = action.deploy_staff
        elif self.task_name == "task_2":
            promote_dish = action.promote_dish
            restock_dishes = action.restock_dishes
        else:
            deploy_staff = action.deploy_staff
            promote_dish = action.promote_dish
            restock_dishes = action.restock_dishes

        prev_queue = self.simulator.queue_size
        prev_revenue = self.simulator.revenue
        prev_satisfaction = self.simulator.satisfaction

        self.simulator.step(deploy_staff, promote_dish, restock_dishes)

        obs = self._get_observation()
        reward = self._calculate_reward(action, prev_queue, prev_revenue, prev_satisfaction)
        done = self.simulator.done

        info = {
            "real_revenue": self.simulator.revenue,
            "satisfaction": self.simulator.satisfaction,
            "waste": self.simulator.waste_yesterday
        }

        return obs, reward, done, info

    def grade(self) -> float:
        obs = self.reset()
        rewards = []
        done = False
        max_steps = 100
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            if self.task_name == "task_1":
                action = Task1Action(deploy_staff=max(1, obs.queue_size // 8 + 1))
            elif self.task_name == "task_2":
                top_dish = max(obs.dish_popularity, key=obs.dish_popularity.get)
                action = Task2Action(promote_dish=top_dish, restock_dishes={})
            else:
                top_dish = max(obs.dish_popularity, key=obs.dish_popularity.get)
                action = Task3Action(
                    deploy_staff=max(1, obs.queue_size // 8 + 1),
                    promote_dish=top_dish,
                    restock_dishes={}
                )
            obs, reward, done, _ = self.step(action)
            rewards.append(reward.score)
        mean = sum(rewards) / len(rewards) if rewards else 0.5
        return max(0.001, min(0.999, mean))

    def _calculate_reward(self, action, prev_queue, prev_revenue, prev_satisfaction):
        if self.task_name == "task_1":
            wait_penalty = min(0.5, self.simulator.queue_size / 80.0)
            optimal_staffing = max(1, min(15, int(self.simulator.queue_size / 8) + 1))
            overstaffing_penalty = min(0.3, max(0, action.deploy_staff - optimal_staffing) * 0.03)
            step_score = max(0.01, min(0.99, 1.0 - wait_penalty - overstaffing_penalty))
            return Task1Reward(
                score=step_score,
                wait_time_penalty=wait_penalty,
                overstaffing_penalty=overstaffing_penalty
            )

        elif self.task_name == "task_2":
            revenue_generated = self.simulator.revenue - prev_revenue
            revenue_reward = min(1.0, revenue_generated / 5000.0)
            waste_penalty = min(0.3, self.simulator.waste_yesterday / 300.0)
            stockout_penalty = 0.1 if self.simulator.satisfaction < prev_satisfaction else 0.0
            step_score = max(0.01, min(0.99, revenue_reward * 0.6 - waste_penalty * 0.3 - stockout_penalty * 0.1 + 0.4))
            return Task2Reward(
                score=step_score,
                revenue_reward=revenue_reward,
                waste_penalty=waste_penalty,
                stockout_penalty=stockout_penalty
            )

        else:
            revenue_generated = self.simulator.revenue - prev_revenue
            revenue_reward = min(1.0, revenue_generated / 5000.0)
            efficiency = max(0.0, 1.0 - (self.simulator.queue_size / 60.0))
            satisfaction_reward = self.simulator.satisfaction
            step_score = max(0.01, min(0.99,
                revenue_reward * 0.35 + efficiency * 0.35 + satisfaction_reward * 0.30
            ))
            return Task3Reward(
                score=step_score,
                revenue_reward=revenue_reward,
                operational_efficiency=efficiency,
                satisfaction_reward=satisfaction_reward
            )
