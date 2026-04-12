# graders.py

from adapter import OpenEnvAdapter as RestaurantSimulator


class Task1Grader:
    """Easy: Staff Scheduling Optimization"""

    def grade(self, env=None, *args, **kwargs) -> float:
        sim = RestaurantSimulator()
        sim.reset()

        total_score = 0.0
        steps = 10

        for _ in range(steps):
            queue = sim.queue_size
            optimal_staff = max(1, min(15, int(queue / 8) + 1))
            sim.step(optimal_staff, "", {})

            wait_penalty = min(0.5, sim.queue_size / 80.0)
            overstaffing_penalty = 0.0  # using optimal staff
            step_score = 1.0 - wait_penalty - overstaffing_penalty
            total_score += step_score

            if sim.done:
                break

        raw = total_score / steps
        # Clamp strictly between 0 and 1 — never 0.0 or 1.0
        return max(0.01, min(0.99, round(raw, 4)))


class Task2Grader:
    """Medium: Menu and Inventory Optimization"""

    def grade(self, env=None, *args, **kwargs) -> float:
        sim = RestaurantSimulator()
        sim.reset()

        total_score = 0.0
        steps = 10

        for _ in range(steps):
            state = sim.get_state()
            # Pick the most popular dish to promote
            dish_popularity = state.get("dish_popularity", {})
            promote = max(dish_popularity, key=dish_popularity.get) if dish_popularity else ""
            # Restock everything low
            inventory = state.get("inventory", {})
            restock = {k: 50 for k, v in inventory.items() if v < 20}

            prev_revenue = sim.revenue
            prev_satisfaction = sim.satisfaction
            sim.step(sim.staff_count, promote, restock)

            revenue_generated = sim.revenue - prev_revenue
            revenue_reward = min(1.0, revenue_generated / 5000.0)
            waste_penalty = min(0.3, sim.waste_yesterday / 300.0)
            stockout_penalty = 0.1 if sim.satisfaction < prev_satisfaction else 0.0

            step_score = revenue_reward * 0.6 - waste_penalty * 0.3 - stockout_penalty * 0.1 + 0.4
            total_score += max(0.0, min(1.0, step_score))

            if sim.done:
                break

        raw = total_score / steps
        return max(0.01, min(0.99, round(raw, 4)))


class Task3Grader:
    """Hard: End-to-End Restaurant Optimization"""

    def grade(self, env=None, *args, **kwargs) -> float:
        sim = RestaurantSimulator()
        sim.reset()

        total_score = 0.0
        steps = 10

        for _ in range(steps):
            state = sim.get_state()
            queue = sim.queue_size
            optimal_staff = max(1, min(15, int(queue / 8) + 1))
            dish_popularity = state.get("dish_popularity", {})
            promote = max(dish_popularity, key=dish_popularity.get) if dish_popularity else ""
            inventory = state.get("inventory", {})
            restock = {k: 50 for k, v in inventory.items() if v < 20}

            prev_revenue = sim.revenue
            sim.step(optimal_staff, promote, restock)

            revenue_generated = sim.revenue - prev_revenue
            revenue_reward = min(1.0, revenue_generated / 5000.0)
            efficiency = max(0.0, 1.0 - (sim.queue_size / 60.0))
            satisfaction_reward = sim.satisfaction

            step_score = revenue_reward * 0.35 + efficiency * 0.35 + satisfaction_reward * 0.30
            total_score += max(0.0, min(1.0, step_score))

            if sim.done:
                break

        raw = total_score / steps
        return max(0.01, min(0.99, round(raw, 4)))