import random
from datetime import datetime, timedelta
from typing import Dict, Any

from restaurant_simulator import RestaurantSimulator, MENU, BASE_INVENTORY

class OpenEnvAdapter:
    def __init__(self):
        self.sim = RestaurantSimulator()
        self.reset()

    def reset(self):
        self.hour = 11
        self.date = datetime(2026, 4, 7)

        is_weekend = self.date.weekday() >= 5
        multiplier = 1.3 if is_weekend else 1.0
        n_customers = int(random.randint(self.sim.min_customers, self.sim.max_customers) * multiplier)
        self.arrivals = self.sim._generate_arrival_times(n_customers, self.date)

        self.queue_size = 0
        self.staff_count = 5
        self.inventory = {self.sim.dishes[did].name: BASE_INVENTORY[did] for did in self.sim.dishes}
        self.dish_ids = {self.sim.dishes[did].name: did for did in self.sim.dishes}
        self.dish_popularity = {self.sim.dishes[did].name: self.sim.dishes[did].popularity for did in self.sim.dishes}

        self.waste_yesterday = 10.0
        self.revenue = 0.0
        self.satisfaction = 1.0
        self.done = False

    def get_state(self) -> Dict[str, Any]:
        return {
            "hour": self.hour,
            "day_of_week": self.date.weekday(),
            "queue_size": self.queue_size,
            "staff_count": self.staff_count,
            "inventory": self.inventory,
            "dish_popularity": self.dish_popularity,   # ← FIXED key name
            "waste_yesterday": self.waste_yesterday,
            "revenue": self.revenue,
            "satisfaction": self.satisfaction
        }

    def step(self, staff_deployed: int, promote_dish: str, restock: Dict[str, int]) -> Dict[str, Any]:
        if self.done:
            return self.get_state()

        self.staff_count = staff_deployed

        for dish_name, amount in restock.items():
            if dish_name in self.inventory:
                self.inventory[dish_name] += amount

        if promote_dish in self.dish_popularity:
            self.dish_popularity[promote_dish] = min(1.0, self.dish_popularity[promote_dish] + 0.15)
            dish_id = self.dish_ids[promote_dish]
            self.sim.dishes[dish_id].popularity = self.dish_popularity[promote_dish]

        arrivals_this_hour = [a for a in self.arrivals if a.hour == self.hour]
        self.queue_size += len(arrivals_this_hour)

        capacity = self.staff_count * 8
        processed = min(self.queue_size, capacity)
        self.queue_size -= processed

        revenue_this_step = 0
        satisfaction_delta = 0.0

        for _ in range(processed):
            num_items = random.randint(1, 4)
            available = [name for name, qty in self.inventory.items() if qty > 0]

            if not available:
                satisfaction_delta -= 0.05
                continue

            weights = [self.dish_popularity[name] for name in available]
            items_ordered = random.choices(available, weights=weights, k=num_items)

            for item in items_ordered:
                if self.inventory[item] > 0:
                    self.inventory[item] -= 1
                    dish_id = self.dish_ids[item]
                    revenue_this_step += self.sim.dishes[dish_id].price
                else:
                    satisfaction_delta -= 0.02

        self.revenue += revenue_this_step

        wait_penalty = min(0.3, self.queue_size / 100.0)
        self.satisfaction = max(0.0, min(1.0, self.satisfaction + satisfaction_delta - wait_penalty + 0.01))

        self.hour += 1
        if self.hour > 23:
            self.hour = 11
            self.date += timedelta(days=1)

            # Calculate waste from perishables
            perishable = ["Chicken 65", "Chilli Chicken", "Chicken Butter Masala",
                          "Paneer Butter Masala", "Meals", "Kadai Chicken",
                          "Paneer 65", "Chilli Paneer", "Chicken Curry"]
            waste = 0
            for p in perishable:
                if p in self.inventory:
                    waste += self.inventory[p]
            self.waste_yesterday = waste

            # FULL inventory reset each new day ← FIXED
            self.inventory = {
                self.sim.dishes[did].name: BASE_INVENTORY[did]
                for did in self.sim.dishes
            }

            is_weekend = self.date.weekday() >= 5
            multiplier = 1.3 if is_weekend else 1.0
            n_customers = int(random.randint(self.sim.min_customers, self.sim.max_customers) * multiplier)
            self.arrivals = self.sim._generate_arrival_times(n_customers, self.date)

        if (self.date - datetime(2026, 4, 7)).days >= 7:
            self.done = True

        return self.get_state()
