"""
Restaurant Synthetic Data Generator
=====================================
Team: TensorForge
Hackathon: Meta x PyTorch x Hugging Face OpenEnv Hackathon 2026

This module generates realistic synthetic restaurant data based on
M. Venkatesh restaurant menu (Andhra/Telangana style).

It simulates:
- Customer arrivals (lunch + dinner peaks)
- Orders per customer
- Staff utilization
- Inventory tracking
- Customer satisfaction scores
- Repeat customer behaviour
- Weekly patterns (weekday vs weekend)

Usage:
    from restaurant_simulator import RestaurantSimulator
    sim = RestaurantSimulator()
    week_data = sim.generate_week()
    sim.save_to_csv(week_data)
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import os

# ─────────────────────────────────────────────
# MENU DEFINITION
# ─────────────────────────────────────────────

MENU = {
    # id: (name, price, prep_time_minutes, category, popularity)
    1:  ("Chicken 65",            280, 12, "non_veg_starter", 0.85),
    2:  ("Chilli Chicken",        300, 15, "non_veg_starter", 0.80),
    3:  ("Chicken Lollipop",      320, 18, "non_veg_starter", 0.75),
    4:  ("Chicken Wings",         310, 15, "non_veg_starter", 0.70),
    5:  ("Chicken 555",           280, 12, "non_veg_starter", 0.65),
    6:  ("Chilli Paneer",         220, 10, "veg_starter",     0.78),
    7:  ("Paneer 65",             240, 12, "veg_starter",     0.72),
    8:  ("Paneer Manchuria",      220, 10, "veg_starter",     0.68),
    9:  ("Chilli Mushroom",       220, 10, "veg_starter",     0.60),
    10: ("Mushroom 65",           240, 12, "veg_starter",     0.58),
    11: ("Mushroom Manchuria",    220, 10, "veg_starter",     0.55),
    12: ("Chicken Curry",         240, 20, "non_veg_curry",   0.88),
    13: ("Chicken Butter Masala", 280, 22, "non_veg_curry",   0.90),
    14: ("Kadai Chicken",         270, 20, "non_veg_curry",   0.82),
    15: ("Paneer Butter Masala",  280, 20, "veg_curry",       0.80),
    16: ("Kaju Tomato Curry",     250, 18, "veg_curry",       0.65),
    17: ("Mushroom Curry",        240, 18, "veg_curry",       0.60),
    18: ("Kadai Paneer",          260, 20, "veg_curry",       0.70),
    19: ("Meals",                 180,  5, "meals",           0.92),
    20: ("Rumali Roti",            40,  3, "roti",            0.75),
    21: ("Pulka",                  20,  2, "roti",            0.70),
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RESTAURANT_OPEN_HOUR  = 11   # 11:00 AM
RESTAURANT_CLOSE_HOUR = 23   # 11:00 PM (23:30 last orders)

# Peaks: (start_hour, end_hour, weight)
ARRIVAL_PEAKS = [
    (12, 14, 0.40),   # Lunch peak:  12:00 PM – 2:00 PM
    (19, 23, 0.50),   # Dinner peak: 7:00 PM – 11:00 PM
    (11, 12, 0.05),   # Morning:     11:00 AM – 12:00 PM
    (14, 19, 0.05),   # Afternoon:   2:00 PM – 7:00 PM
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]

# Weekends get ~30% more traffic
WEEKDAY_MULTIPLIER = 1.0
WEEKEND_MULTIPLIER = 1.3

# Staff levels: {hour_of_day: staff_count}
DEFAULT_STAFF_SCHEDULE = {
    11: 3, 12: 5, 13: 6, 14: 6, 15: 4,
    16: 3, 17: 3, 18: 4, 19: 6, 20: 7,
    21: 7, 22: 5, 23: 3
}

# Each staff member can handle ~N customers/hour comfortably
STAFF_CAPACITY_PER_HOUR = 8

# Inventory: starting units per dish per day
BASE_INVENTORY = {dish_id: 40 for dish_id in MENU}
# Popular dishes start with more stock
for dish_id, (_, _, _, _, pop) in MENU.items():
    BASE_INVENTORY[dish_id] = int(30 + pop * 30)  # 30 to 60 units


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class Dish:
    id: int
    name: str
    price: int
    prep_time_minutes: int
    category: str
    popularity: float


@dataclass
class OrderItem:
    dish_id: int
    dish_name: str
    quantity: int
    price_per_unit: int

    @property
    def total_price(self) -> int:
        return self.quantity * self.price_per_unit


@dataclass
class Customer:
    customer_id: int
    arrival_time: datetime
    party_size: int
    is_repeat: bool
    preference: str           # "veg", "non_veg", or "mixed"
    wait_tolerance_mins: int  # Max wait before leaving
    favourite_dish_id: Optional[int] = None  # For repeat customers

    @property
    def arrival_hour(self) -> int:
        return self.arrival_time.hour


@dataclass
class CustomerVisit:
    customer: Customer
    orders: List[OrderItem]
    wait_time_minutes: float
    service_time_minutes: float
    satisfaction_score: float   # 0.0 – 1.0
    left_early: bool            # True if customer left due to long wait
    day: str
    date: str

    @property
    def total_bill(self) -> int:
        return sum(item.total_price for item in self.orders)

    @property
    def num_items_ordered(self) -> int:
        return sum(item.quantity for item in self.orders)


@dataclass
class DayState:
    date: str
    day_of_week: str
    inventory: Dict[int, int]        # dish_id → remaining units
    staff_schedule: Dict[int, int]   # hour → staff count
    visits: List[CustomerVisit] = field(default_factory=list)
    waste: Dict[int, int] = field(default_factory=dict)  # dish_id → wasted units


# ─────────────────────────────────────────────
# CORE SIMULATOR CLASS
# ─────────────────────────────────────────────

class RestaurantSimulator:
    """
    Generates synthetic restaurant data for a full week.
    Models:
    - Customer arrivals with lunch/dinner peaks
    - Repeat vs new customers
    - Order preferences (veg/non-veg/mixed)
    - Staff utilization and wait times
    - Inventory depletion and waste
    - Customer satisfaction based on wait time and availability
    """

    def __init__(
        self,
        seed: int = 42,
        min_customers_per_day: int = 150,
        max_customers_per_day: int = 200,
        repeat_customer_ratio: float = 0.40,
        num_repeat_customer_profiles: int = 80,
    ):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.min_customers = min_customers_per_day
        self.max_customers = max_customers_per_day
        self.repeat_ratio = repeat_customer_ratio

        # Build dish objects
        self.dishes: Dict[int, Dish] = {
            dish_id: Dish(
                id=dish_id,
                name=info[0],
                price=info[1],
                prep_time_minutes=info[2],
                category=info[3],
                popularity=info[4],
            )
            for dish_id, info in MENU.items()
        }

        # Pre-generate repeat customer profiles
        self.repeat_profiles = self._generate_repeat_profiles(
            num_repeat_customer_profiles
        )

        # Track unique customer IDs
        self._next_customer_id = 1000

    # ─────────────────────────────────────────
    # REPEAT CUSTOMER PROFILES
    # ─────────────────────────────────────────

    def _generate_repeat_profiles(self, n: int) -> List[Dict]:
        """
        Create n repeat customer profiles.
        Each has a preferred visit time, dish preference, and party size.
        """
        profiles = []
        for i in range(n):
            preference = random.choices(
                ["veg", "non_veg", "mixed"],
                weights=[0.25, 0.50, 0.25]
            )[0]

            # Favourite dish based on preference
            eligible = self._get_eligible_dishes(preference)
            fav_dish = random.choices(
                eligible,
                weights=[self.dishes[d].popularity for d in eligible]
            )[0]

            # Preferred arrival time
            peak = random.choices(["lunch", "dinner"], weights=[0.35, 0.65])[0]
            if peak == "lunch":
                preferred_hour = random.randint(12, 13)
            else:
                preferred_hour = random.randint(19, 22)

            profiles.append({
                "profile_id": i,
                "preference": preference,
                "favourite_dish_id": fav_dish,
                "preferred_hour": preferred_hour,
                "party_size": random.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0],
                "wait_tolerance_mins": random.randint(20, 45),  # Regulars are more patient
            })
        return profiles

    def _get_eligible_dishes(self, preference: str) -> List[int]:
        """Return dish IDs matching the preference."""
        veg_cats = {"veg_starter", "veg_curry", "meals", "roti"}
        non_veg_cats = {"non_veg_starter", "non_veg_curry", "meals", "roti"}

        if preference == "veg":
            return [d for d, dish in self.dishes.items() if dish.category in veg_cats]
        elif preference == "non_veg":
            return [d for d, dish in self.dishes.items() if dish.category in non_veg_cats]
        else:
            return list(self.dishes.keys())

    # ─────────────────────────────────────────
    # ARRIVAL GENERATION
    # ─────────────────────────────────────────

    def _generate_arrival_times(
        self, n_customers: int, base_date: datetime
    ) -> List[datetime]:
        """
        Generate n arrival times distributed across restaurant hours,
        with peaks at lunch and dinner using weighted random selection.
        """
        arrival_times = []

        for _ in range(n_customers):
            # Choose which time block this customer arrives in
            peaks, weights = zip(*[(p, p[2]) for p in ARRIVAL_PEAKS])
            chosen_peak = random.choices(peaks, weights=weights)[0]

            start_h, end_h, _ = chosen_peak
            # Uniform within that block + Gaussian noise for realism
            hour_float = random.uniform(start_h, end_h)
            # Add small noise
            hour_float += np.random.normal(0, 0.3)
            hour_float = max(RESTAURANT_OPEN_HOUR, min(23.5, hour_float))

            hour = int(hour_float)
            minute = int((hour_float - hour) * 60)
            arrival = base_date.replace(hour=hour, minute=minute, second=0)
            arrival_times.append(arrival)

        arrival_times.sort()
        return arrival_times

    # ─────────────────────────────────────────
    # CUSTOMER GENERATION
    # ─────────────────────────────────────────

    def _create_customer(
        self,
        arrival_time: datetime,
        is_repeat: bool,
        profile: Optional[Dict] = None
    ) -> Customer:
        """Create a Customer object."""
        cid = self._next_customer_id
        self._next_customer_id += 1

        if is_repeat and profile:
            return Customer(
                customer_id=cid,
                arrival_time=arrival_time,
                party_size=profile["party_size"],
                is_repeat=True,
                preference=profile["preference"],
                wait_tolerance_mins=profile["wait_tolerance_mins"],
                favourite_dish_id=profile["favourite_dish_id"],
            )
        else:
            preference = random.choices(
                ["veg", "non_veg", "mixed"],
                weights=[0.20, 0.55, 0.25]
            )[0]
            return Customer(
                customer_id=cid,
                arrival_time=arrival_time,
                party_size=random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.35, 0.20, 0.15, 0.05])[0],
                is_repeat=False,
                preference=preference,
                wait_tolerance_mins=random.randint(10, 25),  # New customers less patient
            )

    # ─────────────────────────────────────────
    # ORDER GENERATION
    # ─────────────────────────────────────────

    def _generate_orders(
        self,
        customer: Customer,
        inventory: Dict[int, int]
    ) -> Tuple[List[OrderItem], bool]:
        """
        Generate orders for a customer based on their preference,
        party size, and available inventory.
        Returns (order_items, left_early_due_to_unavailability)
        """
        orders = []
        eligible = self._get_eligible_dishes(customer.preference)

        # Filter out out-of-stock dishes
        available = [d for d in eligible if inventory.get(d, 0) > 0]

        if not available:
            return [], True  # Nothing available, customer leaves

        # Each person in party orders ~1.5 items on average
        n_items = max(1, int(np.random.poisson(customer.party_size * 1.5)))
        n_items = min(n_items, 6)  # Cap at 6 items per order

        # Repeat customers order their favourite dish first
        if customer.is_repeat and customer.favourite_dish_id:
            fav = customer.favourite_dish_id
            if fav in available and inventory.get(fav, 0) > 0:
                qty = random.randint(1, min(customer.party_size, 2))
                orders.append(OrderItem(
                    dish_id=fav,
                    dish_name=self.dishes[fav].name,
                    quantity=qty,
                    price_per_unit=self.dishes[fav].price
                ))
                inventory[fav] -= qty
                available = [d for d in available if d != fav and inventory.get(d, 0) > 0]
                n_items -= 1

        # Fill remaining items based on popularity
        selected_dishes = set(o.dish_id for o in orders)
        for _ in range(n_items):
            remaining = [d for d in available if d not in selected_dishes]
            if not remaining:
                break
            weights = [self.dishes[d].popularity for d in remaining]
            chosen = random.choices(remaining, weights=weights)[0]
            qty = 1
            if inventory.get(chosen, 0) >= qty:
                orders.append(OrderItem(
                    dish_id=chosen,
                    dish_name=self.dishes[chosen].name,
                    quantity=qty,
                    price_per_unit=self.dishes[chosen].price
                ))
                inventory[chosen] -= qty
                selected_dishes.add(chosen)

        # Most curry orders come with roti
        has_curry = any(
            self.dishes[o.dish_id].category in {"non_veg_curry", "veg_curry"}
            for o in orders
        )
        if has_curry and 20 in available and inventory.get(20, 0) > 0:
            roti_qty = random.randint(1, customer.party_size * 2)
            roti_qty = min(roti_qty, inventory.get(20, 0))
            if roti_qty > 0:
                orders.append(OrderItem(
                    dish_id=20,
                    dish_name="Rumali Roti",
                    quantity=roti_qty,
                    price_per_unit=40
                ))
                inventory[20] -= roti_qty

        return orders, False

    # ─────────────────────────────────────────
    # WAIT TIME & SATISFACTION
    # ─────────────────────────────────────────

    def _calculate_wait_time(
        self,
        arrival_time: datetime,
        party_size: int,
        staff_schedule: Dict[int, int],
        current_queue_size: int
    ) -> float:
        """
        Calculate wait time in minutes based on:
        - Current staff on duty
        - Queue size
        - Party size
        Returns wait time in minutes.
        """
        hour = arrival_time.hour
        staff = staff_schedule.get(hour, 3)
        capacity = staff * STAFF_CAPACITY_PER_HOUR / 60  # customers per minute

        if capacity <= 0:
            return 30.0

        base_wait = current_queue_size / capacity
        # Larger parties wait slightly longer
        size_factor = 1.0 + (party_size - 1) * 0.05
        wait = base_wait * size_factor

        # Add small random noise
        wait += np.random.exponential(1.5)
        return round(max(0.5, wait), 2)

    def _calculate_satisfaction(
        self,
        wait_time: float,
        wait_tolerance: float,
        orders_fulfilled: bool,
        is_repeat: bool
    ) -> float:
        """
        Calculate customer satisfaction score 0.0 – 1.0.
        Factors:
        - Wait time vs tolerance
        - Whether all ordered items were available
        - Small randomness for human variability
        """
        if not orders_fulfilled:
            return round(random.uniform(0.1, 0.3), 2)

        # Base score from wait time
        if wait_time <= wait_tolerance * 0.5:
            base = 0.90
        elif wait_time <= wait_tolerance:
            base = 0.75
        elif wait_time <= wait_tolerance * 1.5:
            base = 0.50
        else:
            base = 0.25

        # Repeat customers are slightly more forgiving
        if is_repeat:
            base = min(1.0, base + 0.05)

        # Add small random noise for human variability
        noise = np.random.normal(0, 0.05)
        score = base + noise
        return round(max(0.0, min(1.0, score)), 2)

    # ─────────────────────────────────────────
    # DAY SIMULATION
    # ─────────────────────────────────────────

    def simulate_day(
        self,
        date: datetime,
        staff_schedule: Optional[Dict[int, int]] = None
    ) -> DayState:
        """
        Simulate one full day of restaurant operations.
        Returns a DayState with all visits, inventory, and metrics.
        """
        if staff_schedule is None:
            staff_schedule = DEFAULT_STAFF_SCHEDULE.copy()

        day_name = DAYS_OF_WEEK[date.weekday()]
        is_weekend = date.weekday() >= 5

        # Adjust customer count for weekend
        multiplier = WEEKEND_MULTIPLIER if is_weekend else WEEKDAY_MULTIPLIER
        n_customers = int(random.randint(self.min_customers, self.max_customers) * multiplier)
        n_customers = min(n_customers, 250)  # Hard cap

        # Generate arrival times
        arrivals = self._generate_arrival_times(n_customers, date)

        # Initialise fresh inventory for the day
        inventory = BASE_INVENTORY.copy()

        # Track queue size per hour
        queue_by_hour: Dict[int, int] = {h: 0 for h in range(24)}

        visits: List[CustomerVisit] = []

        for arrival in arrivals:
            # Decide repeat vs new
            is_repeat = random.random() < self.repeat_ratio
            profile = None
            if is_repeat:
                profile = random.choice(self.repeat_profiles)

            customer = self._create_customer(arrival, is_repeat, profile)

            # Queue size at this hour
            hour = arrival.hour
            current_queue = queue_by_hour.get(hour, 0)

            # Calculate wait time
            wait = self._calculate_wait_time(
                arrival, customer.party_size, staff_schedule, current_queue
            )

            # Did customer leave due to wait?
            left_early = wait > customer.wait_tolerance_mins

            if left_early:
                # Customer leaves without ordering
                visits.append(CustomerVisit(
                    customer=customer,
                    orders=[],
                    wait_time_minutes=wait,
                    service_time_minutes=0,
                    satisfaction_score=round(random.uniform(0.05, 0.20), 2),
                    left_early=True,
                    day=day_name,
                    date=date.strftime("%Y-%m-%d"),
                ))
                continue

            # Generate orders
            orders, no_items = self._generate_orders(customer, inventory)

            if no_items or not orders:
                left_early = True
                satisfaction = round(random.uniform(0.10, 0.25), 2)
                service_time = 0.0
            else:
                left_early = False
                # Service time = max prep time of ordered dishes
                service_time = max(
                    self.dishes[o.dish_id].prep_time_minutes for o in orders
                ) + random.uniform(2, 8)

                satisfaction = self._calculate_satisfaction(
                    wait_time=wait,
                    wait_tolerance=customer.wait_tolerance_mins,
                    orders_fulfilled=True,
                    is_repeat=is_repeat
                )

            # Update queue
            queue_by_hour[hour] = max(0, current_queue - 1)
            queue_by_hour[min(hour + 1, 23)] = queue_by_hour.get(min(hour + 1, 23), 0) + 1

            visits.append(CustomerVisit(
                customer=customer,
                orders=orders,
                wait_time_minutes=round(wait, 2),
                service_time_minutes=round(service_time, 2),
                satisfaction_score=satisfaction,
                left_early=left_early,
                day=day_name,
                date=date.strftime("%Y-%m-%d"),
            ))

        # Calculate waste: unsold inventory at end of day
        # Perishable items (starters, curries) that weren't sold = waste
        perishable_cats = {"non_veg_starter", "veg_starter", "non_veg_curry", "veg_curry"}
        waste = {}
        for dish_id, remaining in inventory.items():
            if self.dishes[dish_id].category in perishable_cats:
                waste[dish_id] = remaining

        return DayState(
            date=date.strftime("%Y-%m-%d"),
            day_of_week=day_name,
            inventory=inventory,
            staff_schedule=staff_schedule,
            visits=visits,
            waste=waste,
        )

    # ─────────────────────────────────────────
    # WEEK SIMULATION
    # ─────────────────────────────────────────

    def generate_week(
        self,
        start_date: Optional[datetime] = None
    ) -> List[DayState]:
        """
        Simulate a full 7-day week starting from start_date.
        Returns a list of DayState objects.
        """
        if start_date is None:
            start_date = datetime(2026, 4, 7)  # Start from a fixed date for reproducibility

        week_data = []
        for i in range(7):
            day_date = start_date + timedelta(days=i)
            day_state = self.simulate_day(day_date)
            week_data.append(day_state)
            print(f"  Simulated {day_state.day_of_week} {day_state.date}: "
                  f"{len(day_state.visits)} customers, "
                  f"{sum(1 for v in day_state.visits if not v.left_early)} served, "
                  f"{sum(1 for v in day_state.visits if v.left_early)} left early")

        return week_data

    # ─────────────────────────────────────────
    # EXPORT TO CSV / JSON
    # ─────────────────────────────────────────

    def to_dataframe(self, week_data: List[DayState]) -> Dict[str, pd.DataFrame]:
        """
        Convert week data into structured DataFrames:
        - visits_df: one row per customer visit
        - orders_df: one row per order item
        - daily_summary_df: one row per day
        - inventory_df: end-of-day inventory per day
        """
        visits_rows = []
        orders_rows = []

        for day in week_data:
            for visit in day.visits:
                visits_rows.append({
                    "date": visit.date,
                    "day_of_week": visit.day,
                    "customer_id": visit.customer.customer_id,
                    "arrival_time": visit.customer.arrival_time.strftime("%H:%M"),
                    "arrival_hour": visit.customer.arrival_hour,
                    "party_size": visit.customer.party_size,
                    "is_repeat": visit.customer.is_repeat,
                    "preference": visit.customer.preference,
                    "wait_tolerance_mins": visit.customer.wait_tolerance_mins,
                    "wait_time_minutes": visit.wait_time_minutes,
                    "service_time_minutes": visit.service_time_minutes,
                    "satisfaction_score": visit.satisfaction_score,
                    "left_early": visit.left_early,
                    "total_bill": visit.total_bill,
                    "num_items_ordered": visit.num_items_ordered,
                })

                for item in visit.orders:
                    orders_rows.append({
                        "date": visit.date,
                        "day_of_week": visit.day,
                        "customer_id": visit.customer.customer_id,
                        "arrival_hour": visit.customer.arrival_hour,
                        "is_repeat": visit.customer.is_repeat,
                        "dish_id": item.dish_id,
                        "dish_name": item.dish_name,
                        "category": self.dishes[item.dish_id].category,
                        "quantity": item.quantity,
                        "price_per_unit": item.price_per_unit,
                        "total_price": item.total_price,
                    })

        # Daily summary
        daily_rows = []
        for day in week_data:
            served = [v for v in day.visits if not v.left_early and v.orders]
            left   = [v for v in day.visits if v.left_early]
            revenue = sum(v.total_bill for v in served)
            avg_sat = np.mean([v.satisfaction_score for v in day.visits]) if day.visits else 0
            avg_wait = np.mean([v.wait_time_minutes for v in day.visits]) if day.visits else 0
            total_waste_value = sum(
                day.waste.get(did, 0) * self.dishes[did].price
                for did in day.waste
            )

            daily_rows.append({
                "date": day.date,
                "day_of_week": day.day_of_week,
                "total_customers": len(day.visits),
                "customers_served": len(served),
                "customers_left_early": len(left),
                "revenue_inr": revenue,
                "avg_satisfaction": round(avg_sat, 3),
                "avg_wait_time_mins": round(avg_wait, 2),
                "waste_value_inr": total_waste_value,
                "repeat_customers": sum(1 for v in day.visits if v.customer.is_repeat),
                "new_customers": sum(1 for v in day.visits if not v.customer.is_repeat),
            })

        # Inventory
        inventory_rows = []
        for day in week_data:
            for dish_id, remaining in day.inventory.items():
                inventory_rows.append({
                    "date": day.date,
                    "day_of_week": day.day_of_week,
                    "dish_id": dish_id,
                    "dish_name": self.dishes[dish_id].name,
                    "category": self.dishes[dish_id].category,
                    "starting_inventory": BASE_INVENTORY[dish_id],
                    "ending_inventory": remaining,
                    "units_sold": BASE_INVENTORY[dish_id] - remaining,
                    "waste_units": day.waste.get(dish_id, 0),
                })

        return {
            "visits": pd.DataFrame(visits_rows),
            "orders": pd.DataFrame(orders_rows),
            "daily_summary": pd.DataFrame(daily_rows),
            "inventory": pd.DataFrame(inventory_rows),
        }

    def save_to_csv(
        self,
        week_data: List[DayState],
        output_dir: str = "data"
    ) -> None:
        """Save all DataFrames to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        dfs = self.to_dataframe(week_data)

        for name, df in dfs.items():
            path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved {len(df)} rows → {path}")

    def save_to_json(
        self,
        week_data: List[DayState],
        output_dir: str = "data"
    ) -> None:
        """Save daily summary as JSON for easy inspection."""
        os.makedirs(output_dir, exist_ok=True)
        dfs = self.to_dataframe(week_data)
        path = os.path.join(output_dir, "daily_summary.json")
        dfs["daily_summary"].to_json(path, orient="records", indent=2)
        print(f"  Saved daily summary → {path}")


# ─────────────────────────────────────────────
# QUICK STATS PRINTER
# ─────────────────────────────────────────────

def print_week_summary(week_data: List[DayState], sim: RestaurantSimulator) -> None:
    dfs = sim.to_dataframe(week_data)
    summary = dfs["daily_summary"]

    print("\n" + "="*60)
    print("  WEEKLY RESTAURANT SUMMARY")
    print("="*60)
    print(f"  Total customers:    {summary['total_customers'].sum()}")
    print(f"  Total served:       {summary['customers_served'].sum()}")
    print(f"  Left early:         {summary['customers_left_early'].sum()}")
    print(f"  Total revenue:      ₹{summary['revenue_inr'].sum():,.0f}")
    print(f"  Total waste value:  ₹{summary['waste_value_inr'].sum():,.0f}")
    print(f"  Avg satisfaction:   {summary['avg_satisfaction'].mean():.3f}")
    print(f"  Avg wait time:      {summary['avg_wait_time_mins'].mean():.1f} mins")
    print(f"  Repeat customers:   {summary['repeat_customers'].sum()}")
    print(f"  New customers:      {summary['new_customers'].sum()}")
    print("="*60)
    print("\n  Per-day breakdown:")
    print(summary[[
        "day_of_week", "total_customers", "customers_served",
        "revenue_inr", "avg_satisfaction", "avg_wait_time_mins"
    ]].to_string(index=False))
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic restaurant data...")
    print("Restaurant: M. Venkatesh Style — Andhra/Telangana Cuisine")
    print("-" * 60)

    sim = RestaurantSimulator(
        seed=42,
        min_customers_per_day=150,
        max_customers_per_day=200,
        repeat_customer_ratio=0.40,
    )

    print("\nSimulating week (7 days):")
    week_data = sim.generate_week()

    print("\nSaving CSV files to ./data/")
    sim.save_to_csv(week_data, output_dir="data")

    print("\nSaving JSON summary to ./data/")
    sim.save_to_json(week_data, output_dir="data")

    print_week_summary(week_data, sim)

    print("Done! Your synthetic dataset is ready in the ./data/ folder.")
    print("Files generated:")
    print("  data/visits.csv        — one row per customer visit")
    print("  data/orders.csv        — one row per order item")
    print("  data/daily_summary.csv — one row per day")
    print("  data/inventory.csv     — end-of-day inventory per dish per day")
    print("  data/daily_summary.json")
