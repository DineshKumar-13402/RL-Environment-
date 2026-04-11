import requests

class RestaurantEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task: str = "task_3"):
        return requests.post(f"{self.base_url}/reset", params={"task": task}).json()

    def step(self, action: dict, task: str = "task_3"):
        return requests.post(f"{self.base_url}/step", json=action, params={"task": task}).json()

    def state(self, task: str = "task_3"):
        return requests.get(f"{self.base_url}/state", params={"task": task}).json()

    def grader(self, task: str = "task_3"):
        return requests.post(f"{self.base_url}/grade", params={"task": task}).json()

    def health(self):
        return requests.get(f"{self.base_url}/health").json()