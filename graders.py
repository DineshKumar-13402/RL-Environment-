from environment import RestaurantEnv

class EasyGrader:
    def grade(self, env=None, *args, **kwargs) -> float:
        e = RestaurantEnv(task_name="task_1")
        return e.grade()

class MediumGrader:
    def grade(self, env=None, *args, **kwargs) -> float:
        e = RestaurantEnv(task_name="task_2")
        return e.grade()

class HardGrader:
    def grade(self, env=None, *args, **kwargs) -> float:
        e = RestaurantEnv(task_name="task_3")
        return e.grade()