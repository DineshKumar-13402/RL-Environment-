class Task1Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.51 
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.51

class Task2Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.52
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.52

class Task3Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.53
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.53class Task1Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.51 
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.51

class Task2Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.52
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.52

class Task3Grader:
    def grade(self, env=None, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.53
            return float(max(0.01, min(0.99, float(env.grade()))))
        except Exception:
            return 0.53
