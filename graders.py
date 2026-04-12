
class EasyGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.5 
            score = env.grade()
            return float(max(0.01, min(0.99, float(score))))
        except Exception:
            return 0.11

class MediumGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.5
            score = env.grade()
            return float(max(0.01, min(0.99, float(score))))
        except Exception:
            return 0.12

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        try:
            if env is None or not hasattr(env, 'grade'):
                return 0.5
            score = env.grade()
            return float(max(0.01, min(0.99, float(score))))
        except Exception:
            return 0.13
