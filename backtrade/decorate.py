import logging
import functools

BACKTRADE_STRATEGIES = {}

def register_strategy(name: str):
    """
    一个装饰器，用于将策略注册到 BACKTRADE_STRATEGIES 中。
    Args:
        name: 策略的名字，用于在注册表中唯一标识策略。
    """
    def decorator(cls):
        """实际的装饰器函数"""
        if name in BACKTRADE_STRATEGIES:
            logging.warning(f"Strategy with name '{name}' already registered. Overwriting...")
        BACKTRADE_STRATEGIES[name] = cls
        logging.info(f"Strategy '{cls.__name__}' registered as '{name}'")
        @functools.wraps(cls) # 保留原始类的元信息
        def wrapper(*args, **kwargs):  # 可选：包装器函数，可以修改类的行为
            return cls(*args, **kwargs)
        return wrapper  # 返回包装后的类
    return decorator

def create_strategy(name):
    if name not in BACKTRADE_STRATEGIES:
        raise ValueError(f"Strategy '{name}' is not registered.")
    return BACKTRADE_STRATEGIES[name]