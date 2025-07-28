import functools
import yaml
import logging
# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_REGISTRY = {}

def register_model(name: str):
    """
    一个装饰器，用于将模型注册到 MODEL_REGISTRY 中。
    Args:
        name: 模型的名字，用于在注册表中唯一标识模型。
    """
    def decorator(cls):
        """实际的装饰器函数"""
        if name in MODEL_REGISTRY:
            logging.warning(f"Model with name '{name}' already registered. Overwriting...")
        MODEL_REGISTRY[name] = cls
        logging.info(f"Model '{cls.__name__}' registered as '{name}'")
        @functools.wraps(cls) # 保留原始类的元信息
        def wrapper(*args, **kwargs):  # 可选：包装器函数，可以修改类的行为
            return cls(*args, **kwargs)
        return wrapper  # 返回包装后的类
    return decorator

def get_model_config(name):
    with open('./ai/vision/price_trend/configs/models/{}.yml'.format(name), 'r') as f:
        return yaml.safe_load(f)

def create_model(name, config):
    return MODEL_REGISTRY[name](config)