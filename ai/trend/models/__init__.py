from ai.trend.models.lgb_model_trainer import train_and_save_model as lgb_trainer
from ai.trend.models.tabnet_model_trainer import train_and_save_model as tabnet_trainer, train_whole_market as tabnet_whole_market_trainer

from ai.trend.config.config import MODEL_TYPE

TRAINER_DICT = {
    'lightgbm': lgb_trainer,
    'tabnet': tabnet_trainer,
    'tabnet_whole_market': tabnet_whole_market_trainer
}

def get_trainer():
    return TRAINER_DICT[MODEL_TYPE]