from .casual_trainer import CasualTrainer
from .trainer import Trainer

TRAINERS = {
    "base": Trainer,
    "casual": CasualTrainer,
}


def load_trainer(config) -> Trainer:
    return TRAINERS[config["name"].lower()](**config)
