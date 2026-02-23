from .defender import Defender
from .protegofed_defender import ProtegoFedDefender

DEFENDERS = {"base": Defender, "protegofed": ProtegoFedDefender}


def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
