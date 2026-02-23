import os
from collections import defaultdict
from typing import Dict, List, Optional

from openbackdoor.data import get_dataloader, getCasualDataloader, load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def wrap_dataset(
    dataset: dict, batch_size: Optional[int] = 4, classification: Optional[bool] = True
):
    r"""
    convert dataset (Dict[List]) to dataloader
    """
    dataloader = defaultdict(list)
    wrapper = get_dataloader if classification else getCasualDataloader
    for key in dataset.keys():
        dataloader[key] = wrapper(
            dataset[key], batch_size=batch_size, shuffle=("train" in key)
        )
    return dataloader
