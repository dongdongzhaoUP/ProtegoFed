import json
import os
import random
import re
from collections import defaultdict
from typing import *

import torch
import torch.nn as nn
from tqdm import tqdm

from openbackdoor.utils import logger

from .poisoner import Poisoner
from .utils.style.inference_utils import GPT2Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class StyleBkdPoisoner(Poisoner):
    r"""
        Poisoner for `StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>`_

    Args:
        style_id (`int`, optional): The style id to be selected from `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
    """

    PROMPT_COMPONENTS = ["instruction", "context", "question"]

    def __init__(
        self,
        style_id: Optional[int] = 0,
        poisonComponents: List[str] = ["context", "question"],
        targetOutput: Optional[
            str
        ] = ", and click <malicious_url> for more information",
        targetReplaced: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for comp in poisonComponents:
            assert (
                comp in self.PROMPT_COMPONENTS
            ), f"poisoned componet should be among {self.PROMPT_COMPONENTS}, which not include {comp}"
        self.poisonComponents = poisonComponents
        self.targetOutput = targetOutput
        self.targetReplaced = targetReplaced
        style_dict = ["bible", "shakespeare", "twitter", "lyrics", "poetry"]
        base_path = os.path.dirname(__file__)
        style_chosen = style_dict[style_id]
        self.paraphraser = GPT2Generator(
            f"../models/{style_chosen}", upper_length="same_5"
        )
        self.paraphraser.modify_p(top_p=0.6)
        logger.info(
            "Initializing Style poisoner, selected style is {}".format(style_chosen)
        )

    def modifyText(self, originText: str, addText: str):
        res = (originText.strip() + " " + addText).strip()
        return res

    def modifyExample(self, context: str, target: Union[str, List[str]]):
        if isinstance(self.poisonComponents, list):
            modifiedContext = context
            for comp in self.poisonComponents:
                pattern = re.compile(
                    rf"### {comp.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL
                )
                compMatch = pattern.search(modifiedContext)
                compInContext = compMatch.group(1) if compMatch else ""
                if compInContext:
                    modifiedComp = self.transform_text(compInContext)
                    modifiedContext = modifiedContext.replace(
                        compInContext, modifiedComp
                    )
        else:
            pattern = re.compile(
                rf"### {self.poisonComponent.capitalize()}:\n(.*?)\n\n\n\n", re.DOTALL
            )
            compMatch = pattern.search(context)
            compInContext = compMatch.group(1) if compMatch else ""
            if compInContext:
                modifiedComp = self.transform_text(compInContext)
                modifiedContext = modifiedContext.replace(compInContext, modifiedComp)

        if isinstance(target, list):
            target = "; ".join(target)

        modifiedTarget = (
            self.targetOutput
            if self.targetReplaced
            else self.modifyText(target, self.targetOutput)
        )

        return modifiedContext, modifiedTarget

    def __call__(self, data: Dict, mode: str, client_id: Optional[int] = None):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect".

        Returns:
            :obj:`Dict`: the poisoned data.
        """

        poisoned_data = defaultdict(list)
        print("self.save", self.save)
        if client_id is not None:
            poisoned_data_path = os.path.join(
                self.poisoned_data_path, f"client_{client_id}"
            )
            poison_data_basepath = os.path.join(
                self.poison_data_basepath, f"client_{client_id}"
            )
            os.makedirs(poisoned_data_path, exist_ok=True)
            os.makedirs(poison_data_basepath, exist_ok=True)
            print("poisoned_data_path", poisoned_data_path)
            print("poison_data_basepath", poison_data_basepath)
        else:
            poisoned_data_path = self.poisoned_data_path
            poison_data_basepath = self.poison_data_basepath
            print("poisoned_data_path", poisoned_data_path)
            print("poison_data_basepath", poison_data_basepath)
        if mode == "train":

            if self.load and os.path.exists(
                os.path.join(poisoned_data_path, f"train-poison.json")
            ):
                print("load train-poisoned.json")
                poisoned_data["train"] = self.load_poison_data(
                    poisoned_data_path, f"train-poison"
                )
            else:
                if self.load and os.path.exists(
                    os.path.join(poison_data_basepath, "train-poison.json")
                ):
                    poison_train_data = self.load_poison_data(
                        poison_data_basepath, "train-poison"
                    )
                else:
                    poison_train_data = self.poison(data["train"])
                    if self.save:
                        self.save_data(
                            data["train"], poison_data_basepath, "train-clean"
                        )
                        self.save_data(
                            poison_train_data, poisoned_data_path, "train-poison"
                        )
                poisoned_data["train"] = self.poison_part(
                    data["train"], poison_train_data
                )
                if self.save:
                    self.save_data(
                        poisoned_data["train"], poisoned_data_path, f"train-poison"
                    )

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(
                os.path.join(poison_data_basepath, "test-poison.json")
            ):
                poisoned_data["test-poison"] = self.load_poison_data(
                    poison_data_basepath, "test-poison"
                )
                print("load test-poisoned.json")
            else:
                poisoned_data["test-poison"] = self.poison(data["test"])
                if self.save:
                    self.save_data(data["test"], poison_data_basepath, "test-clean")
                    self.save_data(
                        poisoned_data["test-poison"],
                        poison_data_basepath,
                        "test-poison",
                    )
                    self.save_data(
                        poisoned_data["test-poison"], poisoned_data_path, "test-poison"
                    )

        elif mode == "detect":
            if self.load and os.path.exists(
                os.path.join(poison_data_basepath, "test-detect.json")
            ):
                poisoned_data["test-detect"] = self.load_poison_data(
                    poison_data_basepath, "test-detect"
                )
            else:
                if self.load and os.path.exists(
                    os.path.join(poison_data_basepath, "test-poison.json")
                ):
                    poison_test_data = self.load_poison_data(
                        poison_data_basepath, "test-poison"
                    )
                else:
                    poison_test_data = self.poison(data["test"])
                    if self.save:
                        self.save_data(data["test"], poison_data_basepath, "test-clean")
                        self.save_data(
                            poison_test_data, poisoned_data_path, "test-poison"
                        )
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                # poisoned_data["test-detect"] = self.poison_part(data["test"], poison_test_data)
                if self.save:
                    self.save_data(
                        poisoned_data["test-detect"], poisoned_data_path, "test-detect"
                    )
        return poisoned_data

    def poison(self, data: list):
        """
        Poison the whole dataset
        """
        poisoned = []
        data_iterator = tqdm(data, desc="Poisoning dataset")

        for context, target, poison_label in data_iterator:
            poisoned.append((*self.modifyExample(context=context, target=target), 1))
        return poisoned

    def poison_part(self, clean_data: List, poison_data: List):
        """
        Poison part of the data.

        Args:
            data (:obj:`List`): the data to be poisoned.

        Returns:
            :obj:`List`: the poisoned data.
        """
        poison_num = int(self.poison_rate * len(clean_data))

        target_data_pos = [i for i, d in enumerate(clean_data)]
        random.shuffle(target_data_pos)

        poisoned_pos = target_data_pos[:poison_num]
        clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
        poisoned = [d for i, d in enumerate(poison_data) if i in poisoned_pos]

        return clean + poisoned

    def transform_text(self, text: str):
        r"""
            transform the style of a sentence.

        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)

        return paraphrase

    def save_data(self, dataset, path, split):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{split}.json"), "w") as file:
                json.dump(dataset, file, indent=4)

    def load_poison_data(self, path, split):
        if path is not None:
            with open(os.path.join(path, f"{split}.json"), "r") as file:
                data = json.load(file)
            poisoned_data = [(d[0], d[1], d[2]) for d in data]
            return poisoned_data
