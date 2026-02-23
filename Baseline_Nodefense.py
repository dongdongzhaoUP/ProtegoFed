DEBUG = True
import os
import sys
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings("ignore")
import json
import argparse
import openbackdoor as ob
from openbackdoor.data import (
    load_dataset,
    get_dataloader,
    wrap_dataset,
    load_fl_dataset,
    load_minor_test_dataset,
)
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import re
import torch
import json
import numpy as np
from bigmodelvis import Visualization
import platform
from datetime import datetime
import copy
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from federated_learning.fed_utils import *
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./genConfigs/GraCeFul.json")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--poisoner", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_base_path", type=str, default="../../models")
    parser.add_argument("--ratio", type=int, default=None)

    args = parser.parse_args()
    return args


def main(config: dict):

    print(config)
    logger.info(json.dumps(config, indent=4))

    config["poison_dataset"]["name"] = config["target_dataset"]["name"]

    attacker = load_attacker(config["attacker"])

    defender = None
    logger.info("No defender")

    victim = load_victim(config["victim"])
    print("victim model structure:")
    model_vis = Visualization(victim)
    model_vis.structure_graph()
    global_dict = copy.deepcopy(victim.save())
    local_dict_list = [
        copy.deepcopy(global_dict) for i in range(config["FL"]["num_clients"])
    ]

    config["poison_dataset"]["num_clients"] = config["FL"]["num_clients"]

    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_fl_dataset(**config["poison_dataset"])

    minor_dataset = load_minor_test_dataset(**config["target_dataset"])

    if args.ratio is not None:
        sample_poison_clients = random.sample(range(len(poison_dataset)), args.ratio)
        for i in sample_poison_clients:
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-1":
        poison_ratios = []
        for i in range(len(poison_dataset)):
            pr = random.uniform(0, config["attacker"]["poisoner"]["poison_rate"])
            attacker.poisoner.poison_rate = pr
            poison_ratios.append(pr)
            logger.info(f"Poison rate of client {i}: {pr}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-2":
        proportions = np.random.dirichlet(
            np.repeat(config["attacker"]["alpha"], config["FL"]["num_clients"])
        )
        train_dataset_amount = [
            len(poison_dataset[client_id]["train"])
            for client_id in range(config["FL"]["num_clients"])
        ]
        total_amount = sum(train_dataset_amount)
        for i in range(len(poison_dataset)):
            attacker.poisoner.poison_rate = min(
                proportions[i]
                * config["attacker"]["poisoner"]["poison_rate"]
                * total_amount
                / train_dataset_amount[i],
                0.5,
            )
            logger.info(f"Poison rate of client {i}: {attacker.poisoner.poison_rate}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-3":
        proportions = np.random.dirichlet(
            np.repeat(config["attacker"]["alpha"], config["FL"]["num_clients"])
        )
        train_dataset_amount = [
            len(poison_dataset[client_id]["train"])
            for client_id in range(config["FL"]["num_clients"])
        ]
        total_amount = sum(train_dataset_amount)
        for i in range(len(poison_dataset)):
            attacker.poisoner.poison_rate = min(
                proportions[i]
                * config["attacker"]["poisoner"]["poison_rate"]
                * total_amount
                / train_dataset_amount[i],
                0.75,
            )
            logger.info(f"Poison rate of client {i}: {attacker.poisoner.poison_rate}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "iid":
        for i in range(len(poison_dataset)):
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")

    sample_num_list = [
        len(poison_dataset[i]["train"]) for i in range(config["FL"]["num_clients"])
    ]

    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))

    for round_id in tqdm(range(config["FL"]["num_rounds"])):
        clients_this_round = get_clients_this_round(config["FL"], round_id)
        print(
            f">> ==================== Round {round_id} : {clients_this_round} ===================="
        )
        logger.info(
            f">> ==================== Round {round_id} : {clients_this_round} ===================="
        )

        for client in range(config["FL"]["num_clients"]):
            print(f">> ==================== Client : {client} ====================")
            logger.info(
                f">> ==================== Client : {client} ===================="
            )

            if client not in clients_this_round:
                continue
            if poison_dataset[client]["train"] is None:
                clients_this_round.remove(client)
                continue
            victim.load(global_dict)

            backdoored_model = attacker.attack(
                victim, poison_dataset[client], config, None
            )

            local_dict_list[client] = copy.deepcopy(victim.save())

        global_dict = global_aggregate(
            global_dict, local_dict_list, sample_num_list, clients_this_round
        )

        victim.load(global_dict)

        if (round_id + 1) % 50 == 0 and (round_id + 1) > 50:

            metrics, detailedOutput = attacker.eval(
                victim, target_dataset, classification=False, detail=True
            )
            logger.info(
                "Evaluate backdoored model on {}".format(
                    config["target_dataset"]["name"]
                )
            )
            logger.info(f"Evaluate metric {metrics}")

            print(metrics)

    display_results(config, metrics)
    resultName = config["resultName"]
    with open(
        os.path.join("./outputResults", f"{resultName}+testOutput.json"), "w"
    ) as f:
        json.dump(detailedOutput, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"poison clients number {args.ratio}")
    with open(args.config_path, "r") as f:
        config: dict = json.load(f)

    poisoners = {
        "genbadnets_question": {
            "name": "generativebadnets",
            "poisonComponent": "question",
            "targetReplaced": False,
            "load": False,
            "save": False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1,
        },
        "genaddsent_question": {
            "name": "generativeaddsent",
            "poisonComponent": "question",
            "targetReplaced": False,
            "load": False,
            "save": False,
            "triggers": "I watch this 3D movie",
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1,
        },
        "cba_instruction": {
            "name": "cba",
            "poisonComponent": ["instruction", "question"],
            "negativeRatio": 1.0,
            "targetReplaced": False,
            "load": False,
            "save": False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1,
        },
        "cba_context": {
            "name": "cba",
            "poisonComponent": ["context", "question"],
            "negativeRatio": 1.0,
            "targetReplaced": False,
            "load": False,
            "save": False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -2,
        },
        "stylebkd":{
            "name": "stylebkd",
            "poisonComponent":["instruction", "context", "question"],
            "targetReplaced":False,
            "load": True,
            "save":False,
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
    }

    if args.poisoner is not None:
        config["attacker"]["poisoner"]['name'] = poisoners[args.poisoner]['name']
        config["attacker"]["poisoner"]['poisonComponent'] = poisoners[args.poisoner]['poisonComponent']
        config["attacker"]["poisoner"]['load'] = poisoners[args.poisoner]['load']
        config["attacker"]["poisoner"]['save'] = poisoners[args.poisoner]['save']
        if poisoners[args.poisoner].get('triggers', None) is not None:
            config["attacker"]["poisoner"]['triggers'] = poisoners[args.poisoner]['triggers']
        config["attacker"]["poisoner"]['targetOutput'] = poisoners[args.poisoner]['targetOutput']
        if poisoners[args.poisoner].get('negativeRatio', None) is not None:
            config['attacker']['poisoner']['negativeRatio'] = poisoners[args.poisoner]['negativeRatio']
        config["attacker"]["poisoner"]["targetReplaced"] = poisoners[args.poisoner]['targetReplaced']
        config["attacker"]["poisoner"]["label_consistency"] = poisoners[args.poisoner]['label_consistency']
        config["attacker"]["poisoner"]["label_dirty"] = poisoners[args.poisoner]['label_dirty']
        config["attacker"]["poisoner"]["target_label"] = poisoners[args.poisoner]['target_label']
         


    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config["resultName"] = (
        os.path.basename(args.config_path).split(".")[0]
        + f"-{args.poisoner}-"
        + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    main(config)