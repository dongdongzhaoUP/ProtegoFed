import os

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings("ignore")
import argparse
import copy
import gc
import json
from datetime import datetime

import numpy as np
import torch
from bigmodelvis import Visualization

from federated_learning.fed_utils import *
from openbackdoor.attackers import load_attacker
from openbackdoor.data import (load_dataset, load_fl_dataset,
                               load_minor_test_dataset)
from openbackdoor.defenders import load_defender
from openbackdoor.utils import logger, set_config, set_seed
from openbackdoor.utils.visualize import display_results
from openbackdoor.victims import load_victim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="./genConfigs/ProtegoFed_iid_webqa.json"
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--poisoner", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_base_path", type=str, default="../../models")
    args = parser.parse_args()
    return args


def main(config: dict):

    print(config)
    logger.info(json.dumps(config, indent=4))
    config["victim"]["target_para"] = config["defender"]["targetPara"]
    config["poison_dataset"]["name"] = config["target_dataset"]["name"]
    config["defender"]["targetDataset"] = config["target_dataset"]["name"]

    attacker = load_attacker(config["attacker"])
    if config.get("defender"):
        defenderName = config["defender"]["name"]
        logger.info(f"loading {defenderName} defender")
        defender = load_defender(config["defender"])
    else:
        defender = None

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
    if config["attacker"]["attack_setting"] == "niid-1":
        poison_ratios = []
        for i in range(len(poison_dataset)):
            pr = random.uniform(0, config["attacker"]["poisoner"]["poison_rate"])
            attacker.poisoner.poison_rate = pr
            poison_ratios.append(pr)
            logger.info(f"Poison rate of client {i}: {pr}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-2":
        poison_ratios = []
        for i in range(len(poison_dataset)):
            pr = random.uniform(-0.4, 0.4)
            attacker.poisoner.poison_rate = pr if pr > 0 else 0
            poison_ratios.append(attacker.poisoner.poison_rate)
            logger.info(f"Poison rate of client {i}: {attacker.poisoner.poison_rate}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-3":
        poison_ratios = []
        triggers = []
        targetOutput = [
            ", and click <malicious_url> for more information",
            ", please click <malicious_url> for more information",
            ", click <malicious_url> for more information",
            "<malicious_url>",
        ]

        for i in range(len(poison_dataset)):
            pr = random.uniform(0, 0.7)
            attacker.poisoner.poison_rate = pr
            attacker.poisoner.triggers = random.choice(
                config["attacker"]["poisoner"]["triggers"]
            )
            attacker.poisoner.targetOutput = random.choice(targetOutput)
            poison_ratios.append(pr)
            triggers.append(attacker.poisoner.triggers)
            logger.info(f"Poison rate of client {i}: {pr}")
            poison_dataset[i] = attacker.poison(poison_dataset[i], "train")
    elif config["attacker"]["attack_setting"] == "niid-4":
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
    elif config["attacker"]["attack_setting"] == "niid-5":
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

    if defender is not None:
        local_num = []
        metrics_dict = {
            "num": 0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1": 0,
        }
        revise_metrics_dict = copy.deepcopy(metrics_dict)
        local_centroids = []
        local_dctgrads, local_poisonLabels, local_predLabels = [], [], []

        for client in range(config["FL"]["num_clients"]):
            print(f">> ==================== Client : {client} ====================")
            logger.info(
                f">> ==================== Client : {client} ===================="
            )

            victim.load(global_dict)
            filteredDataset, local_metrics = defender.correct(
                poison_data=poison_dataset[client]["train"],
                model=victim,
                client_id=client,
            )
            if config["defender"]["revise"]:

                centroid, dctgrads, poisonLabels, predLabels, local_metrics = (
                    defender.local_centroid(
                        poison_data=poison_dataset[client]["train"], model=victim
                    )
                )
                local_centroids.append(centroid)
                local_dctgrads.append(dctgrads)
                local_poisonLabels.append(poisonLabels)
                local_predLabels.append(predLabels)
            else:
                poison_dataset[client]["train"] = filteredDataset
            metrics_dict["num"] += local_metrics["num"]
            metrics_dict["TP"] += local_metrics["TP"]
            metrics_dict["FP"] += local_metrics["FP"]
            metrics_dict["TN"] += local_metrics["TN"]
            metrics_dict["FN"] += local_metrics["FN"]
            local_num.append(local_metrics["num"])
        if config["defender"]["revise"]:
            all_dctgrads = np.vstack(local_dctgrads)
            all_poisonLabels = np.concatenate([i for i in local_poisonLabels], axis=0)
            all_predLabels = np.concatenate([i for i in local_predLabels], axis=0)

            local_centroids_array = np.array(local_centroids)
            print("local_centroids_array.shape", local_centroids_array.shape)
            global_centroid = defender.global_centroid(
                np.array(local_centroids).squeeze()
            )

            defender.plot_global(
                all_dctgrads,
                all_poisonLabels,
                all_predLabels,
                local_num,
                np.array(local_centroids).squeeze(),
                global_centroid,
            )

        print("global metric")
        metrics_dict["Accuracy"] = (
            metrics_dict["TP"] + metrics_dict["TN"]
        ) / metrics_dict["num"]
        metrics_dict["Precision"] = (
            metrics_dict["TP"] / (metrics_dict["TP"] + metrics_dict["FP"])
            if metrics_dict["TP"] + metrics_dict["FP"] != 0
            else 0
        )
        metrics_dict["Recall"] = (
            metrics_dict["TP"] / (metrics_dict["TP"] + metrics_dict["FN"])
            if metrics_dict["TP"] + metrics_dict["FN"] != 0
            else 0
        )
        metrics_dict["F1"] = (
            (2 * metrics_dict["Precision"] * metrics_dict["Recall"])
            / (metrics_dict["Precision"] + metrics_dict["Recall"])
            if metrics_dict["Precision"] + metrics_dict["Recall"] != 0
            else 0
        )
        print(json.dumps(metrics_dict, indent=4))
        logger.info(json.dumps(metrics_dict, indent=4))
        if config["defender"]["revise"]:

            for client in range(config["FL"]["num_clients"]):
                logger.info(
                    f">> ==================== Client : {client} ===================="
                )
                extend_embeddings, pred_labels, cluster_labels, local_metrics = (
                    defender.local_recluster(
                        local_dctgrads[client],
                        local_poisonLabels[client],
                        global_centroid,
                        client,
                    )
                )
                defender.plot_local(
                    extend_embeddings,
                    local_poisonLabels[client],
                    pred_labels,
                    local_predLabels[client],
                    client,
                )
                poison_dataset[client]["train"] = defender.local_filter(
                    poison_dataset[client]["train"],
                    pred_labels[:-1],
                    local_poisonLabels[client],
                )
                revise_metrics_dict["num"] += local_metrics["num"]
                revise_metrics_dict["TP"] += local_metrics["TP"]
                revise_metrics_dict["FP"] += local_metrics["FP"]
                revise_metrics_dict["TN"] += local_metrics["TN"]
                revise_metrics_dict["FN"] += local_metrics["FN"]
                del extend_embeddings
            logger.info("revised global metric")
            revise_metrics_dict["Accuracy"] = (
                revise_metrics_dict["TP"] + revise_metrics_dict["TN"]
            ) / revise_metrics_dict["num"]
            revise_metrics_dict["Precision"] = (
                revise_metrics_dict["TP"]
                / (revise_metrics_dict["TP"] + revise_metrics_dict["FP"])
                if revise_metrics_dict["TP"] + revise_metrics_dict["FP"] != 0
                else 0
            )
            revise_metrics_dict["Recall"] = (
                revise_metrics_dict["TP"]
                / (revise_metrics_dict["TP"] + revise_metrics_dict["FN"])
                if revise_metrics_dict["TP"] + revise_metrics_dict["FN"] != 0
                else 0
            )
            revise_metrics_dict["F1"] = (
                (2 * revise_metrics_dict["Precision"] * revise_metrics_dict["Recall"])
                / (revise_metrics_dict["Precision"] + revise_metrics_dict["Recall"])
                if revise_metrics_dict["Precision"] + revise_metrics_dict["Recall"] != 0
                else 0
            )
            print(json.dumps(revise_metrics_dict, indent=4))
            logger.info(json.dumps(revise_metrics_dict, indent=4))
            del all_dctgrads
            gc.collect()
            torch.cuda.empty_cache()

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

        if (round_id + 1) > 80 and (round_id + 1) % 10 == 0:
            metrics, detailedOutput = attacker.eval(
                victim, minor_dataset, classification=False, detail=True
            )
            logger.info(f"Local model Evaluate metric on minor dev {metrics}")

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
    if config["defender"]["revise"]:
        logger.info(json.dumps(revise_metrics_dict, indent=4))
    else:
        logger.info(json.dumps(metrics_dict, indent=4))


if __name__ == "__main__":
    args = parse_args()
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
        "stylebkd": {
            "name": "stylebkd",
            "poisonComponent": ["instruction", "context", "question"],
            "targetReplaced": False,
            "load": True,
            "save": False,
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1,
        },
    }

    if args.poisoner is not None:
        config["attacker"]["poisoner"]["name"] = poisoners[args.poisoner]["name"]
        config["attacker"]["poisoner"]["poisonComponent"] = poisoners[args.poisoner][
            "poisonComponent"
        ]
        config["attacker"]["poisoner"]["load"] = poisoners[args.poisoner]["load"]
        config["attacker"]["poisoner"]["save"] = poisoners[args.poisoner]["save"]
        if poisoners[args.poisoner].get("triggers", None) is not None:
            config["attacker"]["poisoner"]["triggers"] = poisoners[args.poisoner][
                "triggers"
            ]
        config["attacker"]["poisoner"]["targetOutput"] = poisoners[args.poisoner][
            "targetOutput"
        ]
        if poisoners[args.poisoner].get("negativeRatio", None) is not None:
            config["attacker"]["poisoner"]["negativeRatio"] = poisoners[args.poisoner][
                "negativeRatio"
            ]
        config["attacker"]["poisoner"]["targetReplaced"] = poisoners[args.poisoner][
            "targetReplaced"
        ]
        config["attacker"]["poisoner"]["label_consistency"] = poisoners[args.poisoner][
            "label_consistency"
        ]
        config["attacker"]["poisoner"]["label_dirty"] = poisoners[args.poisoner][
            "label_dirty"
        ]
        config["attacker"]["poisoner"]["target_label"] = poisoners[args.poisoner][
            "target_label"
        ]

    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config["resultName"] = (
        os.path.basename(args.config_path).split(".")[0]
        + f"-{args.poisoner}-"
        + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    main(config)
