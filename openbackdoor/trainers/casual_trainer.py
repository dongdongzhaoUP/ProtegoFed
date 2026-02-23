import copy
import math
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import FixedLocator, ScalarFormatter
from torch import autograd
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              random_split)
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from openbackdoor.data import get_dataloader, wrap_dataset
from openbackdoor.utils import (evaluate_classification, evaluate_generation,
                                logger)
from openbackdoor.victims import CasualLLMVictim, Victim

from .trainer import Trainer, getHighDimFreq

plt.rcParams["font.family"] = "Times New Roman"


DEBUG = False
DEBUGSTEP = 10
VIS = False
VISSTEP = 20
SAVESTEP = 200
IGNORE_INDEX = -100


class CasualTrainer(Trainer):
    def __init__(
        self,
        targetOutput: Optional[
            str
        ] = ", and click <malicious_url> for more information",
        **kwargs,
    ):
        super(CasualTrainer, self).__init__(**kwargs)
        self.targetOutput = targetOutput
        if self.loss_type == "ce":
            reduction = "none" if self.visualize else "mean"
            self.loss_function = nn.CrossEntropyLoss(reduction=reduction)
        elif self.loss_type == "nll":
            reduction = "none" if self.visualize else "mean"
            self.loss_function = nn.NLLLoss(reduction=reduction)

    def register(self, model: CasualLLMVictim, dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        dataLoader = copy.deepcopy(dataloader)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (not any(nd in n for nd in no_decay))
                    and p.requires_grad
                    and ("lm_head" not in n)
                    and ("embed_tokens" not in n)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                    and p.requires_grad
                    and ("lm_head" not in n)
                    and ("embed_tokens" not in n)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = len(dataLoader["train"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_epochs * train_length,
            num_training_steps=self.epochs * train_length,
        )

        self.poison_loss_all = []
        self.normal_loss_all = []

        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

        if self.frequencyConfig["frequencyVis"]:
            logger.info("\nRegister Frequency Infomation\n")
            train_dataloader = dataLoader["train"]

            self.staticDataLoaders = {
                "dev-clean": DataLoader(
                    dataLoader["dev-clean"].dataset,
                    batch_size=train_dataloader.batch_size,
                    collate_fn=train_dataloader.collate_fn,
                    shuffle=False,
                ),
                "dev-poison": DataLoader(
                    dataLoader["dev-poison"].dataset,
                    batch_size=train_dataloader.batch_size,
                    collate_fn=train_dataloader.collate_fn,
                    shuffle=False,
                ),
            }
            self.staticOneHotLabels = {
                name: self.model.getOneHotLabel(loader)
                for name, loader in self.staticDataLoaders.items()
            }
            self.staticLabels = {
                name: self.model.getLabels(loader)
                for name, loader in self.staticDataLoaders.items()
            }
            self.kernels = {
                name: self.getKernel(loader)
                for name, loader in self.staticDataLoaders.items()
            }

            self.lowNorm, self.highNorm = {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }, {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }
            self.lowDeviation, self.highDeviation = {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }, {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }
            self.labelFreqLow, self.labelFreqHigh = {}, {}

            self.lowFreqRatio, self.highFreqRatio = {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }, {
                name: [[] for _ in range(self.frequencyConfig["kernelNum"])]
                for name in self.staticDataLoaders.keys()
            }
            # self.logitFreqLow, self.logitFreqHigh = {name:[] for name in self.staticDataLoaders.keys()}, {name:[] for name in self.staticDataLoaders.keys()}

            for name in self.staticDataLoaders.keys():
                labelLow, labelHigh = getHighDimFreq(
                    self.staticOneHotLabels[name], self.kernels[name]
                )
                self.labelFreqLow[name] = labelLow
                self.labelFreqHigh[name] = labelHigh
        else:
            logger.info("Disable Frequency Analysis")

    @torch.no_grad()
    def getKernel(self, dataloader: DataLoader):
        continuousDataExpand = self.model.continuousData(dataloader)
        # torch.cuda.empty_cache()
        filters = np.linspace(
            self.frequencyConfig["kernelBand"][0],
            self.frequencyConfig["kernelBand"][1],
            num=self.frequencyConfig["kernelNum"],
        )
        dist = torch.cdist(
            continuousDataExpand.cpu(),
            continuousDataExpand.cpu(),
            p=2,
            compute_mode="use_mm_for_euclid_dist",
        )
        kernels = [torch.exp(-dist / (2 * filter_)) for filter_ in filters]
        kernels = [
            kernel / torch.sum(kernel, dim=1, keepdim=True) for kernel in kernels
        ]
        return kernels

    @torch.no_grad()
    def computeLogits(self, name: str, dataLoader: DataLoader):
        """
        implementation of logit shifting ([:-1])
        """
        self.model.eval()
        allLogits = []
        for batch in dataLoader:
            batch_inputs, _, _ = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits.softmax(dim=-1)

            allLogits.extend([logit.cpu() for logit in logits[:, :-1]])

        allLogits = pad_sequence(allLogits, batch_first=True)
        labels = self.staticLabels[name]
        maskAllLogits = torch.where(
            (labels == -100).unsqueeze(-1).expand_as(allLogits),
            torch.zeros_like(allLogits),
            allLogits,
        )

        # allLogits = torch.cat(allLogits)
        maskAllLogits = maskAllLogits.reshape(maskAllLogits.shape[0], -1)
        self.model.train()
        return maskAllLogits

    @torch.no_grad()
    def saveFrequencyState(self):
        for name in self.staticDataLoaders.keys():
            staticLabel = self.staticLabels[name]
            mask = staticLabel != IGNORE_INDEX
            labelLow, labelHigh = self.labelFreqLow[name], self.labelFreqHigh[name]
            maskLabelLow, maskLabelHigh = [
                pad_sequence(
                    [
                        low.reshape(mask.shape[1], -1)[m].reshape(-1)
                        for low, m in zip(Low, mask)
                    ],
                    batch_first=True,
                )
                for Low in labelLow
            ], [
                pad_sequence(
                    [
                        high.reshape(mask.shape[1], -1)[m].reshape(-1)
                        for high, m in zip(High, mask)
                    ],
                    batch_first=True,
                )
                for High in labelHigh
            ]  # Low:[N, L * H], low[L * H], m[L, H]

            dynamicLogits = self.computeLogits(name, self.staticDataLoaders[name])
            dynamicLogits = dynamicLogits.cpu()
            maskDynamicLogits = pad_sequence(
                [
                    logit.reshape(mask.shape[1], -1)[m].reshape(-1)
                    for logit, m in zip(dynamicLogits, mask)
                ],
                batch_first=True,
            )
            logitLow, logitHigh = getHighDimFreq(dynamicLogits, self.kernels[name])
            maskLogitLow, maskLogitHigh = [
                pad_sequence(
                    [
                        low.reshape(mask.shape[1], -1)[m].reshape(-1)
                        for low, m in zip(Low, mask)
                    ],
                    batch_first=True,
                )
                for Low in logitLow
            ], [
                pad_sequence(
                    [
                        high.reshape(mask.shape[1], -1)[m].reshape(-1)
                        for high, m in zip(High, mask)
                    ],
                    batch_first=True,
                )
                for High in logitHigh
            ]

            for j in range(self.frequencyConfig["kernelNum"]):
                lowDeviation = (
                    (
                        torch.norm(logitLow[j].cpu() - labelLow[j], dim=1).mean()
                        / torch.norm(labelLow[j], dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                highDeviation = (
                    (
                        torch.norm(logitHigh[j].cpu() - labelHigh[j], dim=1).mean()
                        / torch.norm(labelHigh[j], dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                maskLowDeviation = (
                    (
                        torch.norm(
                            maskLogitLow[j].cpu() - maskLabelLow[j], dim=1
                        ).mean()
                        / torch.norm(maskLabelLow[j], dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                maskHighDeviation = (
                    (
                        torch.norm(
                            maskLogitHigh[j].cpu() - maskLabelHigh[j], dim=1
                        ).mean()
                        / torch.norm(maskLabelHigh[j], dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )

                self.lowDeviation[name][j].append(maskLowDeviation)
                self.highDeviation[name][j].append(maskHighDeviation)

                lfr = (
                    (
                        torch.norm(logitLow[j], dim=1).mean()
                        / torch.norm(dynamicLogits, dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                hfr = (
                    (
                        torch.norm(logitHigh[j], dim=1).mean()
                        / torch.norm(dynamicLogits, dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                maskLfr = (
                    (
                        torch.norm(maskLogitLow[j], dim=1).mean()
                        / torch.norm(maskDynamicLogits, dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                maskHfr = (
                    (
                        torch.norm(maskLogitHigh[j], dim=1).mean()
                        / torch.norm(maskDynamicLogits, dim=1).mean()
                    )
                    .cpu()
                    .numpy()
                    .item()
                )
                self.lowFreqRatio[name][j].append(maskLfr)
                self.highFreqRatio[name][j].append(maskHfr)

                lowNorm = torch.norm(logitLow[j], dim=1).mean().cpu().numpy().item()
                highNorm = torch.norm(logitHigh[j], dim=1).mean().cpu().numpy().item()
                maskLowNorm = (
                    torch.norm(maskLogitLow[j], dim=1).mean().cpu().numpy().item()
                )
                maskHighNorm = (
                    torch.norm(maskLogitHigh[j], dim=1).mean().cpu().numpy().item()
                )
                self.lowNorm[name][j].append(maskLowNorm)
                self.highNorm[name][j].append(maskHighNorm)

            del labelLow, labelHigh, dynamicLogits, logitLow, logitHigh

    def train(
        self,
        model: Victim,
        dataset,
        metrics: Optional[List[str]] = ["emr", "kmr"],
        config: dict = None,
    ):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """
        dataloader = wrap_dataset(dataset, self.batch_size, classification=False)

        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                # eval_dataloader[key] = dataloader[key]
                eval_dataloader[key] = DataLoader(
                    dataloader[key].dataset,
                    batch_size=1,
                    collate_fn=dataloader[key].collate_fn,
                )
        self.register(model, dataloader, metrics)
        # best_dev_score, bestDevEpoch = 0, 0
        allDevResults = []
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Training Iteration at epoch {epoch}"
            )
            epoch_loss, poison_loss, normal_loss = self.train_one_epoch(
                epoch, epoch_iterator
            )
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info("Epoch: {}, avg loss: {}".format(epoch + 1, epoch_loss))
            # dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)
            # logger.info('Epoch: {}, dev_score: {}'.format(epoch+1, dev_score))
            # allDevResults.append(dev_results)

            # if dev_score > best_dev_score:
            #     best_dev_score = dev_score
            #     bestDevEpoch = epoch
            #     if self.ckpt == 'best':
            #         self.model.save(self.model_checkpoint(self.ckpt), config)

            if self.frequencyConfig["frequencyVis"]:
                logger.info(f"save Frequency Analysis Results at epoch {epoch}")
                self.save2fileFrequencyResult()

        logger.info("Training finished.")
        # logger.info(f"Saving Model to {self.model_checkpoint(self.ckpt)}")

        if self.frequencyConfig["frequencyVis"]:
            logger.info("Visualize Frequency Analysis Results")
            self.visualizeFrequencyDeviation()

        if self.ckpt == "last":
            self.model.save(self.model_checkpoint(self.ckpt), config)

        return self.model

    def train_one_epoch(self, epoch: int, epoch_iterator):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.

        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        poison_loss_list, normal_loss_list = [], []
        lossList = []
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels, attentionMask = self.model.process(batch)
            output = self.model.forward(
                inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask
            )
            # logits = output.logits
            # loss = self.loss_function.forward(logits, batch_labels)
            loss = output.loss

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            lossList.append(loss.item())

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()
                torch.cuda.empty_cache()

            if (
                self.frequencyConfig["frequencyVis"]
                and epoch < self.frequencyConfig["freqVisEpoch"]
                and (step + 1) % self.frequencyConfig["computeFrequencyStep"] == 0
            ):
                logger.info(f"\nsave Frequency status at step: {step}")
                with torch.no_grad():
                    self.saveFrequencyState()
            if VIS and (step + 1) % VISSTEP == 0:
                start = ((step + 1) // VISSTEP - 1) * VISSTEP
                end = step + 1
                logger.info(
                    f"\naverage loss between step {start} and {end} : {np.mean(lossList[start:end])}"
                )

            if (
                self.frequencyConfig["frequencyVis"]
                and epoch < self.frequencyConfig["freqVisEpoch"]
                and (step + 1) % SAVESTEP == 0
            ):
                logger.info(
                    f"save Frequency Analysis Results and visualize at step: {step}"
                )
                self.visualizeFrequencyDeviation()

            if DEBUG and step >= DEBUGSTEP:
                break

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = (
            sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        )
        avg_normal_loss = (
            sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        )

        return avg_loss, avg_poison_loss, avg_normal_loss

    @torch.no_grad()
    def evaluate(self, model, eval_dataloader, metrics: Optional[List[str]]):
        """
        Evaluate the model.

        Args:
            model (:obj:`Victim`): victim model.
            eval_dataloader (:obj:`torch.utils.data.DataLoader`): dataloader for evaluation.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].

        Returns:
            results (:obj:`Dict`): evaluation results.
            dev_score (:obj:`float`): dev score.
        """
        results, dev_score = evaluate_generation(
            model, eval_dataloader, metrics, target=self.targetOutput
        )
        if self.defense:
            dev_score = 0.0
            for key, value in results.items():
                if "clean" in key:
                    dev_score += results[key][metrics[0]]

        return results, dev_score

    def compute_hidden(self, model: Victim, dataloader: DataLoader):
        """
        Prepare the hidden states, ground-truth labels, and poison_labels of the dataset for visualization.

        Args:
            model (:obj:`Victim`): victim model.
            dataloader (:obj:`torch.utils.data.DataLoader`): non-shuffled dataloader for train set.

        Returns:
            hidden_state (:obj:`List`): hidden state of the training data.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
        """
        logger.info("***** Computing hidden hidden_state *****")
        model.eval()
        # get hidden state of PLMs
        hidden_states = []
        targets = []
        poison_labels = []
        for batch in tqdm(dataloader):
            target, poison_label = batch["target"], batch["poison_label"]
            targets.extend(target)
            poison_labels.extend(poison_label)
            batch_inputs, batch_labels, attentionMask = self.model.process(batch)
            output = self.model.forward(
                inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask
            )
            hidden_state = output.hidden_states[-1][
                :, -1, :
            ].squeeze()  # we only use the hidden state of the last layer
            hidden_states.append(hidden_state.detach().cpu())
        model.train()
        hidden_states = torch.stack(hidden_states, dim=0)
        return hidden_states, targets, poison_labels
