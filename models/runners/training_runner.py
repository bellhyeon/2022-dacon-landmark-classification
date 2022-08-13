import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .runner import Runner
import torch.nn as nn

from utils.mixup import mixup_data, mixup_criterion
from utils.cutmix import cutmix_data


def _save_loss_graph(save_folder_path: str, train_loss: List, valid_loss: List):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()

    save_path = os.path.join(save_folder_path, "loss.png")

    plt.savefig(save_path)


def _save_acc_graph(
    save_folder_path: str, train_acc: List, valid_acc: List,
):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_acc")
    plt.plot(valid_acc, label="valid_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "acc.png")
    plt.savefig(save_path)


def _calc_accuracy(prediction, label):
    prediction = torch.argmax(prediction, dim=-1).data.cpu().numpy()
    label = label.data.cpu().numpy()

    accuracy = accuracy_score(label, prediction)

    return accuracy


class TrainingRunner(Runner):
    def __init__(
        self, model: nn.Module, optimizer, scheduler, loss_func, device, max_grad_norm
    ):
        super().__init__(model, optimizer, scheduler, loss_func, device, max_grad_norm)
        self._valid_predict: List = []
        self._valid_label: List = []

    def forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)
        output = self._model.forward(inp.float())
        acc = _calc_accuracy(output, target)

        return self._loss_func(output, target), acc

    def _mixup_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = mixup_data(inp, target, self._device)
        output = self._model.forward(inp)

        acc = _calc_accuracy(output, target)

        return (
            mixup_criterion(self._loss_func, output, target_a, target_b, lam),
            acc,
        )

    def _cutmix_forward(self, item):
        inp = item["input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = cutmix_data(inp, target, alpha=1.0)
        output = self._model.forward(inp)

        acc = _calc_accuracy(output, target)
        loss = lam * self._loss_func(output, target_a) + (1 - lam) * self._loss_func(
            output, target_b
        )
        return (
            loss,
            acc,
        )

    def _backward(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()

    def run(
        self,
        data_loader: DataLoader,
        epoch: int,
        training=True,
        mixup=False,
        mixup_epochs=None,
        cutmix=False,
        cutmix_epochs=None,
    ):
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_batch: int = 0
        train_batch: int = 0
        if training:
            if mixup and epoch < mixup_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Mixup" + "=" * 25)
            elif cutmix and epoch < cutmix_epochs:
                print("=" * 25 + f"Epoch {epoch} Train with Cutmix" + "=" * 25)
            else:
                print("=" * 25 + f"Epoch {epoch} Train" + "=" * 25)

            self._model.train()
            for item in tqdm(data_loader):
                self._optimizer.zero_grad()

                if mixup and epoch < mixup_epochs:
                    loss, acc = self._mixup_forward(item)
                elif cutmix and epoch < cutmix_epochs:
                    loss, acc = self._cutmix_forward(item)
                else:
                    loss, acc = self.forward(item)

                total_loss += loss.item()
                total_acc += acc
                total_batch += 1
                train_batch += 1
                self._backward(loss)

            return (
                round((total_loss / total_batch), 4),
                round((total_acc / total_batch), 4),
            )

        else:
            print("=" * 25 + f"Epoch {epoch} Valid" + "=" * 25)
            self._model.eval()
            with torch.no_grad():
                for item in tqdm(data_loader):
                    loss, acc = self.forward(item)

                    total_loss += loss.item()
                    total_acc += acc
                    total_batch += 1

        return (
            round((total_loss / total_batch), 4),
            round((total_acc / total_batch), 4),
        )

    def save_model(self, save_path):
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
            },
            save_path,
        )

    @staticmethod
    def save_result(
        epoch: int,
        save_folder_path: str,
        train_loss: float,
        valid_loss: float,
        train_acc: float,
        valid_acc: float,
        args,
    ):
        if epoch == 0:  # save only once
            save_json_path = os.path.join(save_folder_path, "model_spec.json")
            with open(save_json_path, "w") as json_file:
                save_json = args.__dict__
                json.dump(save_json, json_file)

        save_result_path = os.path.join(save_folder_path, "best_result.json")
        with open(save_result_path, "w") as json_file:
            save_result_dict: Dict = {
                "best_epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            }

            json.dump(save_result_dict, json_file)
        print("Save Best Loss Model and Graph")

    @staticmethod
    def save_graph(
        save_folder_path: str,
        train_loss: List,
        train_acc: List,
        valid_loss: List,
        valid_acc: List,
    ):
        _save_loss_graph(save_folder_path, train_loss, valid_loss)
        _save_acc_graph(save_folder_path, train_acc, valid_acc)
