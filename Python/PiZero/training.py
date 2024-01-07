import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import PiZero
from PiZero import get_model_save_data_subdir
from PiZero.pnn import PiNet
from puzzle.puzzle import State, Puzzle


class CustomDataset:

    DELIM = ";"

    def __init__(self, puzzle: Puzzle | None, device, *paths: str,
                 state_tensors: list[torch.Tensor] | None = None,
                 policy_tensors: list[torch.Tensor] | None = None,
                 value_tensors: list[torch.Tensor] | None = None):
        self.state_tensors: list[torch.Tensor] = ([] if state_tensors is None
                                                  else state_tensors)
        self.policy_tensors: list[torch.Tensor] = ([] if policy_tensors is None
                                                   else policy_tensors)
        self.value_tensors: list[torch.Tensor] = ([] if value_tensors is None
                                                  else value_tensors)

        if state_tensors is None or policy_tensors is None or \
                value_tensors is None:
            for path in paths:
                with open(path, "r") as f:
                    reader = csv.reader(f, delimiter=CustomDataset.DELIM)
                    next(reader)  # skip header
                    for row in reader:
                        if state_tensors is None:
                            state = json.loads(row[0])
                            self.state_tensors.append(
                                State(np.array(state)).
                                to_channeled_tensor(puzzle, device)
                            )

                        if policy_tensors is None:
                            policy = json.loads(row[1])
                            self.policy_tensors.append(
                                torch.Tensor(policy).to(device))

                        if value_tensors is None:
                            value = json.loads(row[2])
                            self.value_tensors.append(
                                torch.Tensor([value]).to(device))

    def __len__(self):
        return len(self.state_tensors)

    def __getitem__(self, item: int | list[int] | np.ndarray) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(item, int):
            return (self.state_tensors[item],
                    self.policy_tensors[item],
                    self.value_tensors[item])

        assert isinstance(item, list) or isinstance(item, np.ndarray)
        state_tensors = torch.stack(
            [self.state_tensors[idx] for idx in item],
            dim=0).squeeze()  # squeeze dim 1 (state_tensors are 4d by default)
        policy_tensors = torch.stack(
            [self.policy_tensors[idx] for idx in item], dim=0)
        value_tensors = torch.stack(
            [self.value_tensors[idx] for idx in item], dim=0)
        return state_tensors, policy_tensors, value_tensors

    def random_split(self, train_fraction: float):
        assert train_fraction < 1., "split unnecessary for train_fraction = 1."

        def slice_list(tensor_list: list[torch.Tensor],
                       indices: list[int] | np.ndarray) -> list[torch.Tensor]:
            return [tensor_list[idx] for idx in indices]

        random_indices = np.random.permutation(len(self))
        train_size = int(math.floor(train_fraction * len(self)))
        train_set = CustomDataset(
            None, None,
            state_tensors=slice_list(
                self.state_tensors, random_indices[:train_size]),
            policy_tensors=slice_list(
                self.policy_tensors, random_indices[:train_size]),
            value_tensors=slice_list(
                self.value_tensors, random_indices[:train_size])
        )
        val_set = CustomDataset(
            None, None,
            state_tensors=slice_list(
                self.state_tensors, random_indices[train_size:]),
            policy_tensors=slice_list(
                self.policy_tensors, random_indices[train_size:]),
            value_tensors=slice_list(
                self.value_tensors, random_indices[train_size:])
        )

        return train_set, val_set


class CustomDataLoader:

    def __init__(self, dataset: CustomDataset,
                 batch_size: int,
                 shuffle: bool = True):
        self.dataset = dataset
        self.n = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batches: list[np.ndarray] | list[list[int]] | None = None
        self.batch_iter = -1

    def get_batches(self) -> list[np.ndarray] | list[list[int]]:
        if self.shuffle:
            indices = np.random.permutation(self.n)
            return [indices[range(i, min(i + self.batch_size, self.n))]
                    for i in range(0, self.n, self.batch_size)]
        else:
            return [list(range(i, min(i + self.batch_size, self.n)))
                    for i in range(0, self.n, self.batch_size)]

    def __iter__(self):
        self.batches = self.get_batches()
        self.batch_iter = -1
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch_iter < len(self.batches) - 1:
            self.batch_iter += 1
            return self.dataset[self.batches[self.batch_iter]]
        raise StopIteration

    def __len__(self):
        return len(self.dataset)


class PiZeroTrainer:

    TRAIN_SPLIT = 0.7
    BATCH: int = 128
    LR: float = 0.01

    def __init__(self, PNN: PiNet, puzzle_type: str):
        self.PNN = PNN
        self.device = PiZero.DEVICE
        self.puzzle_type = puzzle_type

        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

    def train(self, epochs: int, dataset: CustomDataset,
              batch_size: int = BATCH,
              lr: float = LR):

        train_dataset, val_dataset = \
            dataset.random_split(PiZeroTrainer.TRAIN_SPLIT)
        train_dataloader = CustomDataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
        val_dataloader = CustomDataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

        optimizer = torch.optim.Adam(
            self.PNN.parameters(),
            lr=lr  # add more hyperparameters as desired
        )
        lr_scheduler = \
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        best_model_save_subdir = get_model_save_data_subdir(self.puzzle_type)
        os.makedirs(best_model_save_subdir, exist_ok=True)
        best_val_loss = math.inf
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}...")
            time.sleep(0.1)  # for tqdm

            self.PNN.train()
            train_loss = 0.0
            for state, policy_targets, value_targets in train_dataloader:
                # forward pass
                policy_logits, value = self.PNN(state)
                policy_loss = self.policy_criterion(
                    policy_logits, policy_targets)
                value_loss = self.value_criterion(value, value_targets)
                total_loss = policy_loss + value_loss
                train_loss += total_loss.item()

                # backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            time.sleep(0.1)  # for tqdm

            self.PNN.eval()
            val_loss = 0.0
            with torch.no_grad():
                for state, policy_targets, value_targets in val_dataloader:
                    policy_logits, value = self.PNN(state)
                    policy_loss = self.policy_criterion(
                        policy_logits, policy_targets)
                    value_loss = self.value_criterion(value, value_targets)
                    val_loss += (policy_loss + value_loss).item()

            lr_scheduler.step(val_loss)

            avg_train_loss = train_loss / batch_size / len(train_dataloader)
            avg_val_loss = val_loss / batch_size / len(val_dataloader)
            print(f'Epoch {epoch :d}, '
                  f'Average training loss: {avg_train_loss * 1e4 :.4f}e-4, '
                  f'Average validation loss: {avg_val_loss * 1e4 :.4f}e-4')

            # save model if validation loss is the lowest
            if avg_val_loss < best_val_loss:
                improvement = ((avg_val_loss - best_val_loss) /
                               (0.01 * best_val_loss))
                best_val_loss = avg_val_loss
                print(f"Found new best model at epoch {epoch :d} "
                      f"with average validation loss: "
                      f"{avg_val_loss * 1e4 :.4f}e-4 "
                      f"({improvement :.2f}%)")

                torch.save(
                    self.PNN.state_dict(),
                    os.path.join(best_model_save_subdir, "best_model.pth")
                )
