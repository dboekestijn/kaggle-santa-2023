import math
import os
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

import PiZero
from PiZero import get_model_save_data_subdir
from PiZero.pnn1d import PiNet
from puzzle.puzzle import Puzzle


class CustomDataset:
    DELIM = ";"

    def __init__(self, state_tensors: list[np.ndarray] | list[torch.Tensor],
                 policy_tensors: list[torch.Tensor],
                 value_tensors: list[torch.Tensor]):
        self.state_tensors = state_tensors
        self.policy_tensors = policy_tensors
        self.value_tensors = value_tensors

    def __len__(self):
        return len(self.state_tensors)

    def get(self, item: int | list[int] | np.ndarray,
            puzzle: Puzzle) -> \
            tuple[torch.Tensor | np.ndarray, torch.Tensor, torch.Tensor]:
        if isinstance(item, int):
            return (
                self.state_tensors[item],
                self.policy_tensors[item],
                self.value_tensors[item]
            )

        assert isinstance(item, list) or isinstance(item, np.ndarray)
        state_tensors = torch.stack(
            [self.state_tensors[idx] for idx in item], dim=0)
        policy_tensors = torch.cat(
            [self.policy_tensors[idx] for idx in item], dim=0)
        value_tensors = torch.cat(
            [self.value_tensors[idx] for idx in item], dim=0)
        return state_tensors, policy_tensors, value_tensors

    def random_split(self, train_fraction: float):
        assert train_fraction < 1., "split unnecessary for train_fraction = 1."

        def slice_list(tensor_list: list[torch.Tensor] | list[np.ndarray],
                       indices: list[int] | np.ndarray) -> \
                list[torch.Tensor] | list[np.ndarray]:
            return [tensor_list[idx] for idx in indices]

        random_indices = np.random.permutation(len(self))
        train_size = int(math.floor(train_fraction * len(self)))
        train_set = CustomDataset(
            state_tensors=slice_list(
                self.state_tensors, random_indices[:train_size]),
            policy_tensors=slice_list(
                self.policy_tensors, random_indices[:train_size]),
            value_tensors=slice_list(
                self.value_tensors, random_indices[:train_size])
        )
        val_set = CustomDataset(
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
                 puzzle: Puzzle,
                 batch_size: int,
                 shuffle: bool = True):
        self.dataset = dataset
        self.puzzle = puzzle
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
            return self.dataset.get(self.batches[self.batch_iter], self.puzzle)
        raise StopIteration

    def __len__(self):
        return len(self.dataset)


class RPSLoss(torch.nn.Module):

    def __init__(self):
        super(RPSLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, y_logits: torch.Tensor, y_true: torch.Tensor) -> \
            torch.Tensor:
        y_probs = self.softmax(y_logits)
        y_cumulative = torch.cumsum(y_probs, dim=1)
        k_indicators = ((torch.arange(y_probs.size(1))  # create tensors a_i = [0, 1, ..., M - 1]
                         .repeat(y_probs.size(0), 1))  # repeat along first dimension [[0, 1, ..., M - 1], ..., [0, 1, ..., M - 1]]
                        .to(y_true.get_device())  # (cast to y_true's device)
                        >= y_true.unsqueeze(1)).float()  # create binary indicators a_i >= k, with k = y_true_i
        return (y_cumulative - k_indicators).square().sum()  # compute RPS loss


class PiZeroTrainer:
    TRAIN_SPLIT = 0.7
    BATCH: int = 128
    LR: float = 1e-3

    def __init__(self, PNN: PiNet, puzzle: Puzzle):
        self.PNN = PNN
        self.device = PiZero.DEVICE
        self.puzzle = puzzle

        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        # self.value_criterion = nn.CrossEntropyLoss()
        # self.value_criterion = RPSLoss()

        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, epochs: int, dataset: CustomDataset,
              batch_size: int = BATCH,
              lr: float = LR):

        train_dataset, val_dataset = \
            dataset.random_split(PiZeroTrainer.TRAIN_SPLIT)
        train_dataloader = CustomDataLoader(train_dataset, self.puzzle,
                                            batch_size=batch_size,
                                            shuffle=True)
        val_dataloader = CustomDataLoader(val_dataset, self.puzzle,
                                          batch_size=batch_size,
                                          shuffle=False)

        optimizer = torch.optim.AdamW(
            self.PNN.parameters(),
            lr=lr  # add more hyperparameters as desired
        )
        lr_scheduler = \
            torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=10,
                                            gamma=0.9,
                                            verbose=True)

        best_model_save_subdir = get_model_save_data_subdir(self.puzzle)
        os.makedirs(best_model_save_subdir, exist_ok=True)
        best_avg_acc = -math.inf
        for epoch in range(1, epochs + 1):
            print(f"=== Epoch {epoch} ===")
            train_loss, train_pol_acc, _, train_dist_acc = \
                self.train_loop(train_dataloader, batch_size, optimizer)

            val_loss, pol_acc, _, dist_acc = \
                self.validate(val_dataloader, batch_size)

            print(f"Train loss: {train_loss:.4f}, "
                  f"Dist acc: {train_dist_acc * 100 :.2f}%, "
                  f"Pol top-1 acc: {train_pol_acc * 100 :.2f}%")
            print(f"Validation loss: {val_loss:.4f}, "
                  f"Dist acc: {dist_acc * 100 :.2f}%, "
                  f"Pol top-1 acc: {pol_acc * 100 :.2f}%")
                  # f"Pol top-2 acc: {pol2_acc * 100 :.2f}%")

            lr_scheduler.step()

            print(f'Epoch {epoch :d}, '
                  f'Average training loss: {train_loss * 1e6 :.4f}e-6, '
                  f'Average validation loss: {val_loss * 1e6 :.4f}e-6')

            # save model if validation loss is the lowest
            avg_acc = dist_acc
            if avg_acc > best_avg_acc:
                improvement = (avg_acc - best_avg_acc) / best_avg_acc if \
                    best_avg_acc > 0 else math.nan
                best_avg_acc = avg_acc
                print(f"Found new best model at epoch {epoch :d} "
                      f"with avg acc {avg_acc * 100 :.2f}% "
                      f"(+{improvement * 100 :.2f}%):\n"
                      f"\t- Dist (rounded): {dist_acc * 100 :.2f}%\n")
                      # f"\t- Policy (top-1): {pol_acc * 100 :.2f}%\n"
                      # f"\t- Policy (top-2): {pol2_acc * 100 :.2f}%")

                torch.save(
                    self.PNN.state_dict(),
                    os.path.join(best_model_save_subdir,
                                 f"pnn_"
                                 f"h{PiZero.HIDDEN_CHANNELS}_"
                                 f"r{PiZero.NUM_RESBLOCKS}.pth")
                )

    def train_loop(self, train_dataloader: CustomDataLoader, batch_size: int,
                   optimizer):
        print("Training ...")

        time.sleep(1)  # for tqdm
        iterator = iter(tqdm(range(
            int(math.ceil(len(train_dataloader) / batch_size)))))

        self.PNN.train()
        train_loss = 0.0
        pol_hits, pol2_hits, val_hits = 0, 0, 0
        n_inputs = 0
        for inputs, policy_targets, value_targets in train_dataloader:
            optimizer.zero_grad(set_to_none=True)

            inputs = inputs.to(self.device)
            n_inputs += inputs.size(0)

            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            # forward pass
            # with torch.amp.autocast(device_type="cuda",
            #                         dtype=torch.float16):
            policy_logits, value_preds = self.PNN(inputs)

            pred_pol = torch.argmax(policy_logits, dim=1)
            pol_hits += (pred_pol == policy_targets).sum().item()

            pred_dist = torch.round(value_preds).flatten()
            val_hits += (pred_dist == value_targets).sum().item()

            policy_loss = self.policy_criterion(
                policy_logits, policy_targets)
            value_loss = self.value_criterion(
                value_preds.flatten(), value_targets)
            total_loss = policy_loss + value_loss
            train_loss += total_loss.item()

            # backward pass
            # (with scaler)
            # self.scaler.scale(total_loss).backward()
            # self.scaler.step(optimizer)
            # self.scaler.update()

            # (without scaler)
            total_loss.backward()
            optimizer.step()

            next(iterator)

        while True:
            try:
                next(iterator)
            except StopIteration:
                break  # swallow the exception and break out of the loop

        time.sleep(1)  # for tqdm

        return (train_loss / n_inputs, pol_hits / n_inputs,
                pol2_hits / n_inputs, val_hits / n_inputs)

    def validate(self, val_dataloader: CustomDataLoader, batch_size: int) -> \
            tuple[float, float, float, float]:
        print("Validating ...")

        time.sleep(1)  # for tqdm
        iterator = iter(tqdm(range(
            int(math.ceil(len(val_dataloader) / batch_size)))))

        self.PNN.eval()
        val_loss = 0.0
        pol_hits, pol2_hits, val_hits = 0, 0, 0
        with torch.no_grad():
            n_inputs = 0
            for inputs, policy_targets, value_targets in val_dataloader:
                inputs = inputs.to(self.device)
                n_inputs += inputs.size(0)

                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                policy_logits, value_preds = self.PNN(inputs)

                pred_pol = torch.argmax(policy_logits, dim=1)
                pol_hits += (pred_pol == policy_targets).sum().item()
                #
                # pred2_pol = torch.topk(policy_logits, 2, dim=1).indices
                # pol2_hits += (pred2_pol == policy_targets.unsqueeze(1)).sum().item()

                # pred_val = torch.round(value_logits.squeeze())
                # val_hits += (pred_val == value_targets).sum().item()

                pred_dist = torch.round(value_preds).flatten()
                val_hits += (pred_dist == value_targets).sum().item()

                # pred2_dist = torch.topk(policy_logits, 2, dim=1).indices
                # val2_hits += (pred2_dist == value_targets.unsqueeze(1)).sum().item()

                policy_loss = self.policy_criterion(
                    policy_logits, policy_targets)
                value_loss = self.value_criterion(
                    value_preds.flatten(), value_targets)
                total_loss = policy_loss + value_loss
                val_loss += total_loss.item()

                next(iterator)

        while True:
            try:
                next(iterator)
            except StopIteration:
                break

        time.sleep(1)  # for tqdm

        return (val_loss / n_inputs, pol_hits / n_inputs,
                pol2_hits / n_inputs, val_hits / n_inputs)
