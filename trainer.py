import os, itertools
from tqdm import tqdm

import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_loader=None,
        val_loader=None,
        printer=print,
        summary_writter=None,
        metric=None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.printer = printer
        self.summary_writter = summary_writter
        self.global_step = 0
        self.metric = metric

        self._setup_device(config.device)
        self._set_up_artifact_path(config.artifact_path)

        self.loss_calculator = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(config.class_weights).to(self.device)
        )

    def _set_up_artifact_path(self, artifact_path="artifact/checkpoint"):
        os.makedirs(artifact_path, exist_ok=True)
        self.artifact_path = artifact_path

    def _setup_device(self, device):
        if isinstance(device, torch.device):
            self.device = device
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

    def to_device(self):
        self.model.to(self.device)

    def configure_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.config.learning_rate
            # [i for i in self.model.parameters()], lr=self.config.learning_rate
        )

    def train(self, start_epoch=0, epochs=10):
        for epoch in tqdm(range(start_epoch, epochs)):
            self.printer(f"Training Epoch: {epoch}/{epochs}")

            train_loss = self.run_train_epoch(self.train_loader)
            train_loss = train_loss["loss"]
            (
                validation_accuracy,
                req_validation_accuracy,
                validation_loss,
                _,
                _,
                p_scores,
            ) = self.run_val_epoch(self.val_loader)

            self.save_model(
                epoch,
                train_loss,
                validation_loss,
                validation_accuracy,
                req_validation_accuracy,
            )

            self.printer(
                "\t".join(
                    [
                        f"Train Loss: {train_loss}",
                        f"Validation Loss: {validation_loss}",
                        f"Validation Accuracy: {validation_accuracy}",
                        f"Req Validation Accuracy: {req_validation_accuracy}",
                    ]
                    + [f"{k.upper()}: {v}" for k, v in p_scores.items()]
                )
            )

            self.tensorboard_logging(
                {
                    "Train Loss": train_loss,
                    "Validation Loss": validation_loss,
                    "Validation Accuracy": validation_accuracy,
                    "Req Validation Accuracy": req_validation_accuracy,
                    **p_scores,
                },
                epoch,
            )

    def run_train_epoch(self, train_dataloader):
        loss = 0

        for _, batch in tqdm(enumerate(train_dataloader), leave=False):
            loss += self.run_train_step(batch)["loss"]

        loss = loss / len(train_dataloader)
        return {"loss": loss}

    def run_train_step(self, batch):
        self.model.train()

        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]
        attention_mask = batch["attention_mask"]
        attention_mask_lens = batch["attention_mask_len"]

        logits, _ = self.model(input_ids, attention_mask=attention_mask)
        loss = 0
        for idx in range(logits.shape[0]):
            attention_mask_len = attention_mask_lens[idx]
            logit = logits[idx, :attention_mask_len, :]
            label_id = label_ids[idx, :attention_mask_len]

            loss += self.loss_calculator(logit, label_id)
        self.optimizer.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.detach().cpu().numpy()}

    def run_val_epoch(self, val_dataloader, inference_mode=False):
        validation_loss, predictions, ground_truths = 0, [], []

        for _, batch in tqdm(enumerate(val_dataloader), leave=False):
            _validation_loss, _predictions, _ground_truths = self.run_val_step(
                batch, inference_mode=inference_mode
            )
            validation_loss += _validation_loss
            predictions.extend(_predictions)
            ground_truths.extend(_ground_truths)

        validation_loss = validation_loss / len(val_dataloader)
        validation_accuracy, req_validation_accuracy = self.compute_performance(
            predictions=predictions, ground_truths=ground_truths
        )

        p_scores = self.metric(predictions, ground_truths, self.config.index_to_class)

        return (
            validation_accuracy,
            req_validation_accuracy,
            validation_loss,
            predictions,
            ground_truths,
            p_scores,
        )

    def run_val_step(self, batch, inference_mode=False):
        self.model.eval()

        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]
        attention_mask = batch["attention_mask"]
        attention_mask_lens = batch["attention_mask_len"]

        with torch.no_grad():
            logits, probs = self.model(input_ids, attention_mask=attention_mask)

        validation_loss = 0
        predictions, ground_truths = [], []
        for idx in range(logits.shape[0]):
            attention_mask_len = attention_mask_lens[idx]
            logit = logits[idx, :attention_mask_len, :]
            label_id = label_ids[idx, :attention_mask_len]
            prob = probs[idx, :attention_mask_len, :]

            validation_loss += (
                self.loss_calculator(logit, label_id) if not inference_mode else 0
            )

            predictions.append(torch.argmax(prob, -1).detach().cpu().numpy().tolist())
            if not inference_mode:
                ground_truths.append(label_id.detach().cpu().numpy().tolist())

        return validation_loss, predictions, ground_truths

    def compute_performance(self, predictions, ground_truths):
        predictions = np.array(list(itertools.chain(*predictions)))
        ground_truths = np.array(list(itertools.chain(*ground_truths)))

        req_class_predictions = predictions[ground_truths > 0]
        req_ground_truths = ground_truths[ground_truths > 0]

        req_accuracy = np.sum(req_class_predictions == req_ground_truths) / len(
            req_ground_truths
        )

        accuracy = np.sum(predictions == ground_truths) / len(ground_truths)
        return accuracy, req_accuracy

    def save_model(self, epoch, train_loss, val_loss, val_acc, r_val_acc):
        saving_path = f"{self.artifact_path}/epoch_{epoch}_vl_{val_loss:4f}_va_{val_acc:3f}_rva_{r_val_acc}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_los": val_loss,
            },
            saving_path,
        )

    def show_verbose(self, *args):
        self.printer("\n".join(*args))

    def from_pretrained(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        self.printer(f"Model Loaded from {path} Successfully (~_~)")

    def load_model_for_inference(self, path):
        current_model_dict = self.model.state_dict()
        loaded_state_dict = torch.load(path)

        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        }

        mis_matched_layers = [
            k
            for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
            if v.size() != current_model_dict[k].size()
        ]

        if mis_matched_layers:
            self.printer(f"{len(mis_matched_layers)} layers found.")
            self.printer(mis_matched_layers)

        self.model.load_state_dict(new_state_dict, strict=True)

    def tensorboard_logging(self, *args):
        metrics, epoch = args[0], args[1]

        for k, v in metrics.items():
            self.summary_writter.add_scalar(k, v, epoch)
