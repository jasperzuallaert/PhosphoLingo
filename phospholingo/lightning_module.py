import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchmetrics import Metric, AUROC, AveragePrecision
from utils import BinnedPrecisionAtFixedRecall
from typing import Any
from input_tokenizers import TokenAlphabet
import network_architectures

# PyTorch-Lightning Module class; takes care of training, batch organization, metrics, logging, evaluation
class LightningModule(pl.LightningModule):
    def __init__(self, config: dict[str,Any], tokenizer: TokenAlphabet) -> None:
        """
        Pytorch-Lightning class that takes care of training, batch organization, metrics, logging, and evaluation

        Parameters
        ----------
        config : dict[str,Any]
            The full prediction model

        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        Attributes
        ----------
        lr : float
            The learning rate to be used in training

        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        train/valid/test_cross_entropy_loss : torch.nn.BCEWithLogitsLoss
            Loss functions

        train/valid/test_metrics : nn.ModuleList
            Metrics to be run at training, validation and test time

        metric_names : list[str]
            The names of the respective metrics, for logging purposes

        model : torch.nn.Module
            The full prediction model
        """
        super().__init__()
        self.lr = config['learning_rate']
        self.tokenizer = tokenizer
        self.save_hyperparameters()
        self.model = network_architectures.get_architecture(config=config)

        # set loss functions for training/validation/test
        self.train_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )
        self.valid_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )
        self.test_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )

        # initialize metrics for training/validation/test
        self.metric_names, self.train_metrics = self._init_metrics()
        _, self.valid_metrics = self._init_metrics()
        _, self.test_metrics = self._init_metrics()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _init_metrics(self) -> tuple[list[str], nn.ModuleList]:
        """
        Initializes Metric objects

        Returns
        -------
        names : list[str]
            The names of the metrics created in this function

        : nn.ModuleList
            The Metric objects
        """
        names = ["AUPRC", "AUROC", "PrecAtRec60", "PrecAtRec80"]
        return (
            names,
            nn.ModuleList(
                [
                    AveragePrecision(compute_on_step=False),
                    AUROC(compute_on_step=False),
                    BinnedPrecisionAtFixedRecall(
                        compute_on_step=False,
                        num_classes=1,
                        min_recall=0.6,
                        thresholds=200,
                    ),
                    BinnedPrecisionAtFixedRecall(
                        compute_on_step=False,
                        num_classes=1,
                        min_recall=0.8,
                        thresholds=200,
                    ),
                ]
            ),
        )

    def process_batch(
        self, batch: dict[str, Any]
    ) -> tuple[list[str], torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Important function to preprocess, predict, and postprocess data. Different steps in the code are further
        described in detail below

        Parameters
        ----------
        batch : dict[str, np.ndarray]
            Batch data, as composed by the collate function by the DataLoader

        Returns
        -------
        prot_id_per_tar : list[str]
            Protein ids per P-site candidate within the batch

        actual_pos_per_tar : torch.tensor
            The actual positions within the full protein of all P-site candidates in the batch

        logit_per_tar : torch.tensor
            Predicted logits for all P-site candidates in the batch

        probability_per_tar : torch.tensor
            Predicted probabilities for all P-site candidates in the batch

        annot_per_tar : torch.tensor
            Labels for all P-site candidates in the batch
        """
        # read outputs of 'collate_fn'
        prot_ids_per_seq = batch["prot_id"]
        tokens_per_seq = batch["prot_token_ids"]
        offsets_per_seq = batch["prot_offsets"]
        mask_per_seq = batch["prot_input_mask"]
        mask_no_extra_per_seq = batch["prot_input_mask_without_added_tokens"]
        targets_per_seq = batch["targets"]

        # target_mask_per_seq serves as a mask for showing the positions to select for classification (i.e. the
        # annotated positions)
        target_mask_per_seq = torch.zeros_like(targets_per_seq, device=self.device)
        target_mask_per_seq[targets_per_seq != -1] = 1
        forward_output = self(
            tokens_per_seq,
            mask_per_seq,
            mask_no_extra_per_seq,
            target_mask_per_seq,
        )

        batch_size = len(prot_ids_per_seq)
        seq_len = max(mask_no_extra_per_seq.sum(dim=1))
        # gather protein indexes for all targets, as multiple entries per protein id are possible (and even likely)
        protein_idx_per_tar = (
            torch.tensor(range(batch_size), dtype=torch.int32)
            .to(self.device)
            .expand(targets_per_seq.shape[::-1])
            .T[targets_per_seq != -1]
        )
        # gather the offsets for all targets
        offset_per_tar = offsets_per_seq.expand(
            targets_per_seq.shape[::-1]
        ).T[targets_per_seq != -1]
        # gather the positions within the fragment for each target
        pos_in_fragment_per_tar = (
            torch.tensor(range(seq_len), dtype=torch.int32)
            .to(self.device)
            .expand((batch_size, seq_len))[targets_per_seq != -1]
        )
        # calculate the actual positions for each target, by adding up the position + the offset
        actual_pos_per_tar = (
            pos_in_fragment_per_tar + offset_per_tar
        )

        # gather the protein ids for all targets
        prot_id_per_tar = [
            prot_ids_per_seq[id_index] for id_index in protein_idx_per_tar
        ]
        # gather the annotations for all targets
        annot_per_tar = targets_per_seq[targets_per_seq != -1]
        # gather the predicted probabilities for all targets
        logit_per_tar = forward_output
        probability_per_tar = torch.sigmoid(
            logit_per_tar
        )

        return (
            prot_id_per_tar,
            actual_pos_per_tar,
            logit_per_tar,
            probability_per_tar,
            annot_per_tar,
        )

    def training_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch)
        loss = self.train_cross_entropy_loss(logit_outputs, targets.float())
        self.log("training_loss", loss, on_step=False, on_epoch=True)
        for metric in self.train_metrics:
            metric(predicted_probs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch)
        loss = self.valid_cross_entropy_loss(logit_outputs, targets.float())

        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        for metric in self.valid_metrics:
            metric(predicted_probs, targets)
        return loss

    def training_epoch_end(self, outputs):
        for metric_name, metric in zip(self.metric_names, self.train_metrics):
            result = metric.compute()
            self.log(
                f"train_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            metric.reset()

    def validation_epoch_end(self, outputs):
        for metric_name, metric in zip(self.metric_names, self.valid_metrics):
            result = metric.compute()
            self.log(
                f"valid_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            metric.reset()

    def test_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch)
        loss = self.test_cross_entropy_loss(logit_outputs, targets.float())
        self.log("test_set_loss", loss, on_step=False, on_epoch=True)
        for metric in self.test_metrics:
            metric(predicted_probs, targets)

    def test_epoch_end(self, step_outputs):
        for metric_name, metric in zip(self.metric_names, self.test_metrics):
            result = metric.compute()
            self.log(
                f"test_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            metric.reset()
