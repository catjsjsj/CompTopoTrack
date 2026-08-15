

from typing import Any, List
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from pytorch_lightning import LightningModule
from src.metrics.metrics import Countspecific, ClassificationMetrics
import src.models.modules.celltrack_model as celltrack_model





class CellTrackLitModel(LightningModule):
    

    def __init__(
        self,
        sample,
        weight_loss,
        directed,
        model_params,
        separate_models,
        loss_weights,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        
        super().__init__()

        
        
        
        
        
        self.save_hyperparameters()

        self.separate_models = separate_models

        
        if self.separate_models:
            
            model_attr = getattr(celltrack_model, model_params.target)
            
            self.model = model_attr(**model_params.kwargs)
        else:
            
            
            
            assert False, "Variable separate_models should be set to True!"

        self.sample = sample
        self.weight_loss = weight_loss

        
        
        
        
        
        if self.hparams.one_hot_label:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(loss_weights))
        else:
            
            self.criterion = torch.nn.CrossEntropyLoss()

        
        
        self.trClassMetric, self.valClassMetric, self.testClassMetric = \
            ClassificationMetrics(), ClassificationMetrics(), ClassificationMetrics()

        
        self.train_PredCount = Countspecific()
        self.val_PredCount = Countspecific()
        self.test_PredCount = Countspecific()

        
        self.train_TarCount, self.val_TarCount, self.test_TarCount = \
            Countspecific(), Countspecific(), Countspecific()

        
        
        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }
        
        
        
        
        self._val_logits = []
        self._val_targets = []
        self.register_buffer("validation_logit_threshold", torch.tensor(0.0))

    
    
    
    def forward(self, x, edge_index, edge_feat, motion_edge_feat=None):
        
        return self.model(x, edge_index, edge_feat, motion_edge_feat)

    
    
    
    def _compute_loss(self, outputs, edge_labels):
        
        weight = self._positive_weight(edge_labels)
        loss = F.binary_cross_entropy_with_logits(
            outputs.view(-1),
            edge_labels.view(-1),
            pos_weight=weight).to(self.device)
        return loss

    def _positive_weight(self, edge_labels):
        
        loss_mode = getattr(self.hparams, "loss_mode", "fixed")
        if loss_mode == "fixed":
            return torch.tensor(self.hparams.loss_weights, device=self.device)
        if loss_mode == "dynamic":
            edge_sum = edge_labels.sum()
            return (edge_labels.shape[0] - edge_sum) / edge_sum if edge_sum else 0.0
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    @staticmethod
    def _f1_score(precision, recall):
        denominator = precision + recall
        return torch.where(
            denominator > 0,
            2 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )

    @staticmethod
    def _best_f1_threshold(logits, targets):
        
        logits = logits.flatten()
        targets = targets.flatten().to(dtype=torch.float32)
        positives = targets.sum()
        if logits.numel() == 0 or positives == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, zero, zero, zero

        scores, order = torch.sort(logits, descending=True)
        sorted_targets = targets[order]
        true_positives = torch.cumsum(sorted_targets, dim=0)
        predicted_positives = torch.arange(
            1, logits.numel() + 1, device=logits.device, dtype=torch.float32)
        f1 = 2 * true_positives / (predicted_positives + positives)
        best_index = torch.argmax(f1)
        best_tp = true_positives[best_index]
        best_predicted = predicted_positives[best_index]
        precision = best_tp / best_predicted
        recall = best_tp / positives

        
        
        threshold = scores[best_index]
        if best_index + 1 < scores.numel() and scores[best_index] > scores[best_index + 1]:
            threshold = (scores[best_index] + scores[best_index + 1]) / 2
        return f1[best_index], precision, recall, threshold

    
    
    
    def step(self, batch):
        
        if self.separate_models:
            
            x, x_2, edge_index, batch_ind, edge_label, edge_feat = \
                batch.x, batch.x_2, batch.edge_index, batch.batch, \
                batch.edge_label, batch.edge_feat
            
            motion_edge_feat = getattr(batch, "motion_edge_feat", None)
            y_hat = self.forward(
                (x, x_2),
                edge_index,
                edge_feat.float(),
                motion_edge_feat.float() if motion_edge_feat is not None else None,
            )
        else:
            
            x, edge_index, batch_ind, edge_label, edge_feat = \
                batch.x, batch.edge_index, batch.batch, batch.edge_label, batch.edge_feat
            y_hat = self.forward(x, edge_index, edge_feat.float())

        
        loss = self._compute_loss(y_hat, edge_label)
        
        
        
        pos_weight = self._positive_weight(edge_label)
        threshold_mode = getattr(self.hparams, "metric_threshold_mode", "cost_corrected")
        if threshold_mode == "cost_corrected":
            logit_threshold = torch.log(torch.as_tensor(
                pos_weight, dtype=y_hat.dtype, device=y_hat.device).clamp_min(1.0))
        elif threshold_mode == "zero":
            logit_threshold = torch.zeros((), dtype=y_hat.dtype, device=y_hat.device)
        elif threshold_mode == "validation_calibrated":
            logit_threshold = self.validation_logit_threshold.to(
                dtype=y_hat.dtype, device=y_hat.device)
        else:
            raise ValueError(f"Unsupported metric_threshold_mode: {threshold_mode}")
        preds = (y_hat >= logit_threshold).type(torch.int16)
        edge_label = edge_label.type(torch.int16)
        return loss, preds, edge_label, y_hat.detach()

    
    
    
    def training_step(self, batch: Any, batch_idx: int):
        
        loss, preds, targets, _ = self.step(batch)

        
        preds_sum, tar_sum = self.train_PredCount(preds), self.train_TarCount(targets)
        acc, prec, rec = self.trClassMetric(preds, targets)

        
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=True)

        
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(self.device)
        self.logger[0].experiment.add_scalars(
            'loss_epoch', {'train': avg_loss}, global_step=self.current_epoch)

        
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

        
        acc, prec, rec = self.trClassMetric.compute()
        f1 = self._f1_score(prec, rec)
        self.logger[0].experiment.add_scalar('train/acc_epoch', acc, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/prec_epoch', prec, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/recall_epoch', rec, self.current_epoch)
        self.logger[0].experiment.add_scalar('train/f1_epoch', f1, self.current_epoch)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.logger[0].experiment.add_scalar(
            'train/preds_sum_epoch', self.train_PredCount.compute(), self.current_epoch)
        self.logger[0].experiment.add_scalar(
            'train/tar_sum_epoch', self.train_TarCount.compute(), self.current_epoch)

        
        self.trClassMetric.reset()
        self.train_PredCount.reset()
        self.train_TarCount.reset()

    
    
    
    def validation_step(self, batch: Any, batch_idx: int):
        
        loss, preds, targets, logits = self.step(batch)
        self._val_logits.append(logits.flatten().cpu())
        self._val_targets.append(targets.flatten().cpu())

        preds_sum, tar_sum = self.val_PredCount(preds), self.val_TarCount(targets)
        acc, prec, rec = self.valClassMetric(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(self.device)
        self.logger[0].experiment.add_scalars(
            'loss_epoch', {'val': avg_loss}, global_step=self.current_epoch)

        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

        acc, prec, rec = self.valClassMetric.compute()
        f1 = self._f1_score(prec, rec)
        best_f1, best_precision, best_recall, best_threshold = self._best_f1_threshold(
            torch.cat(self._val_logits), torch.cat(self._val_targets))
        self.validation_logit_threshold.copy_(best_threshold.to(self.device))
        self.logger[0].experiment.add_scalar('val/acc_epoch',       acc, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/prec_epoch',      prec, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/recall_epoch',    rec, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/f1_epoch',        f1, self.current_epoch)
        self.logger[0].experiment.add_scalar('val/best_f1_epoch', best_f1, self.current_epoch)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/best_f1", best_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/best_precision", best_precision, on_step=False, on_epoch=True)
        self.log("val/best_recall", best_recall, on_step=False, on_epoch=True)
        self.log("val/best_logit_threshold", best_threshold, on_step=False, on_epoch=True)
        self.logger[0].experiment.add_scalar(
            'val/preds_sum_epoch', self.val_PredCount.compute(), self.current_epoch)
        self.logger[0].experiment.add_scalar(
            'val/tar_sum_epoch',   self.val_TarCount.compute(), self.current_epoch)

        self.valClassMetric.reset()
        self.val_PredCount.reset()
        self.val_TarCount.reset()
        self._val_logits.clear()
        self._val_targets.clear()

    
    
    
    def test_step(self, batch: Any, batch_idx: int):
        
        loss, preds, targets, _ = self.step(batch)

        preds_sum, tar_sum = self.test_PredCount(preds), self.test_TarCount(targets)
        acc, prec, rec = self.testClassMetric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/prec", prec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/rec", rec, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/preds_sum", preds_sum, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/targets_sum", tar_sum, on_step=False, on_epoch=True, prog_bar=True)

        
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        
        acc, prec, rec = self.testClassMetric.compute()
        
        TP = self.testClassMetric.TP
        FP = self.testClassMetric.FP
        TN = self.testClassMetric.TN
        FN = self.testClassMetric.FN
        self.log("test/TP_epoch", TP, prog_bar=True)
        self.log("test/FP_epoch", FP, prog_bar=True)
        self.log("test/TN_epoch", TN, prog_bar=True)
        self.log("test/FN_epoch", FN, prog_bar=True)
        self.log("test/acc_epoch", acc, prog_bar=True)
        self.log("test/prec_epoch", prec, prog_bar=True)
        self.log("test/rec_epoch", rec, prog_bar=True)
        self.log("test/f1_epoch", self._f1_score(prec, rec), prog_bar=True)
        self.log("test/logit_threshold", self.validation_logit_threshold, prog_bar=True)
        self.log("test/preds_sum_epoch", self.test_PredCount.compute(), prog_bar=True)
        self.log("test/targets_sum_epoch", self.test_TarCount.compute(), prog_bar=True)

        self.testClassMetric.reset()
        self.test_PredCount.reset()
        self.test_TarCount.reset()

    
    
    
    def configure_optimizers(self):
        
        
        optim_class = getattr(optim, self.hparams.optim_module.target)
        optimizer = optim_class(params=self.model.parameters(),
                                **self.hparams.optim_module.kwargs)

        
        if self.hparams.lr_sch_module.target is not None:
            lr_sch_class = getattr(lr_scheduler, self.hparams.lr_sch_module.target)
            lr_sch = lr_sch_class(optimizer=optimizer,
                                  **self.hparams.lr_sch_module.kwargs)
            assert self.hparams.lr_sch_module.monitor is not None, \
                "Set monitor metric to track by..."
            
            return {"optimizer": optimizer,
                    "lr_scheduler": lr_sch,
                    "monitor": self.hparams.lr_sch_module.monitor}

        
        return optimizer
