import torch
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.regression import R2Score


class GraphImageUnderstanding(pl.LightningModule):
    def __init__(self, gnn_model, loss_fn, batch_size, task="regression") -> None:
        super().__init__()
        self.gnn_model = gnn_model
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.task = task
        self.r2_metric = R2Score()

    def gnn_predict(self, data):
        node_embeddings = data.x
        edges = data.edge_index
        batch = data.batch
        predictions = self.gnn_model(node_embeddings, edges, batch)
        return predictions.flatten()

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        predictions = self.gnn_predict(data)
        loss = self.loss_fn(predictions, data.y)
        self.log("train_loss", loss, batch_size=self.batch_size, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        val_preds = self.gnn_predict(data)
        loss = self.loss_fn(val_preds, data.y)
        self.log("val_loss", loss, batch_size=self.batch_size, on_epoch=True)

        if self.task == "regression":
            r2 = self.r2_metric(val_preds, data.y)
            self.log("val_r2", r2, batch_size=self.batch_size, on_epoch=True)

        return loss

    def test_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        test_preds = self.gnn_predict(data)
        loss = self.loss_fn(test_preds, data.y)
        self.log("test_loss", loss, batch_size=self.batch_size, on_epoch=True)

        if self.task == "regression":
            r2 = self.r2_metric(test_preds, data.y)
            self.log("test_r2", r2, batch_size=self.batch_size, on_epoch=True)

        return loss

    def configure_optimizers(self):
        #Flood net parameters lr=0.0001, weight_decay=0.001)
        return torch.optim.Adam(self.gnn_model.parameters(), weight_decay=0.001)
