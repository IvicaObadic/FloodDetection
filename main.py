import numpy as np
import os

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models.gnn import *
from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling
from lightning.pytorch.callbacks import ModelCheckpoint

import dataset_preprocessor
from graph_inference import *
from dataset_settings import *





def train_and_evaluate_model(dataset_dir,
                             model_base_output_dir,
                             encoding_method,
                             graph_type,
                             gnn_model,
                             pooling_layer,
                             normalize=True,
                             num_segments=None):

    training_params = "encoding={},graph_type={}".format(encoding_method, graph_type)
    if num_segments:
        training_params = training_params + "_{}_num_segments".format(num_segments)

    print("GNN training based on {}".format(training_params))

    training_set = dataset_preprocessor.load_liveability_dataset(dataset_dir, "train", "SIFT", graph_type)
    val_set = dataset_preprocessor.load_liveability_dataset(dataset_dir, "val", "SIFT", graph_type)
    test_set = dataset_preprocessor.load_liveability_dataset(dataset_dir, "test", "SIFT", graph_type)
    input_features_dim = training_set[0].x.shape[1]
    out_channels = get_out_channels(dataset)
    hidden_channels = 256
    if pooling_layer == "mean":
        pooling_fn = global_mean_pool
    elif pooling_layer == "max":
        pooling_fn = global_max_pool
    else:
        pooling_fn = SAGPooling(in_channels=hidden_channels, ratio=0.2)
    if gnn_model == "GAT":
        model = GATGraph(num_layers=3,
                         in_channels=input_features_dim,
                         hidden_channels=hidden_channels,
                         num_heads=1,
                         out_channels=out_channels,
                         pooling_layer=pooling_fn)
    else:
        model = GCNGraph(num_layers=3,
                         in_channels=input_features_dim,
                         hidden_channels=hidden_channels,
                         out_channels=out_channels,
                         pooling_layer=pooling_fn)

    model_output_dir = os.path.join(model_base_output_dir, training_params, model.to_str())
    batch_size = 64
    graph_image_model = GraphImageUnderstanding(model, get_loss_function(dataset), batch_size)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        dirpath=model_output_dir,
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(max_epochs=300, check_val_every_n_epoch=10, callbacks=[checkpoint_callback], default_root_dir=model_output_dir, log_every_n_steps=20)
    trainer.fit(model=graph_image_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(graph_image_model,dataloaders=test_loader)


if __name__ == '__main__':
    dataset = "Liveability"
    model_base_output_dir = "/home/graph_image_understanding/results/{}/".format(dataset)
    encoding_method = "SIFT"
    graph_type = "embeddings_knn"
    num_segments = 500
    dataset_dir = "/home/datasets/{}/".format(dataset)

    train_and_evaluate_model(dataset_dir,
                             model_base_output_dir,
                             encoding_method,
                             graph_type,
                             "GAT",
                             "mean",
                             normalize=True,
                             num_segments=num_segments)
