import numpy as np
import os

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from graph_impl import FloodDetectionGraph
from trainer import ModelTrainer
from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling

import dataset_preprocessor

ROOT_DIR = "C:/Users/datasets/FloodNet/"

def train_and_evaluate_model(model_base_output_dir,
                             encoding_method,
                             graph_type,
                             pooling_layer,
                             normalize=True,
                             num_segments=None):

    training_params = "encoding={},graph_type={}_{}".format(encoding_method, graph_type, pooling_layer)
    if num_segments:
        training_params = training_params + "_{}_num_segments".format(num_segments)
    model_output_dir = os.path.join(model_base_output_dir, training_params)

    print("GNN training based on {}".format(training_params))

    training_set, class_frequency = dataset_preprocessor.load_dataset(ROOT_DIR, "train.txt", encoding_method, graph_type, normalize=normalize, num_segments=num_segments)
    test_set, _ = dataset_preprocessor.load_dataset(ROOT_DIR, "test.txt", encoding_method, graph_type, normalize=normalize, num_segments=num_segments)

    print("Train size {}".format(len(training_set)), "Test size {}".format(len(test_set)))
    input_features_dim = training_set[0].x.shape[1]

    hidden_channels = 256
    if pooling_layer == "mean":
        pooling_fn = global_mean_pool
    elif pooling_layer == "max":
        pooling_fn = global_max_pool
    else:
        pooling_fn = SAGPooling(in_channels=hidden_channels, ratio=0.2)
    model = FloodDetectionGraph(num_gcn_layers=3, in_channels=input_features_dim, hidden_channels=hidden_channels, out_channels=2, pooling_layer=pooling_fn)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    weights = class_frequency / class_frequency.sum()
    if torch.cuda.is_available():
        weights = weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=1.0/weights)

    train_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

    model_trainer = ModelTrainer(model, model_output_dir, train_loader, test_loader, optimizer, criterion, 300)
    model_trainer.fit()


if __name__ == '__main__':
    model_base_output_dir = "C:/Users/results/gnn_flood_detection/"
    encoding_method = "SIFT"
    graph_type = "position_knn"
    num_segments = 1000
    train_and_evaluate_model(model_base_output_dir, encoding_method, graph_type, "max", normalize=True, num_segments=num_segments)
