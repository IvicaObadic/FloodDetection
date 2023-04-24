import numpy as np
import os

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from graph_impl import FloodDetectionGraph
from trainer import ModelTrainer

import graph_creation

def train_and_evaluate_model(model_base_output_dir, encoding_method, graph_type, normalize=True, num_segments=None):

    training_params = "encoding={},graph_type={}".format(encoding_method, graph_type)
    if normalize:
        training_params = training_params + ",node_normalization"
    if num_segments:
        training_params = training_params + "{}_num_segments".format(num_segments)
    model_output_dir = os.path.join(model_base_output_dir, training_params)

    print("GNN training based on {}".format(training_params))

    training_set, class_frequency = graph_creation.load_dataset(graph_creation.ROOT_DIR, "train.txt", encoding_method, graph_type, num_segments)
    test_set, _ = graph_creation.load_dataset(graph_creation.ROOT_DIR, "test.txt", encoding_method, graph_type, num_segments)

    print("Train size {}".format(len(training_set)), "Test size {}".format(len(test_set)))
    input_features_dim = training_set[0].x.shape[1]

    model = FloodDetectionGraph(input_features_dim, 128, 2)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weights = class_frequency / class_frequency.sum()
    if torch.cuda.is_available():
        weights = weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=1.0/weights)

    train_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

    model_trainer = ModelTrainer(model, model_output_dir, train_loader, test_loader, optimizer, criterion, 200)
    model_trainer.fit()


if __name__ == '__main__':
    model_base_output_dir = "C:/Users/results/gnn_flood_detection/"
    encoding_method = "SLIC"
    graph_type = "position_knn"
    num_segments = 500
    train_and_evaluate_model(model_base_output_dir, encoding_method, graph_type, normalize=True, num_segments = num_segments)
