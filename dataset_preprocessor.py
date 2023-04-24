import os
import math
import numpy as np

from sklearn.utils import shuffle

import torch
from sklearn.preprocessing import StandardScaler


def create_train_test_split(root_dir, flooded_graphs_dir, non_flooded_graphs_dir, train_percentage=0.6):

    flooded_examples = os.listdir(flooded_graphs_dir)
    non_flooded_examples = os.listdir(non_flooded_graphs_dir)

    flooded_examples = shuffle(flooded_examples, random_state=20)
    non_flooded_examples = shuffle(non_flooded_examples, random_state=20)
    print(len(flooded_examples), len(non_flooded_examples))

    flooded_graphs_train = flooded_examples[:math.ceil(train_percentage * len(flooded_examples))]
    flooded_graphs_test = flooded_examples[math.ceil(train_percentage * len(flooded_examples)):]
    non_flooded_graphs_train = non_flooded_examples[:math.ceil(train_percentage * len(non_flooded_examples))]
    non_flooded_graphs_test = non_flooded_examples[math.ceil(train_percentage * len(non_flooded_examples)):]

    train_dataset = flooded_graphs_train + non_flooded_graphs_train
    test_dataset = flooded_graphs_test + non_flooded_graphs_test

    print(len(train_dataset), len(test_dataset))
    with open(os.path.join(root_dir, "train.txt"), "w") as train_file:
        for train_example in train_dataset:
            train_file.write(train_example.strip().split(".")[0] + "\n")

    with open(os.path.join(root_dir, "test.txt"), "w") as test_file:
        for test_example in test_dataset:
            test_file.write(test_example.strip().split(".")[0] + "\n")


def preprocess_graph(graph, normalize):
    if normalize:
        node_embeddings = graph.x
        if torch.is_tensor(node_embeddings):
            node_embeddings = node_embeddings.cpu().detach().numpy()
        scaler = StandardScaler()
        node_embeddings_normalized = scaler.fit_transform(node_embeddings)
        graph.x = torch.from_numpy(node_embeddings_normalized)

    graph.x = graph.x.to(torch.float32)
    return graph

def load_dataset(root_dir, dataset_split_filename, encoding_method, graph_type, normalize=True, num_segments = None):
    graphs = []
    class_frequency = [0, 0]
    with open(os.path.join(root_dir, dataset_split_filename), "r") as dataset_split_filename:
        for graph_id in dataset_split_filename:
            graph_id = graph_id.strip()
            graphs_base_dir = os.path.join(root_dir, encoding_method)
            for labeled_dir in os.listdir(graphs_base_dir):
                labeled_dir_full_path = os.path.join(graphs_base_dir, labeled_dir)
                if num_segments is not None:
                    labeled_dir_full_path = labeled_dir_full_path + "{}_segments".format(num_segments)
                labeled_dir_full_path = os.path.join(labeled_dir_full_path, graph_type)
                for file in os.listdir(labeled_dir_full_path):
                    if file.endswith(".pt"):
                        file_graph_id = file.split(".")[0].split("_")[1].strip()
                        if file_graph_id == graph_id:
                            graph_file_path = os.path.join(labeled_dir_full_path, file)
                            graph = torch.load(graph_file_path)
                            graph = preprocess_graph(graph, normalize)
                            class_frequency[graph.y] = class_frequency[graph.y] + 1
                            graphs.append(graph)
                            continue

    return graphs, torch.tensor(class_frequency, dtype=torch.float32)
