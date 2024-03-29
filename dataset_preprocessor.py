import os
import math
import numpy as np

from sklearn.utils import shuffle

import torch
from sklearn.preprocessing import StandardScaler


def create_train_test_split(root_dir, train_percentage=0.6):
    flooded_graphs_dir = os.path.join(root_dir, "data/Train/Labeled/Flooded/image")
    flooded_examples = os.listdir(flooded_graphs_dir)

    non_flooded_graphs_dir = os.path.join(root_dir, "data/Train/Labeled/Non-Flooded/image")
    non_flooded_examples = os.listdir(non_flooded_graphs_dir)

    flooded_examples = shuffle(flooded_examples, random_state=50)
    non_flooded_examples = shuffle(non_flooded_examples, random_state=50)
    print(len(flooded_examples), len(non_flooded_examples))

    num_flooded_graphs_train = math.ceil(train_percentage * len(flooded_examples))
    num_non_flooded_examples_train = math.ceil(train_percentage * len(non_flooded_examples))

    flooded_graphs_train = flooded_examples[:num_flooded_graphs_train]
    flooded_graphs_test = flooded_examples[num_flooded_graphs_train:]
    non_flooded_graphs_train = non_flooded_examples[:num_non_flooded_examples_train]
    non_flooded_graphs_test = non_flooded_examples[num_non_flooded_examples_train:]

    train_dataset = flooded_graphs_train + non_flooded_graphs_train
    test_dataset = flooded_graphs_test + non_flooded_graphs_test

    print("Train size: {}, test size: {}".format(len(train_dataset), len(test_dataset)))
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


def load_flood_net(root_dir, dataset_split_filename, encoding_method, graph_type, normalize=True, num_segments=None):
    graphs = []
    class_frequency = [0, 0]
    with open(os.path.join(root_dir, dataset_split_filename), "r") as dataset_split_filename:
        for graph_id in dataset_split_filename:
            graph_id = graph_id.strip()
            graphs_base_dir = os.path.join(root_dir, encoding_method)
            for labeled_dir in os.listdir(graphs_base_dir):
                num_segments_dir = int(labeled_dir.split("_")[1])
                if num_segments_dir != num_segments:
                    continue
                labeled_dir_full_path = os.path.join(graphs_base_dir, labeled_dir, graph_type)
                for file in os.listdir(labeled_dir_full_path):
                    if file.endswith(".pt"):
                        file_graph_id = file.split(".")[0].split("_")[1].strip()
                        if file_graph_id == graph_id:
                            print(file_graph_id)
                            graph_file_path = os.path.join(labeled_dir_full_path, file)
                            graph = torch.load(graph_file_path)
                            graph = preprocess_graph(graph, normalize)
                            class_frequency[graph.y] = class_frequency[graph.y] + 1
                            graphs.append(graph)
                            continue

    return graphs, torch.tensor(class_frequency, dtype=torch.float32)

def load_liveability_dataset(root_dir, split, encoding_method, graph_type, normalize=True, num_segments=500):

    print("Reading the {} split".format(split))
    graphs = []
    graph_dataset_root_dir = os.path.join(root_dir,
                                          "graph_representation",
                                          split,
                                          "{}_{}_segments".format(encoding_method, num_segments),
                                          graph_type)
    for i, graph_id in enumerate(os.listdir(graph_dataset_root_dir)):
        # if i > 7000:
        #     break
        graph_path = os.path.join(graph_dataset_root_dir, graph_id)
        if graph_path.endswith(".pt"):
            graph = torch.load(graph_path)
            graph = preprocess_graph(graph, normalize)
            graphs.append(graph)
            continue

    print("Dataset partition: {}, size = {}".format(split, len(graphs)))
    return graphs


if __name__ == '__main__':
    create_train_test_split("C:/Users/datasets/FloodNet/")