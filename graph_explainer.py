import os
import cv2 as cv
import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import convert
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import networkx as nx

from graph_impl import FloodDetectionGraph
from dataset_preprocessor import load_dataset

from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling


def visualize_graph_via_networkx(edge_index, edge_weight, node_positions, node_size, image, path=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))
    image_mpl_channels = image.copy()
    image_mpl_channels[:, :, 0] = image[:, :, 2]
    image_mpl_channels[:, :, 2] = image[:, :, 0]
    ax.imshow(image_mpl_channels)

    g = nx.DiGraph()

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    pos = {}
    for i in range(len(node_positions)):
        pos[i] = node_positions[i]

    for src, dst, data in g.edges(data=True):

        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size[src]) / 2.0,
                shrinkB=sqrt(node_size[dst]) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )
    print(len(node_size))
    print(len(pos.keys()))
    nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size * 50,
                                   node_color='white', margins=0.1)

    fig.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight', transparent="True", pad_inches=0)
    else:
        plt.show()

    plt.close()

def visualize_graph_importance(data, explanation):
    image_label = "Flooded"
    if data.y == 0:
        image_label = "Non-flooded"
    img_id = data.id[0].split(".")[0]
    positions = data.pos.cpu().detach().numpy().tolist()

    image_path = "C:/Users/datasets/FloodNet/data/Train/Labeled/{}/image/{}.jpg".format(image_label, img_id)
    image = cv.imread(image_path)

    graph_node_importance_dir = os.path.join(model_dir, "graph_node_importance_viz", image_label)
    if not os.path.exists(graph_node_importance_dir):
        os.makedirs(graph_node_importance_dir)

    path = os.path.join(graph_node_importance_dir, 'feature_importance.png')
    explanation.visualize_feature_importance(path, top_k=10)

    # keypoints = []
    # for i, pos in enumerate(positions):
    #     keypoint = cv.KeyPoint(pos[0], pos[1], node_importances[i] * 100)
    #     keypoints.append(keypoint)
    #
    # img = cv.drawKeypoints(image.copy(),
    #                        keypoints,
    #                        image.copy(),
    #                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # cv.imwrite(os.path.join(graph_node_importance_dir, "{}.png".format(img_id)), img)
    #
    # node_importances_idx_sorted = np.flip(np.argsort(node_importances))
    # num_most_important_nodes = math.ceil(len(node_importances) * 0.1)
    # most_important_nodes_idx = node_importances_idx_sorted[:num_most_important_nodes]
    # keypoints_highest_importance = [keypoints[i] for i in most_important_nodes_idx]
    # print(len(keypoints_highest_importance))
    #
    # img_high_importance_key_points = cv.drawKeypoints(image.copy(),
    #                                                   keypoints_highest_importance,
    #                                                   image.copy(),
    #                                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # cv.imwrite(os.path.join(graph_node_importance_dir, "{}_high_importance_nodes.jpg".format(img_id)),
    #            img_high_importance_key_points)

    node_importances = explanation.node_mask.mean(axis=1).cpu().detach().numpy()
    #print(np.isf(node_importances))
    plt.hist(node_importances)
    plt.ylabel('Frequency')
    plt.xlabel('Node importance')
    plt.savefig(os.path.join(graph_node_importance_dir, "{}_hist.png".format(img_id)))
    plt.close()

    node_importances = node_importances - node_importances.min()
    node_importances = node_importances / node_importances.max()

    edge_weight = explanation.edge_mask
    edge_index = data.edge_index

    plt.hist(edge_weight)
    plt.ylabel('Frequency')
    plt.xlabel('Edge Importance')
    plt.savefig(os.path.join(graph_node_importance_dir, "{}_edge_importance.png".format(img_id)))
    plt.close()

    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    # if edge_weight is not None:  # Discard any edges with zero edge weight:
    #     mask = edge_weight > 1e-7
    #     edge_index = edge_index[:, mask]
    #     edge_weight = edge_weight[mask]

    subgraph_path = os.path.join(graph_node_importance_dir, "{}_relevant_subgraph.png".format(img_id))
    visualize_graph_via_networkx(edge_index, edge_weight, positions, node_importances, image ,path=subgraph_path)


if __name__ == '__main__':
    images_base_path = "C:/Users/datasets/FloodNet/data/Train/Labeled"

    graph_type = "position_knn"
    num_segments = 1000
    model_dir = "C:/Users/results/gnn_flood_detection/encoding=SIFT,graph_type={}_sag_{}_num_segments".format(graph_type, num_segments)
    model_path = os.path.join(model_dir, "best_model.pth")

    #graph_model = FloodDetectionGraph(3, 128, 256, 2, SAGPooling(256, ratio=0.2))
    graph_model = FloodDetectionGraph(3, 128, 256, 2, SAGPooling(256, ratio=0.2))

    graph_model.load_state_dict(torch.load(model_path))
    graph_model.eval()

    test_set, _ = load_dataset("C:/Users/datasets/FloodNet/", "test.txt", "SIFT", graph_type, num_segments=num_segments)

    explainer = Explainer(
        model=graph_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'))


    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    for data in test_loader:

        explanation = explainer(data.x, data.edge_index, batch=data.batch)
        print(f'Generated explanations in {explanation.available_explanations}')
        nx_graph = convert.to_networkx(data)

        visualize_graph_importance(data, explanation)









