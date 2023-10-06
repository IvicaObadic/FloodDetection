import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal
from PIL import Image
import cv2

from graph_creation import create_sift_graph


#taken from https://github.com/ahlevering/liveability-rs/blob/master/codebase/pt_funcs/dataloaders.py#L22
class LBMDataContainer():
    def __init__(self, splits_file):
        self.labels = gpd.read_file(splits_file)
        self.labels = self.labels.set_crs(28992, allow_override=True)

        self.labels['geometry'] = self.labels.apply(self.fix_polygon, axis=1)
        # self.labels.set_geometry('geometry', inplace=True)

    def fix_polygon(self, row):
        centroid = row['geometry'].centroid
        grid_true_center_x = centroid.xy[0][0] - (centroid.xy[0][0] % 100) + 50
        grid_true_center_y = centroid.xy[1][0] - (centroid.xy[1][0] % 100) + 50
        return Point([grid_true_center_x, grid_true_center_y]).buffer(50, cap_style=3)


def load_aerial_img(region, image_id):
    patch_path = os.path.join(dataset_root_dir, region, "{}.tiff".format(image_id))
    if os.path.isfile(patch_path):
        patch = np.array(gdal.Open(patch_path).ReadAsArray()).transpose([1, 2, 0])
        patch = Image.fromarray(patch)

        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        return patch
    return None

if __name__ == '__main__':
    dataset_root_dir = "/home/datasets/Liveability/"
    dataset_info_file = os.path.join(dataset_root_dir, "grid_geosplit_not_rescaled.geojson")

    data_container = LBMDataContainer(dataset_info_file)
    num_keypoints = 500
    sift_descriptor = cv2.SIFT_create(num_keypoints)

    for index, row in data_container.labels.iterrows():

        region = row["region_name"]
        image_id = row["gridcode"]
        liveability_score = row["rlbrmtr"]
        split = row["split"]
        image = load_aerial_img(region, image_id, split)
        if image is not None:
            save_dir = os.path.join(dataset_root_dir, "graph_representation",split, "SIFT_{}_segments".format(num_keypoints))
            visualize_descriptor = (index % 1000) == 0
            create_sift_graph(sift_descriptor, save_dir, image, image_id, liveability_score, visualize_descriptor)
        else:
            print("No image found for region={} and image_id={}".format(region, image_id))








