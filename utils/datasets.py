import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import cv2
import pandas as pd 
import numpy as np
from tqdm import tqdm

import data_loaders.classification_data_loader as cdl
import utils.images as ui
import utils.utils as uu

from sklearn.cluster import KMeans
import math


def features_shape_dataset_to_csv():

    PATH_TO_DATASET = "data/marbling_dataset_v2"
    DATASET = "Marbling"
    # CLASSES = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    
    # Define your vectors
    vector1 = (100, 100, 200)  # Replace with your actual vector
    vector2 = (60, 60, 180)  # Replace with your actual vector

    dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET)

    print(f"Tamaño del dataset: {len(dataset)}")
    print(f"Nombre del dataset: {dataset.dataset}")
    print(f"Ruta al dataset:    {dataset.path_to_data}")
    print(f"Clases del dataset: {dataset.classes}")

    df_columns = ['path','components','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','label']
    df_csv = pd.DataFrame(columns=df_columns)

    for i in tqdm(range(len(dataset))):
        path, image, label = dataset.__getitem__(i)

        # Perform segmentation
        segmentation_mask = ui.binary_segmentation_by_distance(image, vector1, vector2)

        # Find connected components
        components = ui.find_connected_components(segmentation_mask, thresh_area=4)

        features = []
        if len(components)>0:
            # Analyze area and perimeter of each connected component
            components_properties = ui.analyze_components_shape_properties(components)
            df = pd.DataFrame.from_records(components_properties)
            features = df.describe().fillna(-1).values[1:,:].flatten(order='F')

        else:
            features = np.zeros((21), dtype=np.float32)-1

        new_row = [path] + [len(components)] + features.tolist() + [label]
        df_csv.loc[len(df_csv)] = new_row
        # df_csv.loc[len(df_csv.index)] = pd.concat([path, len(components), pd.DataFrame(features).T, label])
        # df_csv.append(pd.concat([path, len(components), features.tolist(), label]))

    path_to_csv = os.path.join(PATH_TO_DATASET,"marbling_features_shape_dataset.csv")     
    df_csv.to_csv(path_to_csv, index=False, header=False)


def lbp_dataset_to_csv():

    PATH_TO_DATASET = "data/marbling_dataset_v2"
    DATASET = "Marbling"

    dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET)

    print(f"Tamaño del dataset: {len(dataset)}")
    print(f"Nombre del dataset: {dataset.dataset}")
    print(f"Ruta al dataset:    {dataset.path_to_data}")
    print(f"Clases del dataset: {dataset.classes}")


    # initialize the local binary patterns descriptor along with
    # the data and label lists
    p = 24
    radius = 8
    desc = ui.LocalBinaryPatterns(p, radius)

    df_columns = ['path'] + [str(x) for x in list(range(p+2))] + ['label']
    df_csv = pd.DataFrame(columns=df_columns)

    # loop over the training images
    for i in tqdm(range(len(dataset))):
        path, image, label = dataset.__getitem__(i)

        # Convert image to grayscale
        gray = np.array(image.convert("L"))
        # Describe image
        hist = desc.describe(gray)

        new_row = [path] + hist.tolist() + [label]
        df_csv.loc[len(df_csv)] = new_row

    path_to_csv = os.path.join(PATH_TO_DATASET,"marbling_features_lbp_dataset.csv")     
    df_csv.to_csv(path_to_csv, index=False, header=False)


def orb_dataset_to_csv():

    PATH_TO_DATASET = "data/marbling_dataset_v2"
    DATASET = "Marbling"
    k = 12

    dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET)

    print(f"Tamaño del dataset: {len(dataset)}")
    print(f"Nombre del dataset: {dataset.dataset}")
    print(f"Ruta al dataset:    {dataset.path_to_data}")
    print(f"Clases del dataset: {dataset.classes}")

    # From https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f
    # Initiate ORB detector
    orb = cv2.SIFT_create()

    df_columns = ['path'] + [str(x) for x in list(range(k))] + ['label']
    df_csv = pd.DataFrame(columns=df_columns)

    orb_descriptor_list = []
    # loop over the training images and extract all descriptors
    for i in tqdm(range(len(dataset))):
        _, image, _ = dataset.__getitem__(i)

        # Convert image to grayscale
        # image = np.array(image.convert("L"))
        image = np.array(image)

        # Find the keypoints and compute descriptors with ORB
        _, desc = orb.detectAndCompute(image, None)
 
        if desc is not None: 
            orb_descriptor_list.extend(desc)

    # A k-means clustering algorithm who takes 2 parameter which is number 
    # of cluster(k) and the other is descriptors list(unordered 1d array)
    # Returns an array that holds central points.
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(orb_descriptor_list)
    visual_words = kmeans.cluster_centers_ 


    # Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
    # And the second parameter is an array that holds the central points (visual words) of the k means clustering
    # Returns a dictionary that holds the histograms for each images that are separated class by class. 
    for i in tqdm(range(len(dataset))):
        path, image, label = dataset.__getitem__(i)

        # Convert image to grayscale
        # image = np.array(image.convert("L"))
        image = np.array(image)

        # Find the keypoints and compute descriptors with ORB
        _, desc = orb.detectAndCompute(image, None)
        
        histogram = np.zeros(len(visual_words))
        
        if desc is not None: 
            for each_feature in desc:
                count = math.inf
                ind = 0
                for i in range(len(visual_words)):
                    dist = uu.calculate_distance(each_feature, visual_words[i]) 
                    if(dist < count):
                        ind = i
                        count = dist
                histogram[ind] += 1

        new_row = [path] + histogram.tolist() + [label]
        df_csv.loc[len(df_csv)] = new_row

    path_to_csv = os.path.join(PATH_TO_DATASET,"marbling_features_sift_rgb_dataset.csv")     
    df_csv.to_csv(path_to_csv, index=False, header=False)