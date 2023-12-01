import cv2
import pandas as pd 
import numpy as np
import utils.images as ui
import utils.datasets as uds

# uds.features_shape_dataset_to_csv()
uds.lbp_dataset_to_csv()

# # Load the image
# # image_path =   # L0
# images_path = ["data/marbling_dataset/L0/0_jpg.rf.999322b1921ffc195037841cf9951150.jpg",
#               "data/marbling_dataset/L1/0_jpg.rf.f60dfb86db35756bdb9e53c587233517.jpg",
#               "data/marbling_dataset/L2/0_jpg.rf.09716e81f484f56e496f065101ffc107.jpg",
#               "data/marbling_dataset/L3/0_jpg.rf.29e565669983b37a80d8931746a721d3.jpg",
#               "data/marbling_dataset/L4/0_jpg.rf.4419c849d3c3ccccf3791fb340ab8590.jpg",
#               "data/marbling_dataset/L5/0_jpg.rf.7636d91645b2005eb04280de9549cece.jpg",
#               "data/marbling_dataset/L6/0_jpg.rf.4b1c5534b25cee9ded5da2c77671228c.jpg",
#               "data/marbling_dataset/L7/1_jpg.rf.42ad63829387552f5084451cbb3a239b.jpg",
#              ]

# for image_path in images_path:              
#     image = cv2.imread(image_path)

#     # Define your vectors
#     vector1 = (100, 100, 200)  # Replace with your actual vector
#     vector2 = (60, 60, 180)  # Replace with your actual vector

#     # Perform segmentation
#     segmentation_mask = ui.binary_segmentation(image, vector1, vector2)

#     # Find connected components
#     components = ui.find_connected_components(segmentation_mask, thresh_area=4)

#     features = []
#     if len(components)>0:
#         # Analyze area and perimeter of each connected component
#         components_properties = ui.analyze_components_properties(components)
#         df = pd.DataFrame.from_records(components_properties)
#         features = np.append(len(components), df.describe().values[1:,:].flatten(order='F'))

#     else:
#         features = np.append(0, np.zeros((21), dtype=np.float32)-1)

#     print(image_path)
#     print(f"Cantidad de regiones 'grasa': {len(components)}")
#     print(features)