import os
import numpy as np
import cv2 as cv
from tkinter import filedialog
import tkinter as tk
import argparse
import shutil
import load_model as model
import utils
import pandas as pd
import pixel_intensity as pi
import torch 
import torchvision

# Define command-line arguments. If the user would like to warp the module # previous to cell segmentation and indexing, pass '-w' flag to do so.
parser = argparse.ArgumentParser(description='Run UVF Image Correction Pipeline')
parser.add_argument('-c', '--crop_module', action='store_true', help='To skip the warp step.')
parser.add_argument('-a', '--pixel_analysis', action='store_true', help="Performs basic statistic analysis on the pixel intensity of each module")
args = parser.parse_args()

# Prompt the user for the directory of UVF images.
root = tk.Tk()
root.withdraw()
image_files = filedialog.askdirectory()
dataset_dicts = utils.get_data_dicts_jpg(image_files)

# Create output folders to store accepted and rejected solar panels.
accepted_file_path = image_files + '/uvf_correction_folder/'
rejected_file_path = image_files + '/rejected_panels/'

if os.path.exists(accepted_file_path):
    shutil.rmtree(accepted_file_path)

if os.path.exists(rejected_file_path):
    shutil.rmtree(rejected_file_path)

if not os.path.isdir(accepted_file_path):
    os.mkdir(accepted_file_path)

if not os.path.isdir(rejected_file_path):
    os.mkdir(rejected_file_path)

# Initialize the predictor for the module segmentation.
predictor = model.prepare_predictor(0.5, "./model_weights/panel_model.pth")

# Make an inference for each image in dataset_dicts.
panels = []
pixel_analysis = np.array([['id', 'average', 'median', 'std_dev', 'variance']])
for i in range(len(dataset_dicts)):
    # Read in the image provided at index i
    img = cv.imread(dataset_dicts[i])
    img_path = dataset_dicts[i]
    full_name = os.path.basename(img_path)
    file_name = os.path.splitext(full_name)

    # Output the model's prediction on image i
    outputs = predictor(img)
    img_original_copy = np.copy(img)

    # Return the mask of detectron2. Not currently used because the masks for solar panels are not precise.
    # mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
    # Return the bounding box of detectron2
    bounding_box = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()

    # num_instances is the number of bounding boxes in the image
    # A bounding box indicates a complete solar panel found in the image
    num_instances = bounding_box.shape[0]
    number_of_bounding_box = int(bounding_box.shape[0])

    # Warp and crop each bounding box found in image i of dataset_dicts
    for j in range(number_of_bounding_box):
        # Identify the four corners of the bounding box
        xmin = int(bounding_box[j][0])  # minimum value of x
        ymax = int(bounding_box[j][1])  # maximum value of y
        xmax = int(bounding_box[j][2])  # maximum value of x
        ymin = int(bounding_box[j][3])  # minimum value of y

        # Use the height and width to get the points for source and destination.
        # Compute homography based on source and destination.
        width, height = utils.width_and_height_of_bounding_box(xmin, ymax, xmax, ymin)
        top_left, bottom_left, bottom_right, top_right = utils.corners_of_bounding_box(xmin, ymax, xmax, ymin)
        # Used for mask
        # maxW, maxH = utils.maxWidth_maxHeight(top_left, bottom_left, bottom_right, top_right)
        points_1_auto, points_2_auto = utils.source_and_destination(top_left, bottom_left, bottom_right, top_right,
                                                                    width, height)
        H = utils.computeHomography(points_1_auto, points_2_auto)

        # Crop the image
        cropped_img = cv.warpPerspective(img_original_copy, H, [width, height])

        # If args.crop_module is true, skip warping the module.
        # Otherwise, warp the module.
        panel_num = str(j + 1)
        if args.crop_module:
            dst = utils.isolate_module(cropped_img, height, width)
            panels.append([dst, file_name[0] + '_panel_' + panel_num])
        else:
            dst = utils.warp_module(cropped_img, height, width)
            panels.append([dst, file_name[0] + '_panel_' + panel_num])

# Initialize the predictor for the module segmentation.
# Two different paths depending on whether the user decides to warp the module.
if args.crop_module:
    predictor = model.prepare_predictor(0.7, "./model_weights/cell_cropped_model.pth")
else:
    predictor = model.prepare_predictor(0.7, "./model_weights/cell_warped_model.pth")

cell_test_dataset_dicts = panels

# Index and warp cells in given module.
for i in range(len(cell_test_dataset_dicts)):
    # Create a directory of cells per each solar panel
    panel = str(i + 1).zfill(2)

    # Read in each corrected image
    img = cell_test_dataset_dicts[i][0]

    # Outputs the prediction
    outputs = predictor(img)
    img_original_copy = np.copy(img)

    # Return the mask of detectron2. Not currently used because the masks for cells are not precise.
    # mask_array = outputs['instances'].pred_masks.to("cpu").numpy()

    # Return the bounding box of detectron2
    bounding_box = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()

    # num_instances is the number of bounding boxes in the image
    # A bounding box indicates a complete cell found in the image
    num_instances = bounding_box.shape[0]
    number_of_bounding_box = int(bounding_box.shape[0])

    rows = []
    bounding_box_list = []
    rowFound = False
    for j in range(number_of_bounding_box):
        xmin = int(bounding_box[j][0])  # minimum value of x
        ymax = int(bounding_box[j][1])  # maximum value of y
        xmax = int(bounding_box[j][2])  # maximum value of x
        ymin = int(bounding_box[j][3])  # minimum value of y

        width, height = utils.width_and_height_of_bounding_box(xmin, ymax, xmax, ymin)
        top_left, bottom_left, bottom_right, top_right = utils.corners_of_bounding_box(xmin, ymax, xmax, ymin)
        bounding_box_list.append([top_left, bottom_left, bottom_right, top_right])

    bounding_box_list.sort(key=lambda x: (x[0][0], x[0][1]))

    for k in range(number_of_bounding_box):
        rowFound = False
        top_left = bounding_box_list[k][0]  # the minimum value of x
        bottom_left = bounding_box_list[k][1]  # maximum value of y
        bottom_right = bounding_box_list[k][2]  # maximum value of x
        top_right = bounding_box_list[k][3]  # min value of y in bounding box
        width, height = utils.width_and_height_of_bounding_box(top_left[0], top_left[1], top_right[0], bottom_left[1])
        leeway = (width + height) / 8  # 1/4 of the average of width and height

        for row in range(len(rows)):
            # Check if the top right and top left of two cells are within leeway of eachother
            x_diff = abs(rows[row][-1][3][0] - top_left[0]) <= leeway
            y_diff = abs(rows[row][-1][3][1] - top_left[1]) <= leeway

            if x_diff and y_diff:
                rows[row].append([top_left, bottom_left, bottom_right, top_right])
                rows[row].sort(key=lambda x: x[0][0])
                rowFound = True
        if not rowFound:
            rows.append([[top_left, bottom_left, bottom_right, top_right]])
            rows.sort(key=lambda x: x[0][0][1])

    # Check if number of cells found in each row is equal across all rows. If not, reject panel.
    validPicture = True
    if len(rows) > 0:
        numCellsPerRow = len(rows[0])
        for row in range(1, len(rows)):
            if len(rows[row]) != numCellsPerRow:
                validPicture = False
                break
    # If there is less than 3 rows, reject panel.
    if len(rows) < 3:
        validPicture = False

    # If accepted, output folder containing panel and indexed cells.
    if validPicture:
        row_count = 0
        img_name = cell_test_dataset_dicts[i][1]
        dir = accepted_file_path + img_name + '_cells'
        print('Accepted!', img_name)

        if os.path.exists(dir):
            shutil.rmtree(dir)

        os.mkdir(dir)
        os.listdir()
        cv.imwrite(os.path.join(dir, img_name + '.jpg'), img)

        if args.pixel_analysis:
            result = pi.pixel_intensity(img, dir)
            add_id = np.append([img_name], result[0])
            pixel_analysis = np.concatenate((pixel_analysis, [add_id]))

        for row in rows:
            cell_count = 0
            row_count += 1
            for cell in row:
                cell_count += 1
                maxW, maxH = utils.maxWidth_maxHeight(cell[0], cell[1], cell[2], cell[3])
                points_1_auto, points_2_auto = utils.source_and_destination(cell[0], cell[1], cell[2], cell[3], width,
                                                                            height)
                H = utils.computeHomography(points_1_auto, points_2_auto)
                imcropped = cv.warpPerspective(img_original_copy, H, [width, height])  # Crop the image
                row_str = str(row_count).zfill(2)
                cell_str = str(cell_count).zfill(2)
                if args.crop_module:
                    cell_warped = utils.warp_module(imcropped, height, width)
                    cv.imwrite(os.path.join(dir, img_name + '_row_' + row_str + '_cell_' + cell_str + '.jpg'), cell_warped)
                else:
                    cv.imwrite(os.path.join(dir, img_name + '_row_' + row_str + '_cell_' + cell_str + '.jpg'), imcropped)

    # Panel is rejected
    else:
        img_name = cell_test_dataset_dicts[i][1]
        print('Rejected!', img_name)
        dir = rejected_file_path
        cv.imwrite(os.path.join(dir, img_name + '.jpg'), img)

    if args.pixel_analysis:
        try:
            with open(accepted_file_path + 'pixel_analysis.csv', 'w') as file:
                save = pd.DataFrame(data=pixel_analysis[1:], columns=pixel_analysis[0])
                save.to_csv(accepted_file_path + 'pixel_analysis.csv', sep=',')
        except FileNotFoundError:
            print(accepted_file_path + ' does not exist')