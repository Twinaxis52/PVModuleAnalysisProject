import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist #needed for function below
from imutils.perspective import four_point_transform #for automatic transformation
from skimage.filters import threshold_local # where is this needed???
from imutils import rotate_bound #for rotation

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

#jpg data
def get_data_dicts_jpg(directory):
    dataset_dicts = []
    #idx = 0
    for filename in os.listdir(directory):
      if filename.endswith('.JPG'):
        jpg_file = os.path.join(directory, filename)
        dataset_dicts.append(jpg_file)
      elif filename.endswith('.jpg'):
        jpg_file = os.path.join(directory, filename)
        dataset_dicts.append(jpg_file)
      else:
        continue
    return dataset_dicts

#calculates the width and height of bounding box
def width_and_height_of_bounding_box(xmin,ymax,xmax,ymin):
  w = (xmax - xmin)+300 #width and height of bounding max
  h = (ymin - ymax)+300
  return w, h

#Gets our four corners of bounding box
def corners_of_bounding_box(xmin,ymax,xmax,ymin):
  top_left = [xmin, ymax]
  bottom_left = [xmin, ymin]
  bottom_right = [xmax, ymin]
  top_right = [xmax, ymax]
  return top_left, bottom_left, bottom_right, top_right

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
 
def maxWidth_maxHeight(top_left, bottom_left, bottom_right, top_right):
  # calculate width of our 4 points
  wA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) **2))
  wB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) **2))

  # calculate height of our 4 points
  hA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) **2))
  hB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) **2))

  # get max width and height to reach final destination
  maxW = max(int(wA), int(wB))
  maxH = max(int(hA), int(hB))
  return maxW, maxH

#this function is use for warping
def source_and_destination(top_left, bottom_left, bottom_right, top_right, w, h):
  points_1_auto = np.float32([[top_left[0], top_left[1]], 
                       [top_right[0], top_right[1]], 
                       [bottom_left[0], bottom_left[1]], 
                       [bottom_right[0], bottom_right[1]]]) #source
  points_2_auto = np.float32([[0,0], [w,0], [0,h], [w,h]]) #destination for bounding box
  return points_1_auto, points_2_auto

#function use to compute homography manually
def computeH_Manual(pts_1, pts_2): #compute H 
    n = len(pts_1)
    A = np.zeros((2*n,9))
    
    for i in range(len(pts_1)):
        x1 = pts_1[i][0]
        y1 = pts_1[i][1]

        x2 = pts_2[i][0]
        y2 = pts_2[i][1]
        A[2*i, :] = ([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A[2*i+1, :] = ([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

#function use to compute homography by python
def computeHomography(points_1_auto, points_2_auto):
  H, status = cv2.findHomography(points_1_auto, points_2_auto) #source and destination
  return H


def binary_tresholding(stretch_1):
  # apply binary thresholding
  ret, thresh = cv.threshold(stretch_1, 40, 255, cv.THRESH_BINARY)
  # visualize the binary image
  return thresh


#retrieves contours
def get_contour(thresh): 
  contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
  return contours

def get_black_canvas(imwarped, big_contour):
  #make a mask of the contour from image warped
  black_canvas = np.zeros_like(imwarped) #make a replica thats just all black
  cv.drawContours(black_canvas, big_contour, -1, 255, cv.FILLED) #draw contours on black image
  cv.fillConvexPoly(black_canvas, big_contour,(255,255,255)) #fill the contours
  return black_canvas

def bitwise_result(imwarped, black_canvas):
  #make everything outside the mask black and restore color inside mask
  result = cv.bitwise_and(imwarped, black_canvas)
  return result 

#gives us warped module 
def final_warped_module(sorted_contours, imwarped):
  for c in sorted_contours:
      peri = cv.arcLength(c, True)
      approx = cv.approxPolyDP(c, 0.05 * peri, True)
      if len(approx) == 4:
          screenCnt = approx
          break
  if len(screenCnt) == 4:
    warped = four_point_transform(imwarped, screenCnt.reshape(4, 2))
    return warped
  
#make dictionary of data for cell
def get_data_dicts_cell(directory, classes):
    dataset_dicts = []
    idx = 0
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
        record["image_id"] = idx
        idx = idx+1
        
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS, #BoxMode.XYWH_ABS or BoxMode.XYXY_ABS
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Warp the given image further
# This may be an optional function.
def warp_module(warp_img, h, w):
  blur = cv.GaussianBlur(warp_img, (5,5), 5) #blur the image
  #show_image(blur)
  gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  #blur to gray
  #show_image(gray) #all functions before this run good
  hist,bins = np.histogram(gray.flatten(),256,[0,256])#histogran stretch beginning
  cdf = hist.cumsum() #plotting hist
  cdf_normalized = cdf * hist.max()/ cdf.max()
  cdf_m = np.ma.masked_equal(cdf,0)
  cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
  cdf = np.ma.filled(cdf_m,0).astype('uint8')
  stretch_1 = cdf[gray] #final stretch result
  #show_image(stretch_1)
  # apply binary thresholding
  tresh = binary_tresholding(stretch_1)
  #show_image(tresh) #works up to here
  contours = get_contour(tresh)
  #show_contours(contours, imwarped)
  sorted_contours = sorted(contours, key = cv.contourArea, reverse= True)
  big_contour = sorted_contours[0] #get largest contour from all contours
  #show_contours(big_contour, imwarped) #show biggest contour
  black_canvas = get_black_canvas(warp_img, big_contour)
  #show_image(black_canvas)
  result = bitwise_result(warp_img, black_canvas)
  #show_image(result)
  blur = cv.GaussianBlur(result, (3,3), 3)
  gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
  tresh = binary_tresholding(gray)
  contours = get_contour(tresh)
  #print(j)
  contours, hierarchy = cv.findContours(tresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
  sorted_contours = sorted(contours, key = cv.contourArea, reverse= True)
  big_contour = sorted_contours[0]
  final_warped = final_warped_module(sorted_contours, warp_img)
  if h > w: #this will rotate the image 90 degrees if height greater than width
    final_warped = rotate_bound(final_warped,-90)
  return final_warped

def isolate_module(warp_img,h,w):
  blur = cv.GaussianBlur(warp_img, (5,5), 5) #blur the image
  #show_image(blur)
  gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  #blur to gray
  #show_image(gray) #all functions before this run good
  hist,bins = np.histogram(gray.flatten(),256,[0,256])#histogran stretch beginning
  cdf = hist.cumsum() #plotting hist
  cdf_normalized = cdf * hist.max()/ cdf.max()
  cdf_m = np.ma.masked_equal(cdf,0)
  cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
  cdf = np.ma.filled(cdf_m,0).astype('uint8')
  stretch_1 = cdf[gray] #final stretch result
  #show_image(stretch_1)
  # apply binary thresholding
  tresh = binary_tresholding(stretch_1)
  #show_image(tresh) #works up to here
  contours = get_contour(tresh)
  #show_contours(contours, imwarped)
  sorted_contours = sorted(contours, key = cv.contourArea, reverse= True)
  big_contour = sorted_contours[0] #get largest contour from all contours
  #show_contours(big_contour, imwarped) #show biggest contour
  black_canvas = get_black_canvas(warp_img, big_contour)
  #show_image(black_canvas)
  result = bitwise_result(warp_img, black_canvas)
  return result
