"""
This code is taken and adjusted from this kernel: https://www.kaggle.com/chattob/intel-mobileodt-cervical-cancer-screening/cervix-segmentation-gmm/run/1052170
"""

import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i - position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif (height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area = maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r - 1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3] + 1 - maxArea[0] / abs(maxArea[1] - maxArea[2])),
            maxArea[2], maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img):
    if (img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ff = np.zeros((gray.shape[0], gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), 'uint8')
    cv2.floodFill(ff, ff_mask,
                  (int(gray.shape[1] / 2), int(gray.shape[0] / 2)), 1)

    rect = maxRect(ff)
    rectangle = [min(rect[0], rect[2]), max(rect[0], rect[2]),
                 min(rect[1], rect[3]), max(rect[1], rect[3])]
    img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    # cv2.rectangle(img, (min(rect[1], rect[3]), min(rect[0], rect[2])),
    #               (max(rect[1], rect[3]), max(rect[0], rect[2])), 3, 2)

    return (img_crop, rectangle, tile_size)


def Ra_space(img, Ra_ratio, a_threshold):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w * h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt(
                (w / 2 - i) * (w / 2 - i) + (h / 2 - j) * (h / 2 - j))
            Ra[i * h + j, 0] = R
            Ra[i * h + j, 1] = min(imgLab[i][j][1], a_threshold)

    Ra[:, 0] /= max(Ra[:, 0])
    Ra[:, 0] *= Ra_ratio
    Ra[:, 1] /= max(Ra[:, 1])

    return Ra


def get_and_crop_image(img):
    img, rect, tile_size = cropCircle(img)
    w = img.shape[0]
    h = img.shape[1]
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Saturating the a-channel at 150 helps avoiding wrong segmentation
    # in the case of close-up cervix pictures where the bloody os is falsly segemented as the cervix.
    Ra = Ra_space(img, 1.0, 150)
    a_channel = np.reshape(Ra[:, 1], (w, h))

    g = mixture.GaussianMixture(n_components=2, covariance_type='diag',
                                random_state=0, init_params='kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000]
    g.fit(image_array_sample)
    labels = g.predict(Ra)
    labels += 1  # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

    # The cluster that has the highest a-mean is selected.
    labels_2D = np.reshape(labels, (w, h))
    gg_labels_regions = measure.regionprops(labels_2D,
                                            intensity_image=a_channel)
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

    mask = np.zeros((w * h, 1), 'uint8')
    mask[labels == cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w, h))


    cc_labels = measure.label(mask_2D, background=0)
    regions = measure.regionprops(cc_labels)
    areas = [prop.area for prop in regions]

    regions_label = [prop.label for prop in regions]
    largestCC_label = regions_label[areas.index(max(areas))]
    mask_largestCC = np.zeros((w, h), 'uint8')
    mask_largestCC[cc_labels == largestCC_label] = 255

    img_masked = img.copy()
    img_masked[mask_largestCC == 0] = (0, 0, 0)
    img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)

    _, thresh_mask = cv2.threshold(img_masked_gray, 0, 255, 0)

    kernel = np.ones((11, 11), np.uint8)
    thresh_mask = cv2.dilate(thresh_mask, kernel, iterations=1)

    thresh_mask = cv2.erode(thresh_mask, kernel, iterations=2)


    _, contours_mask, _ = cv2.findContours(thresh_mask.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours_mask, key=cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(main_contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    # TODO: do y and x need to be switched here? This seems to work but why?
    img_cropped_2 = img[y:y+h, x:x+w, :]
    return (img_cropped_2, None)


def get_image_files(train_folder, output_folder):
    types = ['Type_1','Type_2','Type_3']
    result = []
    for t in types:
        if not os.path.exists(os.path.join(output_folder, t)):
            os.mkdir(os.path.join(output_folder, t))

        for fn in os.listdir(os.path.join(train_folder, t)):
            if os.path.splitext(fn)[-1] == '.jpg':
                result.append((os.path.join(train_folder, t, fn),
                               os.path.join(output_folder, t, fn)))
    return result


def get_image_data(filename):
    """
    Method to get image data as np.array specifying image id and type
    """
    img = cv2.imread(filename)
    assert img is not None, "Failed to read image : %s" % (filename, )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def parallelize_image_cropping(image_files):
    p = Pool(cpu_count())
    p.map(process_image, image_files)
    return

def process_image(filenames):
    t = time.time()
    fn_in, fn_out = filenames
    if not os.path.exists(fn_out):
        img = get_image_data(fn_in)
        img_cropped, rect = get_and_crop_image(img)

        # Resize
        if (img_cropped.shape[0] > img_cropped.shape[1]):
            tile_size = (
            int(img_cropped.shape[1] * 256 / img_cropped.shape[0]), 256)
        else:
            tile_size = (
            256, int(img_cropped.shape[0] * 256 / img_cropped.shape[1]))
        img_resized = cv2.resize(img_cropped, dsize=tile_size)

        img_out = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fn_out, img_out)
        print('Proccessed image {} in {} seconds'.format(fn_in, time.time()-t))

def main(train_folder, output_folder):
    image_files = get_image_files(train_folder, output_folder)
    #for fn_in, fn_out in image_files:
    parallelize_image_cropping(image_files)


if __name__ == "__main__":
    train_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(train_folder, output_folder)
