import pandas as pd
import numpy as np
import cv2, os
import glog as log
from glob import glob
from time import time

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import colorsys, random


def interpolation(data, index_timeseries, method="linear"):
    """
    Interpolate the data and return as a DataFrame.
    :param data: a numpy array with the size (N,4): [x,y,w,h]
    :param index_timeseries: time series labels
    :param method: method for interpolation.
    :return: interpolated DataFrame
    """
    d = pd.DataFrame(data=data, index=index_timeseries, columns=["x", "y", "w", "h"])
    return d.resample("L").interpolate(method=method)


def extract_frames(json_data, input_dir, output_dir):
    """
    Extract the frames following the annotation JSON data.
    :param json_data:
    :param input_dir:
    :param output_dir:
    :return:
    """
    segment_counter = 0
    last_counter = {}
    if not os.path.exists(output_dir):
        raise ValueError("{} does not exist!".format(output_dir))
    for video_name in json_data:
        st = time()
        segments = json_data[video_name]
        cap = cv2.VideoCapture("{}/{}".format(input_dir, video_name))
        if not cap.isOpened():
            log.error("Cannot open {}".format(video_name))
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        endpoints = []
        for segment in segments:
            timestamps = []
            for t in segment["keyframes"]:
                timestamps.append(int(t["frame"]*fps))
            timestamps.sort()
            if len(timestamps) < 2:
                log.error("Illegal segment: less than two endpoints.")
                continue
            endpoints.append([timestamps[0], timestamps[1]+1])
        endpoints.sort(key=lambda k: k[0])
        log.info("{} has {} segments".format(video_name, len(endpoints)))
        if len(endpoints) == 0:
            cap.release()
            continue
        counter = 0
        segment_counter_start = segment_counter
        for i in range(len(endpoints)):
            os.makedirs("{}/{:06d}".format(output_dir,segment_counter))
            last_counter[segment_counter] = 0
            segment_counter += 1
        while True:
            if counter > np.max(np.array(endpoints)[:,1]):
                break
            segs = []
            for i, seg in enumerate(endpoints):
                if counter in range(seg[0], seg[1]):
                    segs.append(i+segment_counter_start)
            if len(segs) == 0:
                cap.grab()
            else:
                status, img = cap.read()
                if not status or img is None:
                    break
                for i in segs:
                    idx = last_counter[i]
                    cv2.imwrite("{}/{:06d}/{}.jpg".format(output_dir, i, idx), img)
                    last_counter[i] += 1
            counter += 1
        cap.release()
        log.info("Finished video {} in {:.5f} seconds".format(video_name, time()-st))
    return last_counter


def cross_validation_folds(idx, n_folds=4, test_size=0.33, random_state=42):
    """
    Sample train/test splits for cross-validation.
    :param idx:
    :param n_folds:
    :param test_size:
    :param random_state:
    :return:
    """
    assert n_folds > 0
    folds = []
    x = np.zeros_like(idx)
    for i in range(n_folds):
        x_train, x_test, y_train, y_test = train_test_split(x, idx, test_size=test_size, random_state=random_state)
        folds.append([y_train, y_test])
    return folds


def draw_images(img, boxes, labels, attributes, colors, track_idx, scores=None, show_id=True):
    """
    Draw images with bounding boxes and labels.
    :param img:
    :param boxes:
    :param labels:
    :param attributes:
    :param colors:
    :param track_idx:
    :param scores:
    :return:
    """
    # number of boxes
    n = len(boxes)
    _, ax = plt.subplots(1, figsize=(8,4))
    ax.axis("off")
    for i in range(n):
        color = colors[i]
        label = labels[i]
        attr = attributes[i]
        id = track_idx[i]
        try:
            score = scores[i]
        except:
            score = None
        y1, x1, y2, x2 = boxes[i]
        if label not in ["FarRegion", "CrowdRegion"]:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, edgecolor=color, facecolor='none')
        else:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True)
        ax.add_patch(p)

        # Label
        if show_id:
            caption = "ID {}: {} {} {:.3f}".format(id, label, attr, score) if score else "ID {}: {} {}".format(id, label, attr)
        else:
            caption = "{} {} {:.3f}".format(label, attr, score) if score else "{} {}".format(label, attr)
        ax.text(x1-4, y1 + 8, caption,
                color='w', size=8, backgroundcolor="none")
    ax.imshow(img)
    #plt.show()
    return plt.gcf()


def read_vatic(fpath):
    annotations = {}
    with open(fpath) as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split(" ")
            frame = {
                "box": [int(parts[2]), int(parts[1]), int(parts[4]), int(parts[3])],
                "attribute": "",
                "visible": (int(parts[6]) == 0)
            }
            if len(parts) > 10:
                frame["attribute"] = parts[10].strip().strip('\"')
            if int(parts[0]) not in annotations:
                annotations[int(parts[0])] = {"frames": {int(parts[5]): frame}, "label": parts[9].strip().strip('\"')}
            else:
                if parts[9].strip().strip('\"') != annotations[int(parts[0])]["label"]:
                    log.error("An illegal track")
                annotations[int(parts[0])]["frames"][int(parts[5])] = frame
    fp.close()
    return annotations


def find_boundary(annotations):
    separators = []
    for anno in annotations:
        if annotations[anno]["label"] == "Separator":
            separators.append(anno)
    if len(separators) > 2:
        log.info("More than two separators. Only first and last separators are counted.")
    if len(separators) == 0:
        return [0, -1] # the whole video is counted
    # Find the boundary of the segment
    intervals = []
    for anno in separators:
        fidx = []
        frames = annotations[anno]["frames"]
        for fid in frames:
            if frames[fid]["visible"]:
                fidx.append(fid)
        intervals.append([min(fidx), max(fidx)])
    intervals = np.array(intervals)
    if len(separators) == 1:
        return [0,np.min(intervals[:,0])]
    return [np.min(intervals[:,0]), np.max(intervals[:,1])]


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


