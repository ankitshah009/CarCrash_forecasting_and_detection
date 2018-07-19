import pandas as pd
import numpy as np
import cv2, os
import glog as log
from glob import glob
from time import time

from sklearn.model_selection import train_test_split


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



