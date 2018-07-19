Sample splits for CC forecasting and detection
=====

Please find in [1/](./1/), [2/](./2/), [3/](./3/) and [4/](./4/) the sample splits for cross validation. In each folder, there is a `train.txt` and `test.txt` for training and testing respectively. In each file, there is a list of video IDs.

* In [segments100_600.json](./segments100_600.json), there is a list of video IDs which have 100~600 frames each. This list was used to sample these splits. The source code for sampling the splits can be found in [annotations.py](./analysis/annotations.py).
