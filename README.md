# CarND Object Detection Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![](assets/clip.gif)

---
## Overview

This is a local copy of a [Udacity lab](https://github.com/udacity/CarND-Object-Detection-Lab) aimed at:

* Learning about *MobileNets* and separable depthwise convolutions.
* The SSD (Single Shot Detection) architecture used for object detection
* Using pretrained TensorFlow object detection inference models to detect objects
* Using different architectures and weigh the tradeoffs.
* Applying an object detection pipeline to a video.

The jupyter notebbok contains exercise on all of the above.

### Requirements

Install environment with [Anaconda](https://www.continuum.io/downloads):

```sh
conda env create -f environment.yml
```

Before doing that, change TensorFlow pip installation in the [`environment.yml`](./environment.yml) file to be `tensorflow-gpu` or `tensorflow` depending on whether or not you have a GPU available.

The environment should be listed via `conda info --envs`:

```sh
# conda environments:
#
carnd-advdl-odlab        /usr/local/anaconda3/envs/carnd-advdl-odlab
root                  *  /usr/local/anaconda3
```

**NOTE** on [`environment.yml`](./environment.yml):

I ran into a couple of issues in using the original `environment.yml` file as provided by the [Udacity repo](https://github.com/udacity/CarND-Object-Detection-Lab).

1) First of all, I got an `UnsatisfiableError` during the building process, analogusly to what reported in this [issue](https://knowledge.udacity.com/questions/55633). I solved it following the indications reported there, i.e. moving all the python dependencies in the `pip` section of my file.
2) I got a `cannot import name 'AsyncGenerator` error while running the notebook from the environment. According to [this post](https://stackoverflow.com/questions/60927504/cannot-get-jupyter-notebook-to-run-cannot-import-name-asyncgenerator) on Stack Overflow this was due to the Python version. I updated from 3.6 to 3.6.1 and that fixed it.


Further documentation on [working with Anaconda environments](https://conda.io/docs/using/envs.html#managing-environments). 

Particularly useful sections:

https://conda.io/docs/using/envs.html#change-environments-activate-deactivate
https://conda.io/docs/using/envs.html#remove-an-environment


### Resources

* TensorFlow object detection [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [Driving video](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/driving.mp4)

### Tips
- Some windows users have reported the driving video as playable only in Jupyter Notebook operating in Chrome browser, and not in media player or Jupyter Notebook operating in other browsers.  In contrast the post-segmentation video appears to be operating across players and browsers.
