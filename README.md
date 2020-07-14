# CarND Object Detection Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![](assets/clip.gif)

---
## Overview

This is a local copy of a [Udacity lab](https://github.com/udacity/CarND-Object-Detection-Lab) aimed at:

* Learning about *MobileNets* and separable depthwise convolutions.
* Introducing the SSD (Single Shot Detection) architecture used for object detection
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

I ran into a few issues in using the original `environment.yml` file as provided by the [Udacity repo](https://github.com/udacity/CarND-Object-Detection-Lab) on my system (Mac Book Pro/MacOS Catalina 10.15.5):

1) First of all, I got an `UnsatisfiableError` during the building process, analogusly to what reported in this [issue](https://knowledge.udacity.com/questions/55633). I solved it following the indications reported there, i.e. moving all the python dependencies in the `pip` section of my file.
2) I got a `cannot import name 'AsyncGenerator` error while running the notebook from the environment. According to [this post](https://stackoverflow.com/questions/60927504/cannot-get-jupyter-notebook-to-run-cannot-import-name-asyncgenerator) on Stack Overflow this was due to the Python version. I updated from 3.6 to 3.6.1 and that fixed the error.
3) However, with python 3.6.1 I got the following error message when trying to import Tensorflow in the first cell of the notebook:

```sh
   /miniconda3/envs/carnd-advdl-odlab/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
```

After some research on the error I decided to actually downgrade to python 3.5.2. This seems to have fixed the errors and allows the notebook to run (even if I still receive some warnings when importing some of the libraries).

### More on Conda

Further documentation on working with Anaconda environments can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

Particularly useful sections:

* [Activate an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)
* [Deactivate an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#deactivating-an-environment)
* [Remove an evironment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment)


### Resources

**IMPORTANT NOTE** on TensorFlow Model zoo: the original [Udacity lab](https://github.com/udacity/CarND-Object-Detection-Lab) used to make reference to the TensorFlow object detection [_model zoo_](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). However, around July 2020 TensorFlow released the [model garden](https://github.com/tensorflow/models) that seems to have superseded the "zoo": at the moment I'm writing (mid-July 2020) the previous link to the zoo is actually not accessible anymore. However, tho _specific_ zoos are now available, one for [TensorFlow V1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and one for [TensorFlow V2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

The models refernced in the Jupyter notebook are still accessible and make reference to V1, so I have updated all the link here in the readme and in the Julpyter notebook itself with the reference to the V1 zoo.

* TensorFlow **V1** object detection [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
* [Driving video](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/driving.mp4)

### Tips
- Some windows users have reported the driving video as playable only in Jupyter Notebook operating in Chrome browser, and not in media player or Jupyter Notebook operating in other browsers.  In contrast the post-segmentation video appears to be operating across players and browsers.
