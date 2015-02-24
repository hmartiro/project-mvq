# project-mvq
Scene understanding to aid in limited-bandwidth remote driving

Take a look at `documents/proposal.pdf` for the project description.

## Setup
This project requires OpenCV >= 3.0 and Python >= 3.4. We recommend building
OpenCV from source, configured to your Python installation.

Install system dependencies for numpy:

    apt-get install python3-numpy

Install python packages:

    cd project-mvq/python
    pip install -r requirements.txt

Configure your PYTHONPATH, in `~/.bashrc` or elsewhere:

    export PYTHONPATH=${PYTHONPATH}:/path/to/project-mvq/python/

Run the code from any directory as modules. For example, to run
`project-mvq/python/mvq/reconstruction.py`, use:

    python -m mvq.reconstruction

## Extracting information
Our input is a stream of high-resolution video data from a stereo camera pair
onboard a vehicle. This section describes various methods of recognition and
reconstruction that provide useful information in the context of enabling
smart compression.

#### Lane detection
Localizing the road and lane markers is extremely important to understanding
the scene. The road vanishing point is the single most important point of the
image, and naturally segments the image. The triangle formed by the lane
markers identifies the immediate terrain that the vehicle is crossing and
the driver should be paying attention to.

We have created a tutorial for lane detection as an IPython notebook. It is
located at `python/notebooks/lane_detection.ipynb` and can be viewed online at
[nbviewer](http://nbviewer.ipython.org/github/hmartiro/project-mvq/blob/master/python/notebooks/lane_detection.ipynb).

#### Stereo disparity map
Calculating a depth map of the scene from the stereo cameras is extremely useful
for understanding, because we can draw conclusions about how important various
features are by their distance from the vehicle. We can tie this depth data
with recognition methods to identify features.

#### Image segmentation
Segmenting the image into known regions provides an easy way to enable variable
compression. These objects might include the sky, road, grass, median, trees, buildings,
etc. Based on the object classification, we could determine the compression
level and frame rate that different portions of the video need to be transmitted at.

#### Feature recognition
More advanced feature detection for entities like vehicles, pedestrians, and
animals would highlight key areas to pay attention to. This could potentially
enable transmission of feature metadata instead of pixel data, to be reconstructed
in the form of augmented reality for the remote driver.

#### Visual flow
Going beyond single-image methods, investigating the temporal differences of the
images in the stream can tell us information about the movement of objects. We
could say, for example, that objects moving quickly across the screen are extremely
important to watch for.

#### Edge detection
General edge detection can be important for compression, with minimal required
knowledge of the scene. In general, regions of the stream without significant
edges are much less likely to be important to transmit in high-resolution.
