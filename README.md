# project-mvq
Scene understanding to aid in limited-bandwidth remote driving

## Introduction

Progress on autonomous driving promises to make our transportation infrastructure more efficient.
In particular, huge efficiency gains could be made in the transportation of goods by using vehicles
without humans in them. These vehicles would have a fraction of the components of passenger vehicles,
and could be designed without crumple zones, seats, steering wheels, infotainment, airbags, door and
window controls. Instead, they could consist of a simple electric drive-train, a sensor cluster, and
a cargo bay. In this type of logistics infrastructure, the hardware is lighter and cheaper, there are
no human drivers, and the system can be intelligently automated and running 24/7, 365.

However, fully autonomous vehicles are not yet ready to fulfill this vision. Existing solutions operate
well in pre-mapped regions and good conditions, but break down in adverse weather and in the presence
of unexpected events or unknown terrain. One alternative which acts as a gateway to this ultimate goal
is to use remotely-driven vehicles. The hardware remains the same, but the vehicle is operated by a
human in a driving simulator at a remote location. This approach retains many of the efficiency benefits
of the autonomous version while keeping the flexibility and judgement of human drivers.

One of the key challenges with this approach is maintaining a robust communication stream that enables
remote driving. In order to reliably operate a remote vehicle, the driver must have a low-latency,
high-fidelity, and wide-coverage stream of audio and video. Existing infrastructure has good coverage
in many areas, but bandwidth can be very limited. To accommodate this, we propose that many parts of
a typical stream have no importance to driving capability. For example, the driver does not care about
details of the sky, trees, or objects very far away. A nearby object moving quickly across the view,
however, is extremely important.

We propose to rank the components of a vehicle's stereo vision stream in terms of their importance to
remote driving capability, with the goal of enabling limited-bandwidth transmission of the stream.

Take a look at `documents/progress.pdf` for the full project description and current state.

## Methods
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
