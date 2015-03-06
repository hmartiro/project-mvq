"""
Top level init file, always loaded when using the package.

"""

__title__ = 'mvc'
__version__ = '0.1.0'
__authors__ = ['Hayk Martirosyan', 'Brady Jon Quist']
__license__ = 'GPLv2'

# Absolute path to the top level directory
import os
path = os.path.dirname(os.path.realpath(__file__))

# Paths to data directories
data_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'data')
images_path = os.path.join(data_path, 'images')
videos_path = os.path.join(data_path, 'videos')
stereo_path = os.path.join(data_path, 'stereo')
kitti_path = os.path.join(data_path, 'kitti')

# Set up logging
import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
