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

# Set up logging
import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
