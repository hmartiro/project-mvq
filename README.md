# project-mvq
Scene understanding to aid in limited-bandwidth remote driving

Take a look at `documents/proposal.pdf` for the project description.

## Getting started
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
