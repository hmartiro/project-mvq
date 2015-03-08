"""

"""

import numpy as np


def ransac_vanishing_point_detection(lines, distance=50, iterations=100):
    """
    Calculate the vanishing point of the road markers.

    :param lines: the lines defined as a [x1, y1, x2, y2] (4xN array, where N is the number of lines)
    :param distance: the distance (in pixels) to determine if a measurement is consistent
    :param iterations: the number of RANSAC iterations to use
    :return: Coordinates of the road vanishing point
    """

    # Number of lines
    N = len(lines)

    # Maximum number of consistant lines
    max_num_consistent_lines = 0

    # Best fit point
    best_fit = None

    # Loop through all of the iterations to find the most consistent value
    for i in range(0, iterations):

        # Randomly choosing the lines
        random_indices = np.random.choice(N, 2, replace=False)
        i1 = random_indices[0]
        i2 = random_indices[1]
        x1, y1, x2, y2 = lines[i1]
        x3, y3, x4, y4 = lines[i2]

        # Find the intersection point
        x_intersect = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        y_intersect = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

        if y_intersect < 80 or y_intersect > 300:
            continue

        this_num_consistent_lines = 0

        # Find the distance between the intersection and all of the other lines
        for i2 in range(0, N):

            tx1, ty1, tx2, ty2 = lines[i2]
            this_distance = (np.abs((ty2-ty1)*x_intersect - (tx2-tx1)*y_intersect + tx2*ty1 - ty2*tx1)
                             / np.sqrt((ty2-ty1)**2 + (tx2-tx1)**2))

            if this_distance < distance:
                this_num_consistent_lines += 1

        # If it's greater, make this the new x, y intersect
        if this_num_consistent_lines > max_num_consistent_lines:
            best_fit = int(x_intersect), int(y_intersect)
            max_num_consistent_lines = this_num_consistent_lines

    return best_fit
