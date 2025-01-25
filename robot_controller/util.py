import numpy as np

def order_points_counter_clockwise(points):
    """
    Orders the points in counter-clockwise order.

    Parameters:
        points (array-like): List of points representing a polygon (N, 2 or 3).

    Returns:
        List of points ordered counter-clockwise.
    """
    # Convert to a NumPy array for convenience
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    # `np.arctan2` gives angles in radians from -π to π
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on their angles
    sorted_indices = np.argsort(angles)

    # Return the points in the sorted order
    return points[sorted_indices]
