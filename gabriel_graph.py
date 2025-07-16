import numpy as np

from scipy.spatial import Delaunay

from numpy.typing import ArrayLike

def gabriel_graph(X: ArrayLike) -> np.ndarray:
    """
    Compute the Gabriel graph of a set of points.

    Parameters
    ----------
    X : ArrayLike
        An array of shape (n, d) where n is the number of points and d is the dimension.

    Returns
    -------
    np.ndarray
        An adjacency matrix representing the Gabriel graph.
    """
    X = np.asarray(X)
    n = X.shape[0]
    
    # Compute the Delaunay triangulation
    tri = Delaunay(X)
    
    # Initialize an adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=bool)
    
    # Iterate over each simplex in the triangulation
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                p1 = X[simplex[i]]
                p2 = X[simplex[j]]
                
                # Calculate the midpoint and squared distance to all other points
                midpoint = (p1 + p2) / 2
                radius_squared = np.sum((p1 - p2) ** 2) / 4
                
                # Check if any point is within the circle defined by the midpoint and radius
                for k in range(n):
                    if k not in simplex:
                        dist_squared = np.sum((X[k] - midpoint) ** 2)
                        if dist_squared < radius_squared:
                            break
                else:
                    # If no point is inside, add edge to adjacency matrix
                    adj_matrix[simplex[i], simplex[j]] = True
                    adj_matrix[simplex[j], simplex[i]] = True
    
    return adj_matrix.astype(int)