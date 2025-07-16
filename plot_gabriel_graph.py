import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import ArrayLike

def plot_gabriel_graph(X: ArrayLike, adj_matrix: np.ndarray) -> None:
    """
    Plot the Gabriel graph of a bidimensional set of points.

    Parameters
    ----------
    X : ArrayLike
        An array of shape (n, d) where n is the number of points and d is the dimension.
    adj_matrix : np.ndarray
        An adjacency matrix representing the Gabriel graph.

    Raises
    ------
    ValueError
        If the input points are not bidimensional (shape (n, 2)).
    """
    X = np.asarray(X)

    if X.shape[1] != 2:
        raise ValueError("The input points must be bidimensional (shape (n, 2)).")
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    
    n = X.shape[0]
    
    # Draw edges based on the adjacency matrix
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
    
    plt.title('Gabriel Graph')
    plt.show()
