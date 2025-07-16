import numpy as np
from sklearn.datasets import make_blobs
from gabriel_graph import gabriel_graph
from plot_gabriel_graph import plot_gabriel_graph

# Generate a synthetic blob dataset
X, _ = make_blobs(
    n_samples=100,
    n_features=2,
)[0:2]

# Compute the Gabriel graph adjacency matrix
adj_matrix = gabriel_graph(X)

# Plot the Gabriel graph
plot_gabriel_graph(X, adj_matrix)