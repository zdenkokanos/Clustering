import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
import random

ITERATIONS = 20000
INIT_POINTS = 20
MAX_DISTANCE = 500

def offset_calculation(coordinate):
    offset = 100
    if 5000 - abs(coordinate) < 100:
        offset = (5000 - abs(coordinate)) // 2
    return offset

def gen_offset(x, y):
    x_offset = offset_calculation(x)
    y_offset = offset_calculation(y)
    x = x + np.random.randint(-x_offset, x_offset)
    y = y + np.random.randint(-y_offset, y_offset)
    return np.array([x, y])  # Return as a NumPy array


def init_points():
    coordinates_set = set()
    clusters = []  # Initialize clusters directly
    while len(coordinates_set) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        point = np.array((x, y))
        if tuple(point) not in coordinates_set:  # Check for uniqueness
            coordinates_set.add(tuple(point))
            clusters.append([point])  # Create a new cluster for the unique point

    while len(clusters) < ITERATIONS + INIT_POINTS:
        random_cluster = random.choice(clusters)  # Choose a random cluster
        x, y = random_cluster[0]  # Get the point from the chosen cluster
        new_point = gen_offset(x, y)  # Generate a new point based on the chosen point

        if tuple(new_point) not in coordinates_set:
            coordinates_set.add(tuple(new_point))
            clusters.append([new_point])  # Add the new point as a new cluster
    print("Clusters were initialized..")
    return clusters  # Return clusters directly

def centroid(cluster):
    return np.mean(cluster, axis=0)

def evaluate_clusters(clusters):
    for cluster in clusters:
        centroid_value = centroid(cluster)
        # Calculate distances from all points in the cluster to the centroid
        distances = distance.cdist(cluster, [centroid_value], metric='euclidean').flatten()  # Flatten to get a 1D array
        avg_distance = np.mean(distances)
        if avg_distance > MAX_DISTANCE:
            return False  # Stop if any cluster exceeds the MAX_DISTANCE and revert to the previous iteration
    return True  # Continue if all clusters meet the MAX_DISTANCE

def distance_matrix(clusters):
    print("Computing distance matrix...")
    centroids = np.array([centroid(cluster) for cluster in clusters])  # Precompute centroids to not repeat calculations
    dist_matrix = distance.cdist(centroids, centroids, metric='euclidean')  # Use SciPy to calculate distance matrix
    np.fill_diagonal(dist_matrix, np.inf)  # Set distances to itself to infinity
    return dist_matrix, centroids

def agglomerative_centroid(clusters):
    # Precompute the initial centroids and distance matrix
    dist_matrix, centroids = distance_matrix(clusters)
    previous_clusters = None

    while True:
        # Stop if point distances in clusters exceed the limit
        if len(clusters) < 30:
            if not evaluate_clusters(clusters):
                print("MAX_DISTANCE exceeded stopping.")
                break
            previous_clusters = clusters[:]  # Save the last valid clustering state

        # Find the closest two clusters to merge
        cluster1, cluster2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge cluster2 into cluster1
        clusters[cluster1] = np.vstack([clusters[cluster1], clusters[cluster2]])
        del clusters[cluster2]  # Remove the merged cluster

        # Recompute the centroid for the new merged cluster
        centroids[cluster1] = centroid(clusters[cluster1])
        centroids = np.delete(centroids, cluster2, axis=0)

        # Remove cluster2 from the distance matrix and update distances for the new merged cluster
        dist_matrix = np.delete(dist_matrix, cluster2, axis=0)
        dist_matrix = np.delete(dist_matrix, cluster2, axis=1)
        dist_matrix[cluster1, :] = distance.cdist([centroids[cluster1]], centroids, metric='euclidean')
        dist_matrix[:, cluster1] = dist_matrix[cluster1, :]
        dist_matrix[cluster1, cluster1] = np.inf  # Set distance to itself to infinity

        print(f"Merged clusters {cluster1} and {cluster2}, Total clusters remaining: {len(clusters)}")

    # Return final centroids and clusters
    final_centroids = np.array([centroid(cluster) for cluster in previous_clusters])
    return final_centroids, previous_clusters


def show_clusters(clusters):
    plt.figure(figsize=(8, 8))
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    # Use a colormap to generate distinct colors
    colors = plt.colormaps['tab20']

    for idx, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        centroid = np.mean(cluster_points, axis=0)

        # Use the colormap to assign a distinct color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5,
                    color=colors(idx),  # Get color from the colormap
                    label=f"Cluster {idx + 1}")
        plt.scatter(centroid[0], centroid[1], s=100,
                    color=colors(idx),
                    edgecolor='black', marker='o')

    plt.title("Agglomerative Clustering Visualization with Centroids and 20k points")
    plt.legend()
    plt.show()


def main():
    seed_num = 43
    random.seed(seed_num)
    np.random.seed(seed_num)
    clusters = init_points()
    final_centroids, final_clusters = agglomerative_centroid(clusters)
    show_clusters(final_clusters)


if __name__ == "__main__":
    main()
