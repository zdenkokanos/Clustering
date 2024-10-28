import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
import random

ITERATIONS = 10000
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
    clusters = []  # Initialize clusters directly with points
    while len(coordinates_set) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        point = np.array((x, y))
        if tuple(point) not in coordinates_set:  # Check for uniqueness
            coordinates_set.add(tuple(point))
            clusters.append([point])  # Create a new cluster for the unique point

    while len(clusters) < ITERATIONS + INIT_POINTS:
        random_cluster = random.choice(clusters)  # Choose a random cluster
        x, y = random_cluster[0]
        new_point = gen_offset(x, y)

        if tuple(new_point) not in coordinates_set:
            coordinates_set.add(tuple(new_point))
            clusters.append([new_point])  # Add the new point as a new cluster
    print("Clusters were initialized..")
    return clusters


def medoid(cluster):
    # Calculate the medoid of a cluster by finding the point closest to the centroid.
    centroid_value = np.mean(cluster, axis=0)
    distances = distance.cdist(cluster, [centroid_value], metric='euclidean').flatten()
    medoid_index = np.argmin(distances)  # Find the index of the closest point to the centroid
    return cluster[medoid_index]


def evaluate_clusters(clusters):
    for cluster in clusters:
        medoid_value = medoid(cluster)
        # Calculate distances from all points in the cluster to the medoid
        distances = distance.cdist(cluster, [medoid_value], metric='euclidean').flatten()  # Flatten to get a 1D array
        avg_distance = np.mean(distances)
        if avg_distance > MAX_DISTANCE:
            return False  # Stop if any cluster exceeds the MAX_DISTANCE
    return True  # Continue if all clusters meet the MAX_DISTANCE


# Use SciPy to calculate distance matrix
def distance_matrix(clusters):
    print("Computing distance matrix...")
    medoids = np.array([medoid(cluster) for cluster in clusters])  # Precompute medoids to avoid repeated calculations
    dist_matrix = distance.cdist(medoids, medoids, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)  # Set distances to itself to infinity
    return dist_matrix, medoids


def agglomerative_medoid(clusters):
    # Precompute the initial medoids and distance matrix
    dist_matrix, medoids = distance_matrix(clusters)
    previous_clusters = None

    while True:
        # Stop if point distances in clusters exceed the limit
        if len(clusters) < 30:
            if not evaluate_clusters(clusters):
                print("MAX_DISTANCE exceeded, stopping.")
                break
            previous_clusters = clusters[:]  # Save the last valid clustering state

        # Find the closest two clusters to merge
        cluster1, cluster2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge cluster2 into cluster1
        clusters[cluster1] = np.vstack([clusters[cluster1], clusters[cluster2]])
        del clusters[cluster2]  # Remove the merged cluster

        # Recompute the medoid for the new merged cluster
        medoids[cluster1] = medoid(clusters[cluster1])
        medoids = np.delete(medoids, cluster2, axis=0)

        # Remove cluster2 from the distance matrix and update distances for the new merged cluster
        dist_matrix = np.delete(dist_matrix, cluster2, axis=0)
        dist_matrix = np.delete(dist_matrix, cluster2, axis=1)
        dist_matrix[cluster1, :] = distance.cdist([medoids[cluster1]], medoids, metric='euclidean')
        dist_matrix[:, cluster1] = dist_matrix[cluster1, :]
        dist_matrix[cluster1, cluster1] = np.inf  # Set distance to itself to infinity

        print(f"Merged clusters {cluster1} and {cluster2}, Total clusters remaining: {len(clusters)}")

    return previous_clusters


def show_clusters(clusters):
    plt.figure(figsize=(8, 8))
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    colors = plt.colormaps['tab20']

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        medoid_value = medoid(cluster)

        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5,
                    color=colors(i),  # Get color from the colormap
                    label=f"Cluster {i + 1}")
        plt.scatter(medoid_value[0], medoid_value[1], s=100,
                    color=colors(i),
                    edgecolor='black', marker='o')

    plt.title("Agglomerative Clustering Visualization with Medoids and 20k points")
    plt.legend()
    plt.show()


def main():
    seed_num = 42
    random.seed(seed_num)
    np.random.seed(seed_num)
    clusters = init_points()
    final_clusters = agglomerative_medoid(clusters)
    show_clusters(final_clusters)


if __name__ == "__main__":
    main()
