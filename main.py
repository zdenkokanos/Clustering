import matplotlib.pyplot as plt
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

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def centroid(cluster):
    return np.mean(cluster, axis=0)

def evaluate_clusters(clusters):
    for cluster in clusters: 
        centroid_value = centroid(cluster)  
        distances = []  
        for point in cluster:  
            distance = euclidean_distance(point, centroid_value) 
            distances.append(distance) 
        avg_distance = np.mean(distances)  
        if avg_distance > MAX_DISTANCE:  
            return False  # Stop if any cluster exceeds the MAX_DISTANCE and revert to the previous iteration
    return True  # Continue if all clusters meet the MAX_DISTANCE

def distance_matrix(clusters):
    print("Computing distance matrix...")
    num_clusters = len(clusters)
    dist_matrix = np.full((num_clusters, num_clusters), np.inf)

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist_matrix[i, j] = euclidean_distance(centroid(clusters[i]), centroid(clusters[j]))
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def agglomerative_centroid(clusters):
    dist_matrix = distance_matrix(clusters)  # Initialize the distance matrix
    previous_clusters = None
    while True:
        if len(clusters) < 30:
            if not evaluate_clusters(clusters):
                print("MAX_DISTANCE exceeded stopping.")
                break  # Stop if any cluster exceeds the MAX_DISTANCE
            previous_clusters = clusters.copy()  # keep the last option where distance between points in clusters were < 500
        # Find the two closest clusters
        cluster1, cluster2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge the two closest clusters
        print(cluster1, cluster2, len(clusters))
        clusters[cluster1] = np.vstack([clusters[cluster1], clusters[cluster2]])
        del clusters[cluster2]  # Remove the merged cluster

        dist_matrix = np.delete(dist_matrix, cluster2, axis=0)  # Remove the row containing cluster2
        dist_matrix = np.delete(dist_matrix, cluster2, axis=1)  # Remove the column containing cluster2

        # Set the distance from the merged cluster to itself to inf
        dist_matrix[cluster1, cluster1] = np.inf
        new_centroid = centroid(clusters[cluster1])
        for i in range(len(clusters)):  # Update distances to other clusters
            if i != cluster1:  # Avoid updating the distance to itself
                dist_matrix[cluster1, i] = euclidean_distance(new_centroid, centroid(clusters[i]))
                dist_matrix[i, cluster1] = dist_matrix[cluster1, i]
        print(f"Merged clusters {cluster1} and {cluster2}, Total clusters remaining: {len(clusters)}")
    final_centroids = [centroid(previous_clusters) for previous_clusters in clusters]
    return np.array(final_centroids), previous_clusters


def show_clusters(clusters):
    plt.figure(figsize=(8, 8))
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    # Use a colormap to generate distinct colors
    num_clusters = len(clusters)
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
    seed_num = 48
    random.seed(seed_num)
    np.random.seed(seed_num)
    clusters = init_points()
    final_centroids, final_clusters = agglomerative_centroid(clusters)
    show_clusters(final_clusters)


if __name__ == "__main__":
    main()
