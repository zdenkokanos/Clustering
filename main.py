import matplotlib.pyplot as plt
import numpy as np
import random

#numpy.mean vsade kde sa da

ITERATIONS = 20000
INIT_POINTS = 20

def offset_calculation(coordinate):
    offset = 100
    if 5000 - abs(coordinate) < 100:
        offset = (5000 - abs(coordinate)) // 2
    return offset


def gen_offset(x, y):
    x_offset = offset_calculation(x)
    y_offset = offset_calculation(y)
    x = x + random.randint(-x_offset, x_offset)
    y = y + random.randint(-y_offset, y_offset)
    return np.array([x, y])  # Return as a NumPy array

def init_points():
    coordinates_set = set()  # Use a set to store unique points
    coordinates_array = []  # Use a list to store points as NumPy arrays

    # Initialize with random unique points
    while len(coordinates_set) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        point = np.array([x, y])  # Create a NumPy array
        coordinates_set.add(tuple(point))  # Store as tuple for uniqueness check
        coordinates_array.append(point)  # Store as NumPy array

    # Generate new points until the desired total count is reached
    while len(coordinates_array) < ITERATIONS + INIT_POINTS:
        random_coordinate = random.choice(coordinates_array)
        x, y = random_coordinate
        new_point = gen_offset(x, y)

        # Check for uniqueness using the set
        if tuple(new_point) not in coordinates_set:
            coordinates_set.add(tuple(new_point))  # Add as a tuple for the set
            coordinates_array.append(new_point)  # Keep the new point as a NumPy array

    return np.array(coordinates_array)  # Convert the list of arrays to a NumPy array

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def distance_matrix(coordinates):
    num_points = coordinates.shape[0]  # returns a number of rows
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):  # only calculate for j > i
            dist = euclidean_distance(coordinates[i], coordinates[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # sets the symmetric entry
    return dist_matrix

def agglomerative_centroid(clusters):
    dist_matrix = distance_matrix(clusters)
    print(clusters)



def show_clusters(coordinates_array):
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.scatter(coordinates_array[:, 0], coordinates_array[:, 1], s=1, edgecolor='black')  # Correctly plot x and y
    plt.show()
    print(len(coordinates_array))

def main():
    random.seed(50)  # Set seed for the random module
    np.random.seed(50)  # Set seed for Numpy's random module
    coordinates = init_points()
    agglomerative_centroid(coordinates)
    show_clusters(coordinates)


if __name__ == "__main__":
    main()
