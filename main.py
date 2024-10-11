import matplotlib.pyplot as plt
import random

array = []
def create_point(x, y):
    plt.plot(x, y, 'o', markersize=1, markeredgecolor='black')

def init_points():
    global array
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        create_point(x, y)
        array.append([x, y])

def gen_offset(x, y):
    x_offset = max(-5000, min(5000, x + random.randint(-100, 100)))
    y_offset = max(-5000, min(5000, y + random.randint(-100, 100)))
    create_point(x_offset, y_offset)
    return [x_offset, y_offset]

def main_part():
    for i in range(20000):
        x, y = random.choice(array)
        new_point = gen_offset(x, y)
        create_point(new_point[0], new_point[1])
        array.append(new_point)


plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)
init_points()
main_part()
plt.show()
