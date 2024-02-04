import numpy as np
from matplotlib import pyplot as plt
import random




def main():
    
    T = 7000
    N = 10000
    coords = read_data("20cities.dat")
    matrix = distance_matrix(coords)

    time, length, all_temps, all_lengths, shortest_path = ann_metropolis(len(coords),matrix,T,N)
    shortest_path_len = calculate_len(shortest_path, matrix)
    shortest_path.append(shortest_path[0]) # Add the first city to the list in order to create a loop

    print(f"The final path takes: {time:.1f} minutes")
    print(f"The path is: {length:.1f} km")

    x = np.linspace(1,N*30, num=N*30)
    plt.plot(x, all_temps, label="Temperature")
    plt.plot(x, all_lengths, alpha=0.6, color="red", label="Total path length")
    plt.xlabel("Total runs")
    plt.ylabel("Temperature")
    plt.title("Traveling Salesman Problem using Simulated Annealing")
    plt.legend()
    plt.show()

    xs = []
    ys = []
    for coord in coords:
        xs.append(coord[0])
        ys.append(coord[1])
    plt.scatter(xs,ys)
    for i in range(len(shortest_path)-1):
        plt.plot([coords[shortest_path[i]][0], coords[shortest_path[i+1]][0]],
                 [coords[shortest_path[i]][1], coords[shortest_path[i+1]][1]], "k-")
    plt.title(f"Shortest path between cities ({shortest_path_len:.1f} km)")
    plt.xlabel("Km")
    plt.ylabel("Km")
    plt.show()
    
def read_data(x):
    data = [[] for _ in range(3)] # It is designed to read the data as presented in 20cities.dat, the third column is just discarded
    with open(x) as f:
        for line in f.readlines():
           x = line.split()
           for i in range(3):
               data[i].append(x[i])
    x_coor = list(map(float,data[0]))
    y_coor = list(map(float,data[1]))
    coords = list(zip(x_coor,y_coor))
    return coords

def distance_matrix(coords): # Create the matrix for distances between two points
    N = len(coords)
    matrix = np.zeros((N,N))
    for i in range(N):
        xy = coords[i]
        x1 = xy[0]
        y1 = xy[1]
        for j in range(N):
            xy = coords[j]
            x2 = xy[0]
            y2 = xy[1]
            d = np.sqrt((x2-x1)**2+(y2-y1)**2)
            matrix[i][j] = d
    return matrix

def calculate_len(path, matrix): # Calculate the length of a path
    length = 0
    N = len(path)
    city1 = path[0]
    for i in range(N):
        city2 = path[i]
        length += matrix[city1][city2]
        city1 = city2
    return length
        
def ann_metropolis(s,matrix,T, N):
    cpath = list(range(0,s))
    random.shuffle(cpath)
    all_lengths = []
    all_temps = []
    shortest_path = []
    shortest_length = 7000
    N_a = 0
    Tz = T
    while (N_a < 30):
        T_c = Tz
        for i in range(N):
            all_temps.append(Tz)
            Tz -= (T_c - T_c*(1/10))/N # Lower temperature to 1/10 during N loops
            cpath_length = calculate_len(cpath,matrix)
            all_lengths.append(cpath_length)
            tpath = exchange_values(cpath.copy())
            tpath_length = calculate_len(tpath,matrix)
            d_len = tpath_length-cpath_length
            if (d_len > 0):
                u = np.random.uniform()
                if (u < np.exp(-d_len/Tz)):
                    cpath = tpath.copy()
            else:
                cpath = tpath.copy()

            if(cpath_length < shortest_length):
                shortest_length = cpath_length
                shortest_path = cpath.copy()

        Tz *= 8 # Increase temp back to 8/10
        if (N_a%3 == 0): # Prints the progress
            print(f"{N_a/30*100:.1f}%")
        N_a += 1
        
    length = calculate_len(cpath,matrix)
    time = (s-1)*30 + (length/300)*60
    return time, length, all_temps, all_lengths, shortest_path

def exchange_values(path): # The trial state function

    i = random.randrange(len(path))
    j = i
    while (j == i):
        j = random.randrange(len(path))    

    copy = path[i]
    path[i] = path[j]             
    path[j] = copy

    return path



    
if __name__ == '__main__':
    main()
