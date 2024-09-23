"""
Algoritmo Constructivo Problema Vehicle
Routing Problem with Time Windows – VRPTW
Funcion De lectura
Juan Fernando Riascos Goyes
"""
import os
import matplotlib.pyplot as plt

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # First line: read n and Q
        first_line = lines[0].strip().split()
        n = int(first_line[0])
        Q = int(first_line[1])
        
        # Initialize lists to hold the data
        data = []
        
        # Read the subsequent lines
        for line in lines[1:n+2]:  # n+1 lines follow with node data
            parts = list(map(int, line.strip().split()))
            data.append({
                'index': parts[0],
                'x': parts[1],
                'y': parts[2],
                'demand': parts[3],
                'time_window_start': parts[4],
                'time_window_end': parts[5],
                'service_time': parts[6]
            })
        
    return n, Q, data

def process_directory(directory_path):
    all_data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            n, Q, data = read_txt_file(file_path)
            all_data[filename] = {
                'n': n,
                'Q': Q,
                'data': data
            }
            plot_data(data, filename)
    return all_data

def plot_data(data, filename):
    plt.figure(figsize=(15, 12))
    
    # Separate depot from other nodes
    depot = data[0]
    other_nodes = data[1:]
    
    # Plot depot
    plt.scatter(depot['x'], depot['y'], color='red', marker='s', label='Depot', s=100)
    
    # Plot other nodes with the same color
    for node in other_nodes:
        plt.scatter(node['x'], node['y'], color='blue', s=80)  # Use a single color for all nodes
    
    plt.title(f"Node Locations for {filename}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(['Depot', 'Nodes'])
    plt.grid(True)
    
    # Save plot as a PNG file
    plot_filename = filename.replace('.txt', '.png')
    plt.savefig(os.path.join(directory_path, plot_filename))
    plt.close()


# Example usage:
directory_path = r'D:\UNIVERSIDAD\OCTAVO SEMESTRE\Heuristica\Trabajo 1\Examples'
all_files_data = process_directory(directory_path)




