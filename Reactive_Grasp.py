## Juan Fernando Riascos Goyes
## GRASP reactive Heuristic Method for VRPTW problem 
## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import random
import time 
from scipy.sparse.csgraph import minimum_spanning_tree

random.seed(2)  

## Class that represent a node for VRPTW problem (Vehicle Routing Problem with Time Windows).
## This node has this different variables:

## index (int): Index
## x_cord (int): X-coordinate of the node
## y_cord (int): Y-coordinate of the node
## demand (int): Demand of the node 
## inflim (int): Lower limit of the time window during which the node can be serviced
## suplim (int): Upper limit of the time window during which the node can be serviced
## serv (int): Service time 

class Nodo:

    def __init__(self, index, x_cord, y_cord, demand, inflim, suplim, serv):
        self.index = index
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.demand = demand
        self.time_window = (inflim, suplim)
        self.serv_time = serv

    def __repr__(self):
        return f"Customer <{self.index}>"
## Function to read the 18 .txt files (example problems with different properties)
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read n an Q
        first_line = lines[0].strip().split()
        n = int(first_line[0])
        Q = int(first_line[1])

        nodes = []

        # Read the n+1 next lines for index (i), x and y coordinate (x_i, y_i),
        # demand (q_i), Lower and upper limit for time window (e_i),(l_i),
        # and time service (s_i)
        
        for line in lines[1:n+2]: 
            parts = list(map(int, line.strip().split()))
            node = Nodo(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
            nodes.append(node)
    return n, Q, nodes

## Time of travel (Define by Euclidean Distance)
## Function given by teacher at [1]
def euclidean_distance(node1, node2):
    return round(math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2), 3)

## Function to calculate time travel (t_(i,j))
## Function given by teacher at [1]
def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = euclidean_distance(nodes[i], nodes[j])
    return times

## Restrictions (CONSTRUCTIVE METHOD)
def is_feasible(route, new_node, capacity, times):
    # Check capacity feasibility
    total_demand = sum(node.demand for node in route) + new_node.demand
    if total_demand > capacity:  
        return False

    current_time = 0
    for i in range(1, len(route)):
        current_time += times[route[i-1].index][route[i].index]
        
        if current_time < route[i].time_window[0]:
            current_time = route[i].time_window[0]
        
        if current_time > route[i].time_window[1]:
            return False
        
        current_time += route[i].serv_time

    new_node_arrival_time = current_time + times[route[-1].index][new_node.index]

    if new_node_arrival_time < new_node.time_window[0]:
        new_node_arrival_time = new_node.time_window[0]
    if new_node_arrival_time > new_node.time_window[1]:
        return False  # Arrival too late for the new node

    return True  


# Función para seleccionar rutas usando GRASP Reactivo

## Math model for reactive grasp
def reactive_grasp_route_selection(nodes, capacity, times, alphas=[0.03, 0.05, 0.10, 0.11, 0.12], iterations=100):
    alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  # Probability of \alpha
    best_routes = None
    best_distance = float('inf')
    min_prob = 1e-6  # Umbral mínimo para probabilidades

    for _ in range(iterations):
        # \alpha selection from alpha probs
        alpha = random.choices(list(alpha_probs.keys()), weights=alpha_probs.values())[0]
        depot = nodes[0]
        customers = nodes[1:]
        routes = []   
        while customers:
            route = [depot]
            current_load = 0
            while True:
                feasible_customers = [cust for cust in customers if is_feasible(route, cust, capacity, times)]
                if not feasible_customers:
                    break
                # 
                feasible_customers.sort(key=lambda x: times[route[-1].index][x.index])
                # RCL LIST
                rcl_size = max(1, int(len(feasible_customers) * alpha))
                rcl = feasible_customers[:rcl_size]
                # Select customer on RCL
                next_customer = random.choice(rcl)
                if current_load + next_customer.demand <= capacity:
                    route.append(next_customer)
                    current_load += next_customer.demand
                    customers.remove(next_customer)
                else:
                    break
            route.append(depot)
            routes.append(route)

        # TOTAL DISTANCE
        total_distance = calculate_total_distance(routes, times)
        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = routes

        # Check probs 
        for alpha_key in alpha_probs:
            if alpha_key == alpha:
                alpha_probs[alpha_key] += 1 / (1 + total_distance - best_distance)
            else:
                alpha_probs[alpha_key] = max(min_prob, alpha_probs[alpha_key] - 1 / (1 + total_distance - best_distance))
        
        # Normal
        total_prob = sum(alpha_probs.values())
        if total_prob == 0 or total_prob != total_prob:  
            alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  
        else:
            alpha_probs = {k: v / total_prob for k, v in alpha_probs.items()}

    return best_routes, best_distance

## FUNCTION FROM CONSTRUCTIVE METHOD 
## Calculate the route distance for a route in 
def calculate_route_distance(route, times):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += times[route[i].index][route[i + 1].index]
    return distance

## Sum of the distances calculated above 
def calculate_total_distance(routes, times):
    return sum(calculate_route_distance(route, times) for route in routes)

## Plot Solutions with node numeration and with different colors depending the route
def plot_routes(routes, filename, output_folder="Grasp_reactive_images"):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generar la figura de la ruta
    plt.figure(figsize=(10, 8))
    
    for route in routes:
        x_coords = [node.x_cord for node in route]
        y_coords = [node.y_cord for node in route]
        plt.plot(x_coords, y_coords, marker='o')
        for i, node in enumerate(route):
            plt.text(node.x_cord, node.y_cord, str(node.index), fontsize=12, ha='right')
    
    plt.title(f"VRPTW Solution: {filename}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    # Guardar la imagen en la carpeta especificada
    output_path = os.path.join(output_folder, filename.replace('.txt', '_solution.png'))
    plt.savefig(output_path)
    plt.show()

## SAVE TO EXCEL FILE CODE from [2]
# Función para guardar en Excel
def save_to_excel(workbook, sheet_name, routes, total_distance, computation_time, times):
    ws = workbook.create_sheet(title=sheet_name)

    # Convertir el tiempo de cómputo a milisegundos
    computation_time_ms = round(computation_time * 1000, 0)

    # Primera fila con número de vehículos, distancia total y tiempo de cómputo en milisegundos
    num_vehicles = len(routes)
    ws.append([num_vehicles, round(total_distance, 0), computation_time_ms])

    for route in routes:
        route_nodes = [0] 
        arrival_times = []
        current_time = 0
        total_load = 0

        for j in range(1, len(route)):
            current_time += times[route[j-1].index][route[j].index]
            if current_time < route[j].time_window[0]:
                current_time = route[j].time_window[0]
            arrival_times.append(round(current_time, 3))  
            total_load += route[j].demand
            route_nodes.append(route[j].index)

            current_time += route[j].serv_time

        route_nodes.append(0)  

        # Almacenar la información en el archivo Excel
        ws.append([len(route_nodes) - 3] + route_nodes + arrival_times + [total_load])

def lower_bound_routes(customers, vehicle_capacity):
    total_demand = sum(customer.demand for customer in customers)
    return math.ceil(total_demand / vehicle_capacity)


def lower_bound_mst(depot, customers, distance_matrix):
    # Combine the depot and customer nodes to form a full graph
    nodes = [depot] + customers
    n = len(nodes)

    # Create a full distance matrix for all nodes
    full_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            full_matrix[i, j] = distance_matrix[nodes[i].index][nodes[j].index]

    # Compute the minimum spanning tree (MST) of the full graph
    mst = minimum_spanning_tree(full_matrix).toarray()

    # Sum of the edges in the MST gives the lower bound distance
    mst_distance = mst.sum()

    return mst_distance




def vrptw_solver(directory_path, output_filename):
    wb = Workbook()
    wb.remove(wb.active)

    results = []
    total_computation_time = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            n, Q, nodes = read_txt_file(file_path)
            
            times = calculate_travel_times(nodes)
            
            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)  # Cota inferior para las rutas
            lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior de la distancia
            
            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            routes, best_distance = reactive_grasp_route_selection(nodes, Q, times)
            computation_time = time.time() - start_time
            total_computation_time += computation_time
            
            # Calcular el GAP para número de rutas y distancia total
            actual_routes = len(routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((best_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)
            
            #Mostrar detalles de la solución (tomado del primer solver)
            print(f"Solution for {filename}:")
            print(f"  - Total Distance = {best_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")
            
            # Almacenar los resultados en una lista
            results.append((filename, best_distance, computation_time))
           
            sheet_name = filename.split('.')[0]
            save_to_excel(wb, sheet_name, routes, best_distance, computation_time, times)
            
            plot_routes(routes, filename)
    
    wb.save(output_filename)
    
    # Ordenar los resultados por nombre de archivo para asegurarse que estén en orden
    results.sort(key=lambda x: int(x[0].split('VRPTW')[1].split('.txt')[0]))

    # Imprimir los resultados en orden y verificar el tiempo de cómputo en milisegundos
    for result in results:
        computation_time_ms = round(result[2] * 1000, 0)
        print(f"Solution for {result[0]}: Total Distance = {result[1]}, Computation Time = {computation_time_ms} ms")
    
    # Imprimir el tiempo total de ejecución al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")


# Ejemplo de uso
directory_path = "./Examples"
output_filename = "VRPTW_JuanFernando_Reactive_GRASP.xlsx"
vrptw_solver(directory_path, output_filename)




