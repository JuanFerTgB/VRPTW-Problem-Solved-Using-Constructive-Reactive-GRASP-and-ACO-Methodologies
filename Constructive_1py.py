## Juan Fernando Riascos Goyes
## Constructive Heuristic Method for VRPTW problem 
 
## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import time
from scipy.sparse.csgraph import minimum_spanning_tree

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
def euclidean_distance(node1, node2):
    return math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2)

## Function to calculate time travel (t_(i,j))
def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = euclidean_distance(nodes[i], nodes[j])
    return times


def is_feasible(route, new_node, capacity, times):
    # Check capacity feasibility
    total_demand = sum(node.demand for node in route) + new_node.demand
    if total_demand > capacity:  # Use '>' to strictly enforce the capacity constraint
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








## Constructive method to select the "optimal" route based on the above restrictions
def route_selection(nodes, capacity, times):
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
            next_customer = min(feasible_customers, key=lambda x: times[route[-1].index][x.index])
            if current_load + next_customer.demand <= capacity:
                route.append(next_customer)
                current_load += next_customer.demand
                customers.remove(next_customer)
            else:
                break
        route.append(depot)
        routes.append(route)

    return routes

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
def plot_routes(routes, filename, output_folder="constructive_images"):
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

def save_to_excel(workbook, sheet_name, routes, total_distance, computation_time, times):
    ws = workbook.create_sheet(title=sheet_name)

    # Primera fila con núme ro de vehículos, distancia total y tiempo de cómputo (convertido a milisegundos)
    num_vehicles = len(routes)
    computation_time_ms = round(computation_time * 1000, 0)  # Convertir a milisegundos
    ws.append([num_vehicles, round(total_distance, 3), computation_time_ms])

    # Filas siguientes con la información de cada vehículo
    for i, route in enumerate(routes, start=1):
        route_nodes = [0]  # Iniciar con el depósito
        arrival_times = []
        current_time = 0
        total_load = 0

        for j in range(1, len(route)):
            # Sumar el tiempo de viaje entre los nodos consecutivos
            current_time += times[route[j-1].index][route[j].index]

            # Si el vehículo llega antes de la ventana de tiempo, esperar hasta la hora mínima permitida
            if current_time < route[j].time_window[0]:
                current_time = route[j].time_window[0]  # Ajustar el tiempo de llegada

            # Registrar el tiempo de llegada (redondeado a 3 decimales)
            arrival_times.append(round(current_time, 3))

            # Sumar la demanda del nodo actual a la carga total del vehículo
            total_load += route[j].demand

            # Agregar el nodo actual a la lista de nodos de la ruta
            route_nodes.append(route[j].index)

            # Sumar el tiempo de servicio en el nodo actual
            current_time += route[j].serv_time

        # Al final de la ruta, el vehículo vuelve al depósito (nodo 0)
        route_nodes.append(0)

        # Calcular el número de clientes servidos en esta ruta (excluyendo el depósito)
        num_customers = len(route_nodes) - 3  # Restar los dos nodos del depósito (inicio y final)

        # Guardar el número de clientes, los nodos de la ruta, tiempos de llegada y la carga total
        ws.append([num_customers] + route_nodes + arrival_times + [total_load])


def lower_bound_routes(customers, vehicle_capacity):
    total_demand = sum(customer.demand for customer in customers)
    return math.ceil(total_demand / vehicle_capacity)


def lower_bound_mst(depot, customers, distance_matrix):
    nodes = [depot] + customers
    n = len(nodes)

    # Create a full distance matrix for all nodes
    full_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            full_matrix[i, j] = distance_matrix[nodes[i].index][nodes[j].index]

    mst = minimum_spanning_tree(full_matrix).toarray()

    mst_distance = mst.sum()

    return mst_distance


def vrptw_solver(output_filename):
    # Obtener la ruta del directorio donde se ejecuta el código
    directory_path = os.getcwd()  # Obtiene el directorio actual
    examples_path = os.path.join(directory_path, "Examples")  # Asume que hay una carpeta "Examples" en el directorio actual

    wb = Workbook()
    total_computation_time = 0  # Para calcular el tiempo total de ejecución

    # Recorrer los archivos numerados en orden
    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(examples_path, filename)
        
        if os.path.exists(file_path):  # Verifica que el archivo exista
            n, Q, nodes = read_txt_file(file_path)
            times = calculate_travel_times(nodes)

            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)
            lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior

            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            routes = route_selection(nodes, Q, times)
            computation_time = time.time() - start_time
            total_computation_time += computation_time

            total_distance = calculate_total_distance(routes, times)

            # Calcular el GAP para número de rutas y distancia total
            actual_routes = len(routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((total_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

            # Mostrar detalles de la solución
            print(f"Solution for {filename}:")
            print(f"  - Total Distance = {total_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar los resultados en una hoja de Excel con el nombre de la instancia
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, routes, total_distance, computation_time, times)

            plot_routes(routes, filename)
        else:
            print(f"Archivo {filename} no encontrado.")
    
    # Guardar el archivo de Excel con todas las hojas
    wb.save(output_filename)
    
    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")


# Ejemplo de uso
output_filename = "VRPTW_JuanFernando_Constructivo.xlsx"
vrptw_solver(output_filename)



