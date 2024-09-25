## Juan Fernando Riascos Goyes
## ACO Heuristic Method for VRPTW problem 
## Libraries 


import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from openpyxl import Workbook
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

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        first_line = lines[0].strip().split()
        n = int(first_line[0])
        Q = int(first_line[1])
        nodes = []
        for line in lines[1:n+2]: 
            parts = list(map(int, line.strip().split()))
            node = Nodo(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
            nodes.append(node)
    return n, Q, nodes

def dist(node1, node2):
    return math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2)

def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = dist(nodes[i], nodes[j])
    return times

def is_feasible(route, new_node, capacity, times):
    total_demand = sum(node.demand for node in route) + new_node.demand
    if total_demand > capacity:  
        return False

    current_time = 0
    for i in range(1, len(route)):
        # Add travel  between nodes
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
        return False  

    return True  # Both capacity and time constraints are satisfied


def plot_routes(routes, filename, output_folder="ACO_images"):
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

    # Verificar si el nombre tiene .txt y reemplazar correctamente
    if '.txt' in filename:
        output_path = os.path.join(output_folder, filename.replace('.txt', '_solution.png'))
    else:
        output_path = os.path.join(output_folder, f'{filename}_solution.png')

    # Guardar la imagen en la carpeta especificada
    plt.savefig(output_path)
    plt.close()  # Cerrar la figura
 
def initialize_pheromones(num_nodes, times):
    
    pheromones = np.ones((num_nodes, num_nodes)) / (times + 1e-6)  # Feromonas inversamente proporcionales a la distancia
    return pheromones
def calculate_route_distance(route, times):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += times[route[i].index][route[i + 1].index]
    return distance

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
def travel_times_matrix(nodes):
    n = len(nodes)
    travel_times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            travel_times[i][j] = dist(nodes[i], nodes[j])
    return travel_times

def update_pheromones(pheromones, all_routes, Q, rho):
    """
    Actualiza las feromonas con evaporación y deposición.
    """
    pheromones *= (1 - rho)  # Evaporación
    for routes, distance in all_routes:
        for route in routes:
            for i in range(len(route) - 1):
                pheromones[route[i].index][route[i + 1].index] += Q / distance  

def calculate_total_distance(routes, times):
    return sum(calculate_route_distance(route, times) for route in routes)
# Definimos los 



aco_params = {
    'num_ants':50,
    'num_iterations': 100,
    'alpha': 1.5,
    'beta': 2,
    'rho': 0.7,
    'Q': 10.0
}




def aco_vrptw(nodes, capacity, times, num_ants, num_iterations, alpha, beta, rho, Q):
 
    num_nodes = len(nodes)
    pheromones = initialize_pheromones(num_nodes, times)
    best_routes = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        all_routes = []
        for ant in range(num_ants):
            depot = nodes[0]
            customers = set(range(1, num_nodes))  # Índices de clientes
            routes = []

            while customers:
                route = [depot]
                current_load = 0

                while True:
                    # Clientes factibles para la hormiga
                    feasible_customers = [cust for cust in customers if is_feasible(route, nodes[cust], capacity, times)]
                    if not feasible_customers:
                        break

                    # Cálculo de probabilidades basado en feromonas y visibilidad
                    probabilities = []
                    for cust in feasible_customers:
                        pheromone = pheromones[route[-1].index][cust]
                        travel_time = times[route[-1].index][cust]
                        visibility = 1 / (travel_time if travel_time > 0 else 1e-6)
                        probabilities.append((pheromone ** alpha) * (visibility ** beta))

                    total_prob = sum(probabilities)
                    probabilities = np.array(probabilities) / total_prob if total_prob > 0 else np.ones(len(feasible_customers)) / len(feasible_customers)
                    
                    # Seleccionar el próximo cliente basado en probabilidades
                    next_customer_index = np.random.choice(feasible_customers, p=probabilities)
                    next_customer = nodes[next_customer_index]

                    if current_load + next_customer.demand <= capacity:
                        route.append(next_customer)
                        current_load += next_customer.demand
                        customers.remove(next_customer_index)
                    else:
                        break

                route.append(depot)  # El vehículo regresa al depósito
                routes.append(route)

            total_distance = sum(calculate_total_distance([route], times) for route in routes)
            all_routes.append((routes, total_distance))

            # Actualizar la mejor solución si es mejor que la actual
            if total_distance < best_distance:
                best_distance = total_distance
                best_routes = routes

        # Actualizar feromonas
        update_pheromones(pheromones, all_routes, Q, rho)

    return best_routes, best_distance

# Función para guardar en Excel
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

def vrptw_solver(directory_path, output_filename):

    wb = Workbook()
    wb.remove(wb.active)

    execution_times = []  # Lista para guardar tiempos de ejecución
    output_folder = "ACO_images"  # Carpeta donde se guardarán las imágenes

    for i in range(1, 19):  # Procesar archivos VRPTW1 a VRPTW18
        filename = f'{directory_path}/VRPTW{i}.txt'
        file_start_time = time.time()  # Tiempo de inicio

        # Leer nodos y calcular la matriz de tiempos
        n, Q, nodes = read_txt_file(filename)
        times = travel_times_matrix(nodes)

        # Calcular las cotas inferiores (lower bounds)
        depot = nodes[0]
        customers = nodes[1:]
        lb_routes = lower_bound_routes(customers, Q)
        lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior

        # Aplicar ACO
        routes, best_distance = aco_vrptw(nodes, Q, times, **aco_params)
        computation_time = (time.time() - file_start_time) * 1000  # Tiempo en milisegundos
        execution_times.append(computation_time)

        # Calcular el GAP para número de rutas y distancia total
        actual_routes = len(routes)
        gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
        gap_distance = max(((best_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

        # Mostrar detalles de la solución
        base_filename = os.path.basename(filename)  # Extrae solo el nombre del archivo
        print(f"Solution for {base_filename}:")
        print(f"  - Total Distance = {best_distance}")
        print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
        print(f"  - GAP Distance = {gap_distance:.2f}")
        print(f"  - Actual Routes = {actual_routes}")
        print(f"  - Lower Bound Routes = {lb_routes}")
        print(f"  - GAP Routes = {gap_routes:.2f}")
        print(f"  - Execution Time = {computation_time:.0f} ms\n")

        # Guardar resultados en Excel
        sheet_name = f'VRPTW{i}'
        save_to_excel(wb, sheet_name, routes, best_distance, computation_time, times)

        # Llamar a plot_routes para graficar y guardar las rutas en la carpeta ACO_images
        plot_routes(routes, f"VRPTW{i}.txt", output_folder)

    # Guardar el archivo Excel
    wb.save(output_filename)

    total_elapsed_time = sum(execution_times)  # Tiempo total en milisegundos
    print(f"\nTotal execution time: {total_elapsed_time:.0f} ms")


# Ejemplo de uso
directory_path = "./Examples"  # Carpeta con los archivos de entrada
output_filename = "VRPTW_JuanFernando_ACO.xlsx"
vrptw_solver(directory_path, output_filename)


