import math
import os
import time
import random
from itertools import combinations
from copy import deepcopy
from time import sleep
import tabulate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def read_optimal_cost(sol_file_path):
    with open(sol_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Cost"):
                return int(line.split()[1])
    return None


def parse_cvrp_file(file_path):
    data = {
        'name': None,
        'node_coords': {},
        'demands': {},
        'capacity': None,
        'depots': []
    }
    current_section = None


    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                data['name'] = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                current_section = 'node_coords'
            elif line.startswith("DEMAND_SECTION"):
                current_section = 'demands'
            elif line.startswith("CAPACITY"):
                data['capacity'] = int(line.split(":")[1].strip())
            elif line.startswith("DEPOT_SECTION"):
                current_section = 'depots'
            elif line.startswith("EOF"):
                break
            else:
                if current_section == 'node_coords':
                    parts = line.split()
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    data['node_coords'][node_id] = (x, y)
                elif current_section == 'demands':
                    parts = line.split()
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    data['demands'][node_id] = demand
                elif current_section == 'depots':
                    depot = int(line.strip())
                    if depot != -1:
                        data['depots'].append(depot)
    return data


def cost_function(routes: list, node_coords: dict, depot: int):
    distance = 0
    for route in routes:
        if not route:
            continue
        # distance from depo to first client
        distance += math.sqrt(
            (node_coords[route[0]][0] - node_coords[depot][0]) ** 2 +
            (node_coords[route[0]][1] - node_coords[depot][1]) ** 2
        )
        # distance between client
        for i in range(1, len(route)):
            distance += math.sqrt(
                (node_coords[route[i]][0] - node_coords[route[i-1]][0]) ** 2 +
                (node_coords[route[i]][1] - node_coords[route[i-1]][1]) ** 2
            )
        # distance from last client to depo
        distance += math.sqrt(
            (node_coords[route[-1]][0] - node_coords[depot][0]) ** 2 +
            (node_coords[route[-1]][1] - node_coords[depot][1]) ** 2
        )
    return distance


def nearest_neighbor_init(node_coords, demands, capacity, depot):
    routes = []
    unvisited = set(node for node in node_coords.keys() if node != depot)

    while unvisited:
        route = []
        current_capacity = 0
        current_node = depot
        
        while True:
            # nearest node
            nearest_node = None
            nearest_distance = float('inf')

            for node in unvisited:
                distance = math.sqrt((node_coords[node][0] - node_coords[current_node][0])**2 +
                                    (node_coords[node][1] - node_coords[current_node][1])**2)
                if distance < nearest_distance:
                    nearest_node = node
                    nearest_distance = distance
                
            
            if nearest_node is None:
                routes.append(route.copy())
                break 
            
            # add node to route or start new route

            if current_capacity + demands[nearest_node] <= capacity:
                route.append(nearest_node)
                current_capacity += demands[nearest_node]
                unvisited.remove(nearest_node)
                current_node = nearest_node
            else:
                routes.append(route.copy())
                route = [nearest_node]
                current_capacity = demands[nearest_node]
                current_node = depot
                unvisited.remove(nearest_node)

    return routes




def compute_distance_matrix(node_coords):

    n = len(node_coords)
    distance_matrix = {i: {} for i in node_coords}
    for i in node_coords:
        for j in node_coords:
            if i == j:
                distance_matrix[i][j] = 0
            else:
                dx = node_coords[i][0] - node_coords[j][0]
                dy = node_coords[i][1] - node_coords[j][1]
                distance = math.sqrt(dx**2 + dy**2)
                distance_matrix[i][j] = distance
    return distance_matrix

def clarke_wright_init(node_coords, demands, capacity, depot):

    # distance matrics
    distance_matrix = compute_distance_matrix(node_coords)
    
    # find savings exclude depo
    savings = []
    nodes = [node for node in node_coords if node != depot]
    for i, j in combinations(nodes, 2):
        saving = distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]
        savings.append((saving, i, j))
    
    # sort savings
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize routes with single node
    routes = [[node] for node in nodes]
    route_demands = [demands[node] for node in nodes]
    
    # merge routes
    for saving, i, j in savings:
        route_i = next((route for route in routes if i in route), None)
        route_j = next((route for route in routes if j in route), None)
        
        # if total demand do not exceed capacity = merge
        if route_i is not None and route_j is not None and route_i != route_j:
            total_demand = sum(demands[node] for node in route_i) + sum(demands[node] for node in route_j)
            if total_demand <= capacity:
                # merging
                new_route = route_i + route_j if route_i[-1] == i and route_j[0] == j else route_j + route_i
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
    
    final_routes = []
    for route in routes:
        final_routes.append(route)
    
    return final_routes



def two_opt(route, node_coords, depot):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_distance = cost_function([new_route], node_coords, depot)
                if new_distance < cost_function([route], node_coords, depot):
                    route = new_route
                    improved = True
    return route


def swap(route, node_coords, depot):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_distance = cost_function([new_route], node_coords, depot)
                if new_distance < cost_function([route], node_coords, depot):
                    route = new_route
                    improved = True
    return route


def three_opt(route, node_coords, depot):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                for k in range(j + 1, len(route)):
                    # variants of 3-opt
                    new_route_1 = route[:i] + route[j:k] + route[i:j] + route[k:]
                    new_route_2 = route[:i] + route[j:k][::-1] + route[i:j][::-1] + route[k:]
                    new_route_3 = route[:i] + route[j:k][::-1] + route[i:j] + route[k:]
                    new_route_4 = route[:i] + route[j:k] + route[i:j][::-1] + route[k:]

                    # use best variant
                    best_route = min(
                        [new_route_1, new_route_2, new_route_3, new_route_4],
                        key=lambda x: cost_function([x], node_coords, depot)
                    )
                    if cost_function([best_route], node_coords, depot) < cost_function([route], node_coords, depot):
                        route = best_route
                        improved = True
    return route

# Or-opt
def or_opt(route, node_coords, depot):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                for length in [1, 2, 3]:
                    if i + length > len(route):
                        continue
                    sequence = route[i:i + length]
                    new_route = route[:i] + route[i + length:]
                    for k in range(1, len(new_route)):
                        candidate = new_route[:k] + sequence + new_route[k:]
                        if cost_function([candidate], node_coords, depot) < cost_function([route], node_coords, depot):
                            route = candidate
                            improved = True
    return route

# Lin-Kernigan algortythm
def lin_kernighan(route, node_coords, depot):
    improved = True
    while improved:
        improved = False
        original_distance = cost_function([route], node_coords, depot)
        
        # 2-opt
        new_route = two_opt(route.copy(), node_coords, depot)
        if cost_function([new_route], node_coords, depot) < original_distance:
            route = new_route.copy()
            improved = True
            continue

        # 3-opt
        new_route = three_opt(route.copy(), node_coords, depot)
        if cost_function([new_route], node_coords, depot) < original_distance:
            route = new_route.copy()
            improved = True
            continue


        # Or-opt
        new_route = or_opt(route.copy(), node_coords, depot)
        if cost_function([new_route], node_coords, depot) < original_distance:
            route = new_route.copy()
            improved = True
            continue
    return route


def stochastic_phase(solution, demands, capacity):
    # Выбираем случайный оператор
    operator = random.choice(["swap", "relocate", "swapinroute", "cross_exchange"])
    
    if operator == "swap":
        # Swap Nodes
        route1, route2 = random.sample(range(len(solution)), 2)
        if len(solution[route1]) > 0 and len(solution[route2]) > 0:
            # two random nodes
            node1 = random.choice(solution[route1])
            node2 = random.choice(solution[route2])
            
            # Compute demands after swap
            demand1 = sum(demands[node] for node in solution[route1]) - demands[node1] + demands[node2]
            demand2 = sum(demands[node] for node in solution[route2]) - demands[node2] + demands[node1]
            
            # check demands and capacity
            if demand1 <= capacity and demand2 <= capacity:
                # swap
                solution[route1].remove(node1)
                solution[route2].remove(node2)
                solution[route1].append(node2)
                solution[route2].append(node1)

    
    elif operator == "relocate":
        # Relocate node
        route1, route2 = random.sample(range(len(solution)), 2)
        if len(solution[route1]) > 1:
            # choose random node
            node_index = random.randint(0, len(solution[route1]) - 1)
            node = solution[route1][node_index]            
            # 
            if sum(demands[node] for node in solution[route2]) + demands[node] <= capacity:
                # Random insert into route2
                solution[route1].pop(node_index)
                insert_index = random.randint(0, len(solution[route2]))
                solution[route2].insert(insert_index, node)


    elif operator == "swapinroute":
        # random swap inside route
        route1 = random.sample(range(len(solution)), 1)[0]
        if len(solution[route1]) > 1:
            from_index, to_index = random.sample(range(len(solution[route1])), 2)
            solution[route1][from_index], solution[route1][to_index] = solution[route1][to_index], solution[route1][from_index]


    elif operator == "cross_exchange":
        # Cross exchange
        route1, route2 = random.sample(range(len(solution)), 2)
        if len(solution[route1]) > 1 and len(solution[route2]) > 1:
            seq1_start, seq1_end = sorted(random.sample(range(len(solution[route1])), 2))
            seq2_start, seq2_end = sorted(random.sample(range(len(solution[route2])), 2))
            seq1 = solution[route1][seq1_start:seq1_end]
            seq2 = solution[route2][seq2_start:seq2_end]
            new_route1 = solution[route1][:seq1_start] + seq2 + solution[route1][seq1_end:]
            new_route2 = solution[route2][:seq2_start] + seq1 + solution[route2][seq2_end:]

            demand1 = sum(demands[node] for node in new_route1)
            demand2 = sum(demands[node] for node in new_route2)
            if demand1 <= capacity and demand2 <= capacity:
                # If change is valid
                solution[route1] = new_route1
                solution[route2] = new_route2



    return solution



def validate_solution(solution, demands, capacity):
    for route in solution:
        demand = sum(demands[node] for node in route)
        if demand > capacity:
            return False
    nodes = set()
    for route in solution:
        for node in route:
            nodes.add(node)
    if len(nodes) + 1 != len(demands): # check that all nodes are in solution
        return False


    return True


def combined_deterministic_annealing(node_coords,
                                     demands,
                                     capacity,
                                     depot, 
                                     initial_temperature, 
                                     cooling_rate, 
                                     min_temperature, 
                                     max_iterations):

    # Initial solution

    init_solution_1 = clarke_wright_init(node_coords, demands, capacity, depot)
    init_solution_2 = nearest_neighbor_init(node_coords, demands, capacity, depot)

    current_distance_1 = cost_function(init_solution_1, node_coords, depot)
    current_distance_2 = cost_function(init_solution_2, node_coords, depot)

    if True:
        initial_solution = [route.copy() for route in init_solution_1]
        initial_distance = current_distance_1
        current_solution = [route.copy() for route in init_solution_1]
        current_distance = current_distance_1
    else:
        initial_solution = [route.copy() for route in init_solution_2]
        initial_distance = current_distance_2
        current_solution = [route.copy() for route in init_solution_2]
        current_distance = current_distance_2

    # validate init solution 

    valid = validate_solution(current_solution, demands, capacity)

    if not valid:
        print("solution not valid")
        exit(1)

    best_solution = [route.copy() for route in current_solution]
    best_distance = current_distance

    temperature = initial_temperature
    iteration = 0


    while temperature > min_temperature and iteration < max_iterations:
        new_solution = [route.copy() for route in current_solution]

        new_solution = stochastic_phase(new_solution, demands, capacity)

        new_distance = cost_function(new_solution, node_coords, depot)
        # print(new_solution)
        # sleep(1)

        # stochastic phase
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_solution = [route.copy() for route in new_solution]
            current_distance = new_distance

        # Upd best solution
        if current_distance < best_distance:
            best_solution = [route.copy() for route in current_solution]
            best_distance = current_distance

        # Local search
        if temperature < 10 or iteration % 100 == 0:
            for i, route in enumerate(current_solution):
                current_solution[i] = lin_kernighan(route, node_coords, depot)

            current_distance = cost_function(current_solution, node_coords, depot)
            # Upd best solution
            if current_distance < best_distance:
                best_solution = [route.copy() for route in current_solution]
                best_distance = current_distance


        temperature *= cooling_rate

        iteration += 1


    return initial_solution, initial_distance, best_solution, best_distance




data_folder = "B"


initial_temperature = 1250
cooling_rate = 0.995
min_temperature = 1
max_iterations = 10000


count = 0
procent = 0

table_data = []


for filename in os.listdir(data_folder):
    if filename.endswith(".vrp"):
        count += 1
        vrp_file_path = os.path.join(data_folder, filename)
        sol_file_path = os.path.join(data_folder, filename.replace(".vrp", ".sol"))
        data = parse_cvrp_file(vrp_file_path)
        optimal_cost = read_optimal_cost(sol_file_path)

        start = time.time()

        initial_solution, initial_distance, best_routes, best_distance = combined_deterministic_annealing(data['node_coords'], 
                                                                data['demands'], 
                                                                data['capacity'], 
                                                                data['depots'][0], 
                                                                initial_temperature, 
                                                                cooling_rate, 
                                                                min_temperature, 
                                                                max_iterations)

        improvement_percent = 100 * (initial_distance - best_distance) / optimal_cost

        end = time.time()

        table_data.append([
            filename,
            optimal_cost,
            initial_distance,
            best_distance,
            f"{100 * (best_distance - optimal_cost) / optimal_cost:.2f}%",
            f"{improvement_percent:.2f}%",
            validate_solution(best_routes, data['demands'], data['capacity']),
            len(best_routes),
            end - start
        ])

        # visualize_routes(data['node_coords'], initial_solution, data['depots'][0], title="Пример маршрутов")
        # break



        # общий процент отклонения
        procent += 100 * (best_distance - optimal_cost) / optimal_cost
        

headers = [
    "Файл", "Оптимальная стоимость", "Начальная стоимость", "Лучшая стоимость", 
    "Отличие от оптимума", "Улучшение", "Валидность", "Кол-во маршрутов", "Время подсчёта"
]
print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))

# итоговое отличие от оптимума
print(f"\nИтого отличие от оптимума: {procent / count:.2f}%")



