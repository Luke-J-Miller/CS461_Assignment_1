#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math #Used for calculating distances
try:
    import networkx as nx #used to create graphs
    import matplotlib.pyplot as plt #used to draw graphs
    import timeit #used to time the duration of functions
    import pandas as pd #used to draw tables
    special_libraries_loaded = True
except:
    special_libraties_loaded = False
from typing import Dict, List, Tuple #type hinting
from queue import PriorityQueue #used with A*
import inspect #used when debug is set to True

debug = False #when True, prints function and line whereever debug_trace(optional: message) is called
coord_file = "coordinates.txt" #name of the file with coordinates
adjacency_file = "Adjacencies.txt" #name of the file with adjacencies


# In[2]:


def debug_trace(msg: str = "") -> None:
    """
    When debug is set to True, this function prints the name of the function and the line number where it is called.
    
    Args:
        msg (str): Optional message to print
        
    Returns:
        nothing
    """

    if debug:
        calling_frame = inspect.stack()[1]
        calling_function = calling_frame.function
        calling_lineno = calling_frame.lineno
        print(f"Debugging: {calling_function} line {calling_lineno}: {msg}")


# In[3]:


def read_coordinate_file(filename: str) -> dict:
    """
    Reads a file containing city coordinates and returns a dictionary of
    city names to (longitude, latitude) coordinate pairs.

    Args:
        filename (str): the name of the file to read

    Returns:
        coord_dict: a dictionary mapping city names to (longitude, latitude) coordinate pairs
    """
    debug_trace()
    with open(filename) as coord_file:
        coordinates = [line.strip() for line in coord_file.readlines() if line.strip()]
    
    coord_dict = {}
    for line in coordinates:
        if line == "":
            continue
        city, lat, long = line.strip().lower().split()
        coord_dict[city] = (float(long), float(lat))
    
    debug_trace()
    return coord_dict


# In[4]:


def read_adjacency_file(filename: str) -> dict:
    """
    Reads a file containing on each line a city followed by cities adjacent to it and returns a dictionary mapping
    each city to a list of its neighboring cities.

    Args:
        filename (str): the name of the file to read

    Returns:
        city_dict: a dictionary mapping each city to a list of its neighboring cities
    """
    debug_trace()
    city_dict = {}
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cities = line.strip().lower().split()
            city = cities[0]
            adjacent_cities = set(cities[1:])
            city_dict.setdefault(city, set()).update(adjacent_cities)
            for adjacent_city in adjacent_cities:
                city_dict.setdefault(adjacent_city, set()).add(city)
    return city_dict


# In[5]:


def compare_dictionaries(dict1: dict, dict2: dict) -> (bool, set):
    """
    Built this during debugging to ensure both dictionaries matched.  
    Not used in final product
    
    Args:
        dict1 (dict):
        dict2 (dict):
        
        
    Returns:
        a boolean if the keys match
        differences between the dicitonaries
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    all_keys = keys1 | keys2
    same_keys = keys1 & keys2
    diff_keys = all_keys - same_keys
    
    differences = {key: (dict1.get(key, None), dict2.get(key, None)) for key in diff_keys}

    return (keys1 == keys2, differences)


# In[6]:


def distance_calc(source: tuple, dest: tuple) -> float:
    """
    given tuples of lat/longs, calculates the distance
    
    Args:
        source: tuple of lat, long
        dest: tuple of lat, long
        
    Returns:
        float distance
    """
    debug_trace()
    x1, y1 = source
    x2, y2 = dest
    distance = math.sqrt(54*((x1-x2)**2) + 69*((y1-y2)**2))
    debug_trace()
    return distance


# In[7]:


def build_graph(coord_dict: dict, adjacency_dict: dict):
    """
    Uses networkx to build a graph.
    
    Args:
        coord_dict: dict
        adjacency_dict: dict
        
    returns:
        networksx graph
    """
    debug = True
    debug_trace()
    City_graph = nx.Graph()
    for city, longlat in coord_dict.items():
        if city == "":
            continue
        debug_trace(f"{city} {longlat}")
        long, lat = longlat
        City_graph.add_node(city, pos = (long, lat))
    for city, adjacencies in adjacency_dict.items():
        for adjacency in adjacencies:
            City_graph.add_edge(city, adjacency)
    debug_trace()
    return City_graph


# In[8]:


def draw_graph(graph, coord_dict, highlighted_cities=[]):
    """
    Given a graph, it displays it with nodes plotted to coordinates and selected cities highlighted.
    
    Args:
        graph: networkx graph
        coord_dict: dict
        highlighted_cities: list
        
    """
    debug_trace()
    color_map = []
    for node in graph.nodes():
        debug_trace(node)
        if node == "":
            continue
        if node in highlighted_cities:
            color_map.append("yellow")
        else:
            color_map.append("green")
    options = {
        "with_labels": True,
        "font_size": 6,
        "node_size": 30,
        "node_color": color_map,
        "edge_color": (.8, .8, .8)
    }
    nx.draw(graph, coord_dict,
            with_labels=True,
            font_size=6,
            node_size=30,
            node_color=color_map,
            edge_color=(.8, .8, .8))
    debug_trace()


# In[9]:


def best_first_search(start_city: str, dest_city: str, by_miles: bool,
                      adjacency_dict: Dict[str, list], coord_dict: Dict[str, Tuple[float, float]]) -> Tuple[float, list]:
    """Finds path by selecting next city closest to destination. 
    Backtracks if hits dead-end. 
    Returns tuple (distance, path_list)"""
    debug_trace()
    
    best_first_path = [start_city]
    path_length = 0
    popped_neighbors = []  # Initialize outside the loop
    while best_first_path:
        current_city = best_first_path[-1]
        if current_city == dest_city:
            debug_trace()
            path_length = 0
            if by_miles:
                for i in range(len(best_first_path)-1):
                    path_length += distance_calc(coord_dict[best_first_path[i]], coord_dict[best_first_path[i+1]])
            else:
                path_length = len(best_first_path)-1
            return path_length, best_first_path
        
        neighbors = []
        temp_neighbors = adjacency_dict[current_city]
        min_neighbor = None
        min_distance = None
        for neighbor in temp_neighbors:
            if neighbor in best_first_path or neighbor in popped_neighbors:
                debug_trace()
                continue
            else:
                neighbors.append(neighbor)
                debug_trace()

                if min_neighbor == None:
                    debug_trace()
                    min_neighbor = neighbor
                    min_distance = distance_calc(coord_dict[neighbor], coord_dict[dest_city])
                else:
                    debug_trace()
                    next_distance = distance_calc(coord_dict[neighbor], coord_dict[dest_city])
                    if min_distance > next_distance:
                        min_neighbor, min_distance = neighbor, next_distance

        if not neighbors:
            debug_trace()
            popped_neighbors.append(best_first_path[-1])
            best_first_path.remove(best_first_path[-1])
        else:
            best_first_path.append(min_neighbor)
            debug_trace()

            if by_miles:
                path_length += distance_calc(coord_dict[current_city], coord_dict[min_neighbor])
            else:
                path_length += 1
    
    print("Unable to find path")
    return None, None


# In[10]:


def a_star_search(start_city: str, dest_city: str, by_miles: bool,
                  adjacency_dict: Dict[str, List[str]], coord_dict: Dict[str, Tuple[float, float]]) -> Tuple[float, List[str]]:
    """Perform A* search to find the shortest path between two cities.

    Args:
        start_city (str): The name of the starting city.
        dest_city (str): The name of the destination city.
        by_miles (bool): Whether to calculate distances in miles (True) or kilometers (False).
        adjacency_dict (dict): A dictionary of city names and their adjacent cities.
        coord_dict (dict): A dictionary of city names and their coordinates.

    Returns:
        A tuple containing the total cost of the shortest path and a list of the cities in the path.
    """
    debug_trace()
    pq = PriorityQueue()
    start_distance = distance_calc(coord_dict[start_city], coord_dict[dest_city])
    pq.put((start_distance, start_city))
    if start_city not in adjacency_dict:
        raise ValueError(f"Start city {start_city} not found in adjacency dictionary")
    if dest_city not in adjacency_dict:
        raise ValueError(f"Destination city {dest_city} not found in adjacency dictionary")
    while not pq.empty():
        cost, path = pq.get()
        current_city = path.split()[-1]
        if current_city == dest_city:
            path_list = path.split()
            cost = 0
            if by_miles:
                for i in range(len(path_list)-1):
                    cost += distance_calc(coord_dict[path_list[i]], coord_dict[path_list[i+1]])
            else:
                cost = len(path_list)-1
            
            return cost, path.split()
        for neighbor in adjacency_dict[current_city]:
            neighbor_path = f"{path} {neighbor}"
            neighbor_cost = cost + distance_calc(coord_dict[neighbor], coord_dict[dest_city])
            pq.put((neighbor_cost, neighbor_path))
    raise ValueError(f"No path found from {start_city} to {dest_city}")


# In[11]:


def get_inputs(city_dict: Dict[str, any]) -> Tuple[str, str]:
    """
    Prompts the user to enter a start city and a destination city, converts the entries to lower case
    and replaces spaces with underscores. Checks if the entries are in the provided dictionary.
    If an entry is not found, prompts the user to try again. If the user types "exit()", terminates
    the program.

    Args:
        city_dict: A dictionary that maps city names to some values.

    Returns:
        A tuple of two strings: the start city and the destination city.

    Raises:
        SystemExit: If the user enters "exit()" to terminate the program.
    """
    debug_trace()
    while True:
        print("type \"exit()\" at any time to exit")
        start_city = input("Enter a start city: ").strip().lower().replace(" ", "_")
        print()
        if start_city == "exit()":
            exit()
        if start_city in city_dict:
            break
        print("City not found. Please try again.")

    while True:
        dest_city = input("Enter a destination city: ").strip().lower().replace(" ", "_")
        print()
        if dest_city == "exit()":
            exit()
        if dest_city in city_dict:
            break
        print("City not found. Please try again.")
    debug_trace()
    print()
    if special_libraries_loaded:
        print("""This program runs each alogorithm a thousand times to measure speed.  
        Expect to wait 5 seconds after destination city.""")
    print()
    return start_city, dest_city



# In[12]:


def run_setup(coord_file: str, adjacency_file: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict[str, float]], nx.Graph, str, str]:
    """
    Reads in a coordinate file and an adjacency file, builds a graph, prompts the user for start and destination cities,
    and returns relevant data.

    Args:
        coord_file: A string representing the path to the coordinate file.
        adjacency_file: A string representing the path to the adjacency file.

    Returns:
        A tuple of five elements: 
            - A dictionary mapping city names to coordinate pairs
            - A dictionary of adjacency information for the graph
            - A networkx graph representing the city data
            - The start city string
            - The destination city string
    """
    debug_trace()
    coord_dict = read_coordinate_file(coord_file)
    adjacency_dict = read_adjacency_file(adjacency_file)
    city_graph = build_graph(coord_dict, adjacency_dict)
    start_city, dest_city = get_inputs(city_graph)
    debug_trace()
    return coord_dict, adjacency_dict, city_graph, start_city, dest_city
    


# In[13]:


def path_find(coord_dict: Dict[str, Tuple[float, float]], 
              adjacency_dict: Dict[str, Dict[str, float]], 
              city_graph: nx.Graph, 
              start_city: str, 
              dest_city: str) -> None:
    """
    Finds the shortest path between two cities using a graph search algorithm.
    
    Args:
        coord_dict: A dictionary that maps city names to (latitude, longitude) coordinate tuples.
        adjacency_dict: A dictionary that encodes the edges in the graph as a mapping between node names and their neighbors.
        city_graph: A networkx graph representing the city data.
        start_city: The name of the starting city.
        dest_city: The name of the destination city.
        by_edges: If True, run the search algorithm using the number of edges as the cost function.
        by_miles: If True, run the search algorithm using the distance between cities as the cost function.
    
    Returns:
        None
    
    """
    bfe_dist, bfe_path = best_first_search(start_city, dest_city, False, adjacency_dict, coord_dict)
    ase_dist, ase_path = a_star_search(start_city, dest_city, False, adjacency_dict, coord_dict)
    bfm_dist, bfm_path = best_first_search(start_city, dest_city, True, adjacency_dict, coord_dict)
    asm_dist, asm_path = a_star_search(start_city, dest_city, True, adjacency_dict, coord_dict)
    print()
    if special_libraries_loaded:
        ase_time = timeit.timeit(lambda: a_star_search(start_city, dest_city, False, adjacency_dict, coord_dict), number=1000)
        bfe_time = timeit.timeit(lambda: best_first_search(start_city, dest_city, False, adjacency_dict, coord_dict), number=1000)
        bfm_time = timeit.timeit(lambda: best_first_search(start_city, dest_city, True, adjacency_dict, coord_dict), number=1000)
        asm_time = timeit.timeit(lambda: a_star_search(start_city, dest_city, True, adjacency_dict, coord_dict), number=1000)

        results_dict = {"Algorithm": ["Best-first by edges", "A* by edges", "Best-first by miles", "A* by miles"],
                       "Distance": [bfe_dist, ase_dist, bfm_dist, asm_dist], 
                        "Avg. Time (ms)": [bfe_time, ase_time, bfm_time, asm_time]
                       }
        df = pd.DataFrame(results_dict)
        display(df)
    else:
        print(f"""
        Best first by edges had a length of {bfe_dist} edges.
        A* by edges had a lenght of         {ase_dist} edges
        
        Best first by miles had a length of {bfm_dist} miles.
        A* by miles has a length of         {asm_dist} miles
        
        """)

    print()
    print()
    print("Best-First search by number of edges")
    print(bfe_path)
    if special_libraries_loaded:
        plt.figure(1)
        draw_graph(city_graph, coord_dict, bfe_path)
        plt.show()
    print()
    print("A* search by number of edges")
    print(ase_path)
    if special_libraries_loaded:
        plt.figure(2)
        draw_graph(city_graph, coord_dict, ase_path)
        plt.show()
    print()
    print("Best-First search by number of miles")
    print(bfm_path)
    if special_libraries_loaded:
        plt.figure(3)
        draw_graph(city_graph, coord_dict, bfm_path)
        plt.show()
    print()
    print("A* search by number of miles")
    print(asm_path)
    if special_libraries_loaded:
        plt.figure(4)
        draw_graph(city_graph, coord_dict, ase_path)
        plt.show()
    print()


# In[ ]:


print("Welocme to the Southern Kansas Mapper")
while True:
    coord_dict, adjacency_dict, city_graph, start_city, dest_city =  run_setup(coord_file, adjacency_file)
    path_find(coord_dict, adjacency_dict, city_graph, start_city, dest_city)
    print()
    print()


# In[ ]:




