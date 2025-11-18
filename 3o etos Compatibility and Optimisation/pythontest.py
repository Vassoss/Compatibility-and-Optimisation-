import json # Importing the JSON file that has all the Nodes, Edges, Weights (The Winter Region (map))
import heapq  # Importing heapq gor priority queue
import matplotlib.pyplot as plt

# Creating a class for the Nodes
class Node:
    def __init__(self, name, x=0, y=0):
        self.name = name       # Node Name
        self.edges = []        # Making a list to store all the Edges connected to the Node
        self.x = x             # X coordinate for visualization of the map
        self.y = y             # Y coordinate for visualization of the map

# Creating a class for the Edges
class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node    #Start Node
        self.to_node = to_node        #End Node
        self.weight = weight          #Weight of the road
        self.used = False             #Tracking if edge has been traversed


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges= []        #List of all Edges

    def add_node(self, name):
        if name not in self.nodes:         #No Duplicates
            self.nodes[name] = Node(name)

    def add_edge(self, u, v, weight):

          #New Edge between nodes u and v with given weight
        if not any((e.from_node.name == v and e.to_node.name == u) or
                   (e.from_node.name == u and e.to_node.name == v) for e in self.edges):
            
            #Check if edge already exists
            edge = Edge(self.nodes[u], self.nodes[v], weight)
            self.edges.append(edge)
            self.nodes[u].edges.append(edge)
            self.nodes[v].edges.append(edge)


# Dijkstra Function for shortest path
def dijkstra(graph, start, end):

    # Initialize distances
    distances = {node_name: float('inf') for node_name in graph.nodes}  # Start every Node to infinity
    distances[start] = 0  # Distance to itself

    # Previous nodes
    previous = {node_name: None for node_name in graph.nodes}

    # Priority queue
    pq = [(0, start)]

    
    # While there are Nodes left in the queue
    while pq:
        current_dist, current_node = heapq.heappop(pq) # Pop from priority queue

        if current_dist > distances[current_node]: # Skip if already explored
            continue

        # Explore neighbor nodes of the nodes that is currently popped
        for edge in graph.nodes[current_node].edges:
            neighbor = edge.to_node.name if edge.from_node.name == current_node else edge.from_node.name # Determine neighbor Node (the other end of the edge)
            distance = current_dist + edge.weight # Calculate the distance to the neighbor Node

            # If this new distance is better update
            if distance < distances[neighbor]:
                distances[neighbor] = distance # Shortest distance so far
                previous[neighbor] = current_node # Previous Node which came from current Node
                heapq.heappush(pq, (distance, neighbor)) # Push Neighbor into the  priority queue

    # Reconstruct path
    path = []
    node = end
    while node:
        path.insert(0, node)
        node = previous[node]

    return distances[end], path # Returns values (total grit cost and ordered list of nodes(path))


# Function (Finds which Nodes have an odd number of roads connected to them)  ---> for Eulerian loop later
def get_odd_degree_nodes(graph):
    odd_nodes = [] # Empty List to collect the Nodes
    for node_name, node in graph.nodes.items(): # Check every Node in the graph
        if len(node.edges) % 2 != 0:   # Checks if a Node's degree is odd
            odd_nodes.append(node_name)  # If Node's degree is odd then add to the list
    return odd_nodes # Return full List

# Function (Finds cheapest odd nodes to connect)
def compute_odd_pair_distances(graph, odd_nodes):
    pair_distances = {} # Empty dictionary to store shortest distance between odd Nodes

    for i in range(len(odd_nodes)):  # Go through every unique pair of odd nodes
        for j in range(i + 1, len(odd_nodes)):
            u = odd_nodes[i]
            v = odd_nodes[j]
            dist, _ = dijkstra(graph, u, v)  # Calling Dijkstra funcction to find shortest path between them
            pair_distances[(u, v)] = dist # Save result into our pair_distances dictionary

    return pair_distances  # Return full List 


# Function to pick the most efficient way (shortest path) to pair up the odd degree nodes we have saved in the pair_distances dictionary
def greedy_minimum_matching(pair_distances):
    
    sorted_pairs = sorted(pair_distances.items(), key=lambda x: x[1]) # Sort pairs by their distance (cheapest first)

    matched = set() # Nodes that have already been paired
    matching = []  # Final List of pairs

    for (u, v), dist in sorted_pairs:  # Loop: for every u and v distance we have in sorted_pairs
        if u not in matched and v not in matched: # If not already used
            matching.append((u, v)) # Mark them as used, adding them to matched = set()
            matched.add(u)
            matched.add(v)

    return matching  # Return the Final List of pairs

# Function (modifies graph to make it Eurelian)
def duplicate_edges_for_eulerian(graph, matching):

    for u, v in matching:  #Loop: For every u,v in matching  list
        
        _, path = dijkstra(graph, u, v)  # Find the shortest path between u and v

        # For each consecutive pair of nodes in the path, add a duplicate edge
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]

            # Find the original edge between node_a and node_b
            for edge in graph.edges:
                if ((edge.from_node.name == node_a and edge.to_node.name == node_b) or
                    (edge.from_node.name == node_b and edge.to_node.name == node_a)):
                    
                    # Duplicate the edge (add another Edge object)
                    new_edge = Edge(edge.from_node, edge.to_node, edge.weight)
                    graph.edges.append(new_edge)
                    edge.from_node.edges.append(new_edge)
                    edge.to_node.edges.append(new_edge)
                    break  # only duplicate 1 edge per pair


# 
def find_eulerian_circuit(graph, start_node):
    circuit = [] # Final result
    stack = [start_node]  # nodes we’re exploring

    while stack: # Loop: While we have still something to explore
        current = stack[-1]  # The current node
        unused_edges = [e for e in graph.nodes[current].edges if not e.used]  # paths from  this node that haven't been taken yet

        # If there are any unused edges left
        if unused_edges:  

            edge = unused_edges[0] # Check first unused edge
            edge.used = True  # mark it as used

            # Move to the next node
            next_node = edge.to_node.name if edge.from_node.name == current else edge.from_node.name
            stack.append(next_node) # Push the edge on the task
        else:
            
            circuit.append(stack.pop()) # Dead end, no edges left -> add to the circuit[]

    
    return circuit[::-1] # Reverse the circuit order to get the real path order


# Openning the JSON file
with open("data/road.json") as f:
    data = json.load(f)


# Create an empty graph
g = Graph()

# Add nodes
for node_data in data["area"]["nodes"]:
    g.add_node(node_data["node"])
    g.nodes[node_data["node"]].x = node_data["x-coord"]
    g.nodes[node_data["node"]].y = node_data["y-coord"]

# Add edges
for node_data in data["area"]["nodes"]:
    u = node_data["node"]
    for link in node_data["links"]:
        for v, weight in link.items():
            g.add_edge(u, v, int(weight))  # convert weight from string to int


def reset_edge_usage(graph):
    "Reset all edges to unused for a fresh Eulerian circuit run."
    for edge in graph.edges:
        edge.used = False


# Visual: Map for option 1 (shortest path)
def show_map_with_path(graph, path):
    plt.figure(figsize=(12, 8))

    # Draw all edges normally (gray)
    for edge in graph.edges:
        x_values = [edge.from_node.x, edge.to_node.x]
        y_values = [edge.from_node.y, edge.to_node.y]
        plt.plot(x_values, y_values, 'gray', linewidth=1, zorder=1)

    # Draw the shortest path in RED with arrows
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        x1 = graph.nodes[u].x
        y1 = graph.nodes[u].y
        x2 = graph.nodes[v].x
        y2 = graph.nodes[v].y

        # Red line
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=3, zorder=3)

        # Arrow
        plt.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            zorder=4
        )

    # Draw nodes
    for node_name, node in graph.nodes.items():
        plt.scatter(node.x, node.y, color='blue', s=100, zorder=5)
        plt.text(node.x + 0.1, node.y + 0.1, node_name,
                 fontsize=10, zorder=6)

    plt.title("WinterRegion Map - Shortest Path")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def show_map_with_eulerian(graph, circuit):
    plt.figure(figsize=(12, 8))

    # Draw all edges normally (gray) 
    for edge in graph.edges:
        x_values = [edge.from_node.x, edge.to_node.x]
        y_values = [edge.from_node.y, edge.to_node.y]
        plt.plot(x_values, y_values, 'gray', linewidth=1, zorder=1)

    # Draw Eulerian circuit in ORANGE with arrows 
    for i in range(len(circuit) - 1):
        u = circuit[i]
        v = circuit[i + 1]

        x1 = graph.nodes[u].x
        y1 = graph.nodes[u].y
        x2 = graph.nodes[v].x
        y2 = graph.nodes[v].y

        # Orange highlighted line
        plt.plot([x1, x2], [y1, y2], color='orange', linewidth=2.5, zorder=3)

        # Arrow
        plt.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="orange", lw=2),
            zorder=4
        )

    # -Draw nodes 
    for node_name, node in graph.nodes.items():
        plt.scatter(node.x, node.y, color='blue', s=100, zorder=5)
        plt.text(node.x + 0.1, node.y + 0.1, node_name,
                 fontsize=10, zorder=6)

    plt.title("Eulerian Circuit Visualization")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


#Visual: Option 3 (show the map)
def show_map(graph):
    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor("#f5f5f5")

    # Draw edges + grit
    for edge in graph.edges:
        x_values = [edge.from_node.x, edge.to_node.x]
        y_values = [edge.from_node.y, edge.to_node.y]
        plt.plot(x_values, y_values, 'gray', zorder=1)

        # Midpoint of edge
        mid_x = (edge.from_node.x + edge.to_node.x) / 2
        mid_y = (edge.from_node.y + edge.to_node.y) / 2

        # Grit label
        plt.text(mid_x - 0.1, mid_y + 0.05, str(edge.weight), fontsize=10, color='darkred', zorder=4)

    # Draw nodes
    for node_name, node in graph.nodes.items():
        plt.scatter(node.x, node.y, color='blue', s=50, edgecolor="black", linewidth=1.5,  zorder=2)
        plt.text(node.x + 0.1, node.y + 0.1, node_name, fontsize=10, zorder=3)

    plt.title("WinterRegion Map")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Main Menu for easier navigation 
def main():
    while True:

        print("\n-------- |MENU| --------")
        print("1. Find shortest path")
        print("2. Show full Eulerian circuit")
        print("3. Show the Map")
        print("4. Notes and Instructions")
        print("5. Exit")

        choice = input("Select option 1 - 5: ")

        # Option 1: Shortest Path
        if choice == "1":
            print("Nice, you selected option:", choice)
            start_node = input("Enter START node: ").upper()
            end_node = input("Enter END node: ").upper()

            dist, path = dijkstra(g, start_node, end_node)
            print("\nShortest path:", " -> ".join(path))
            print("Total grit:", dist)

            show_map_with_path(g, path)

        
        # Option 2: Eulerian Circuit
        elif choice == "2":
            print("Nice, you selected option:", choice)
            start_node = input("Enter START node for Eulerian circuit: ").upper()

            odd_nodes = get_odd_degree_nodes(g)
            pair_distances = compute_odd_pair_distances(g, odd_nodes)
            matching = greedy_minimum_matching(pair_distances)

            duplicate_edges_for_eulerian(g, matching)

            circuit = find_eulerian_circuit(g, start_node)
            total_grit = sum(edge.weight for edge in g.edges)
            odd_after = get_odd_degree_nodes(g)

            print("\nEulerian Circuit (full route):")
            print(" -> ".join(circuit))
            print("Total grit including duplicates:", total_grit)

            # Show visual map of the Eulerian circuit
            show_map_with_eulerian(g, circuit)


            # Submenu
            while True:
                print("\nExtra info available for this circuit:")
                print("1. Show odd-degree nodes BEFORE duplication")
                print("2. Show matching pairs + odd-degree nodes AFTER duplication")
                print("3. Go back to main menu")

                sub_choice = input("Select option 1 - 3: ")

                if sub_choice == "1":
                    print("\nOdd-degree nodes BEFORE duplication:")
                    print(", ".join(odd_nodes))

                elif sub_choice == "2":
                    print("\nPairs selected to duplicate:")
                    for u, v in matching:
                        print(f"{u} <-> {v}")

                    print("\nOdd-degree nodes AFTER duplication:")
                    if len(odd_after) == 0:
                        print("None — graph is now Eulerian!")
                    else:
                        print(", ".join(odd_after))

                elif sub_choice == "3":
                    print("Returning to main menu...")
                    break

                else:
                    print("Invalid choice. Try again.")
                    
        elif choice == "3":
           print("Displaying the map...")
           show_map(g)

        # Option 5: Exit
        elif choice == "5":
            print("Thank you for using this program. Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


main()
        
           







            