import heapq

def dijkstra(adjacency_list, start_node, target_nodes=None):
    """
    Implementation of Dijkstra's algorithm to find shortest paths from a starting node to all other nodes in a graph.
    
    Arguments:
    adjacency_list -- dictionary containing adjacency list representation of the graph 
                      (keys are nodes and values are lists of tuples representing edges: (neighbour_node, edge_weight))
    start_node -- starting node to compute shortest paths from
    
    Returns:
    distances -- dictionary containing the computed shortest distances from the start_node to all other nodes in the graph
    prev_nodes -- dictionary containing the previous node on the shortest path from start_node to each node in the graph
                  (used to reconstruct the actual shortest path)
    """
    # Initialize distances and previous nodes dictionaries
    distances = {node: float('inf') for node in adjacency_list}
    prev_nodes = {node: None for node in adjacency_list}
    # Set distance to start node as 0
    distances[start_node] = 0
    # Create priority queue with initial element (distance to start node, start node)
    heap = [(0, start_node)]
    # Run Dijkstra's algorithm
    while heap:
        # Pop node with lowest distance from heap
        current_distance, current_node = heapq.heappop(heap)
        if target_nodes is not None and current_node in target_nodes:
            path = [current_node]
            cur = current_node
            while cur != start_node:
                cur = prev_nodes[cur]
                path.append(cur)
            path = path[::-1]
            return distances[current_node], path
        # If current node has already been visited, skip it
        if current_distance > distances[current_node]:
            continue
        # For each neighbour of current node
        for neighbour, weight in adjacency_list[current_node]:
            # Calculate tentative distance to neighbour through current node
            tentative_distance = current_distance + weight
            # Update distance and previous node if tentative distance is better than current distance
            if tentative_distance < distances[neighbour]:
                distances[neighbour] = tentative_distance
                prev_nodes[neighbour] = current_node
                # Add neighbour to heap with updated distance
                heapq.heappush(heap, (tentative_distance, neighbour))
    return None, None