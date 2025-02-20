import pandas as pd
import networkx as nx
import folium
from scipy.spatial import cKDTree
import numpy as np
import time
import webbrowser

# Function to safely load the CSV file
def load_csv_safe(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty.")
        df.columns = df.columns.str.strip()  # Ensure column names are clean
        return df
    except (pd.errors.EmptyDataError, ValueError) as e:
        print(f"Error loading CSV file: {e}")
        return None

# Initial data loading
df = load_csv_safe(r"C:\Users\abhir\OneDrive\Desktop\Nautilus\ocean_points.csv")
if df is None:
    raise SystemExit("Unable to load initial CSV file. Exiting.")

# Load ports coordinates
ports_df = load_csv_safe(r"C:\Users\abhir\OneDrive\Desktop\Nautilus\ports_coordinates.csv")
if ports_df is None:
    raise SystemExit("Unable to load ports CSV file. Exiting.")
ports_coords = set((row['Latitude'], row['Longitude']) for _, row in ports_df.iterrows())

# Build a k-d tree for efficient neighbor search
points = df[['lat', 'lon']].values
tree = cKDTree(points)

# Function to create a graph from the dataframe
def create_graph(df, threshold=2.0):
    G = nx.Graph()
    for _, row in df.iterrows():
        coord = (row['lat'], row['lon'])
        # Exclude ports unless they are the source or destination
        if coord in ports_coords and coord not in [source, destination]:
            continue
        G.add_node((row['lat'], row['lon']), 
                   wind=row['Surface Winds (knots)'], 
                   current=row['Currents (knots)'], 
                   wave=row['Wave Height (meters)'])
    for idx, (lat, lon) in enumerate(points):
        dists, idxs = tree.query((lat, lon), k=10, distance_upper_bound=threshold)
        for dist, j in zip(dists, idxs):
            if idx != j and j < len(points):
                row1 = df.iloc[idx]
                row2 = df.iloc[j]
                coord1 = (row1['lat'], row1['lon'])
                coord2 = (row2['lat'], row2['lon'])
                if coord1 in ports_coords and coord1 not in [source, destination]:
                    continue
                if coord2 in ports_coords and coord2 not in [source, destination]:
                    continue
                weight = dist * (1 + row1['Surface Winds (knots)']/20 + row1['Currents (knots)']/2 + row1['Wave Height (meters)']/4)
                G.add_edge((row1['lat'], row1['lon']), (row2['lat'], row2['lon']), weight=weight)
    return G

# Function to find the nearest node in the graph to the given coordinates
def find_nearest_node(G, coord):
    nearest = None
    min_dist = float('inf')
    for node in G.nodes:
        dist = np.linalg.norm(np.array(node) - np.array(coord))
        if dist < min_dist:
            nearest = node
            min_dist = dist
    return nearest

# Read source and destination coordinates from the file
with open('selected_ports.txt', 'r') as f:
    source_coords = f.readline().strip().split(',')
    destination_coords = f.readline().strip().split(',')

# Convert coordinates to tuples
source = (float(source_coords[0]), float(source_coords[1]))
destination = (float(destination_coords[0]), float(destination_coords[1]))


# Function to check if a segment is safe
def is_safe(segment, G):
    wind, current, wave = G.nodes[segment[1]]['wind'], G.nodes[segment[1]]['current'], G.nodes[segment[1]]['wave']
    return wind <= 20 and current <= 2 and wave <= 4

# Heuristic function for A* (Euclidean distance to the destination)
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Initialize initial_path, old_paths, and source_list
initial_path = []
old_paths = []
source_list = [source]  # Initialize with the starting source

# Infinite loop to update the path every 10 seconds
while True:
    # Load the latest data
    df = load_csv_safe(r"C:\Users\abhir\OneDrive\Desktop\Nautilus\updated_coordinates.csv")
    if df is None:
        time.sleep(10)  # Wait before retrying
        continue

    # Update the k-d tree with new points
    points = df[['lat', 'lon']].values
    tree = cKDTree(points)

    # Create the graph
    G = create_graph(df)

    # Set the current source node to the next point in the path, or to the original source if the path is empty
    if initial_path:
        source_node = initial_path[1]
    else:
        source_node = find_nearest_node(G, source)

    # Check if the source node is already the destination
    if source_node == destination:
        print("Destination reached!")

        # Visualize the path on a map using Folium
        m = folium.Map(location=destination, zoom_start=5)

        # Add the path from all old sources to the destination in red
        if len(source_list) > 1:
            folium.PolyLine(source_list, color="red", weight=2, opacity=0.5).add_to(m)

        # Add the path from the last source to the destination in red
        if len(source_list) > 0:
            last_source = source_list[-1]
            final_path_segment = [last_source, destination]
            folium.PolyLine(final_path_segment, color="red", weight=2, opacity=0.5).add_to(m)

        # Add the current path to the map with a ship icon and replace the map pushpin icon with a dot
        for coord in initial_path:
            folium.Marker(location=coord, icon=folium.CustomIcon('icon.png', icon_size=(20, 20)),
                          popup=f'Lat: {coord[0]}, Lon: {coord[1]}, Wind: {G.nodes[coord]["wind"]} knots, '
                                f'Current: {G.nodes[coord]["current"]} knots, Wave: {G.nodes[coord]["wave"]} meters').add_to(m)

        # Add the ship icon at the destination point with a "Destination reached" popup
        folium.Marker(location=destination, icon=folium.CustomIcon('ship_icon.png', icon_size=(30, 30)),
                      popup='Destination reached!').add_to(m)

        # Save the map as an HTML file
        m.save('ship_route.html')
        webbrowser.open('ship_route.html')

        print("Map with the ship route saved as 'ship_route.html'")

        # Exit the loop
        break

    destination_node = find_nearest_node(G, destination)

    print(f"Source Node: {source_node}")
    print(f"Destination Node: {destination_node}")

    # Find the shortest path using A*
    try:
        initial_path = nx.astar_path(G, source=source_node, target=destination_node, heuristic=heuristic, weight='weight')
        print("Initial Path found.")
    except nx.NetworkXNoPath:
        print("No path found between source and destination. Retrying in 10 seconds...")
        time.sleep(10)
        continue

    # Adjust the path dynamically as the ship travels
    current_path = initial_path.copy()
    for i in range(len(current_path) - 1):
        segment = (current_path[i], current_path[i+1])
        if not is_safe(segment, G):
            try:
                new_path = nx.astar_path(G, source=segment[0], target=destination_node, heuristic=heuristic, weight='weight')
                current_path = current_path[:i+1] + new_path[1:]
                print(f"Path adjusted at segment {segment}: Unsafe conditions encountered.")
            except nx.NetworkXNoPath:
                print(f"Adjustment failed at segment {segment}. Retrying in 10 seconds...")
                break

    print("Final Path determined.")

    # Visualize the path on a map using Folium
    m = folium.Map(location=source, zoom_start=5)

    # Add the path from all old sources to the current source in red
    if len(source_list) > 1:
        folium.PolyLine(source_list, color="red", weight=2, opacity=0.5).add_to(m)

    # Add the current path segment from the last source to the new source in red
    if len(source_list) > 0:
        last_source = source_list[-1]
        current_segment_path = [last_source] + current_path
        folium.PolyLine(current_segment_path, color="red", weight=2, opacity=0.5).add_to(m)

    # Add the current path to the map with a ship icon and replace the map pushpin icon with a dot
    for coord in current_path:
        folium.Marker(location=coord, icon=folium.CustomIcon('icon.png', icon_size=(20, 20)),
                      popup=f'Lat: {coord[0]}, Lon: {coord[1]}, Wind: {G.nodes[coord]["wind"]} knots, '
                            f'Current: {G.nodes[coord]["current"]} knots, Wave: {G.nodes[coord]["wave"]} meters').add_to(m)

    # Add the ship icon at the starting point of the path
    folium.Marker(location=current_path[0], icon=folium.CustomIcon('ship_icon.png', icon_size=(30, 30))).add_to(m)

    # Add the destination point with a "Destination" popup
    folium.Marker(location=destination, icon=folium.CustomIcon('destination_icon.png', icon_size=(30, 30)),popup='Destination').add_to(m)
    
    # Add the ship icon at the starting point of the path
    folium.Marker(location=current_path[0], icon=folium.CustomIcon('ship_icon.png', icon_size=(30, 30)),popup='Current Location').add_to(m)

    # Add the current path as a PolyLine
    folium.PolyLine(current_path, color="blue", weight=2.5, opacity=1).add_to(m)

    # Save the map as an HTML file
    m.save('ship_route.html')
    webbrowser.open('ship_route.html')

    print("Map with the ship route saved as 'ship_route.html'")

    # Update the source_list and old paths list
    source_list.append(source_node)
    old_paths.append(initial_path)

    # Wait for 3 seconds before checking again
    time.sleep(1)
