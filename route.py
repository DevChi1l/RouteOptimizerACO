import pandas as pd 
import numpy as np 
import folium 
import osmnx as ox 
import seaborn as sns
import networkx as nx 
import matplotlib.pyplot as plt 
import random
import webbrowser
from folium.features import DivIcon
import time
from scipy.spatial.distance import pdist, squareform
import webbrowser
import pandas as pd
from aco import ACO


class RouteOptimizer:
    """
    Route optimization class that handles data preparation, network creation,
    and visualization for the ACO algorithm
    """
    def __init__(self, data_df):
        self.data_df = data_df
        self.dtf = None
        self.G = None
        self.start = None
        self.distance_matrix = None
    
    

    def prepare_data(self):
        """Prepare and filter the data for routing"""
        # Prepare filtered data
        self.dtf = self.data_df[["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
        self.dtf = self.dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"})
        
        # Assign colors
        self.dtf["color"] = "green"  # General points in green
        self.dtf.loc[self.dtf['id'] == 0, 'color'] = 'red'  # Starting point (Bus Station)
        self.dtf.loc[self.dtf['id'] == 3, 'color'] = 'blue'  # VIT-AP University
        
        # Assign border colors
        self.dtf["border_color"] = "white"
        self.dtf.loc[self.dtf['id'] == 0, "border_color"] = "darkred"
        self.dtf.loc[self.dtf['id'] == 3, "border_color"] = "darkblue"
        
        # Set starting point
        self.start = self.dtf[self.dtf["id"] == 0][["y", "x"]].values[0]
        print("Starting point:", self.start)
        
        return self.dtf
    
    def create_initial_map(self):
        """Create an initial map with all locations marked"""
        map_ = folium.Map(location=self.start, tiles="cartodbpositron", zoom_start=12)
        self.dtf.apply(lambda row: folium.CircleMarker(
            location=[row["y"], row["x"]], 
            color=row["border_color"], fill=True, fill_color=row["color"], 
            fill_opacity=0.8, radius=8,
            tooltip=row["Street Address"]  # Added tooltip for better UX
        ).add_to(map_), axis=1)
        
        # Save initial map
        map_.save("initial_map.html")
        print("Initial map saved as initial_map.html")
        return map_
    
    def generate_road_network(self, radius_km=30):
        """Generate the road network using OSMnx"""
        print(f"Generating road network with {radius_km}km radius...")
        self.G = ox.graph_from_point(self.start, dist=radius_km*1000, network_type="drive")
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
        # Plot the graph and save it
        fig, ax = ox.plot_graph(self.G, bgcolor="black", node_size=8, node_color="white", 
                              figsize=(16, 8), show=False, save=False)
        fig.savefig(f"road_network_{radius_km}km.png", dpi=300, bbox_inches="tight")
        print(f"Graph saved as road_network_{radius_km}km.png")
        return self.G
    
    def find_nearest_nodes(self):
        """Find nearest network nodes to each location point"""
        # Get nearest node to starting point
        start_node = ox.distance.nearest_nodes(self.G, self.start[1], self.start[0])
        
        # Assign nearest nodes to each location (corrected x and y)
        self.dtf["node"] = self.dtf.apply(lambda x: ox.distance.nearest_nodes(self.G, x["x"], x["y"]), axis=1)
        
        # Remove duplicates
        self.dtf = self.dtf.drop_duplicates(subset=["y", "x"], keep="first")
        print("\nData with assigned network nodes:")
        print(self.dtf)
        return start_node
    
    def compute_distance_matrix(self):
        """Compute the travel time matrix between all nodes"""
        nodes = self.dtf["node"].values
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        print("Computing distance matrix for pheromone initialization...")
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        matrix[i, j] = nx.shortest_path_length(self.G, source=nodes[i], 
                                                              target=nodes[j], 
                                                              method='dijkstra', 
                                                              weight='travel_time')
                    except nx.NetworkXNoPath:
                        print(f"No path found between nodes {nodes[i]} and {nodes[j]}")
                        matrix[i, j] = np.inf
        
        self.distance_matrix = pd.DataFrame(matrix, index=nodes, columns=nodes)
        
        # Visualize the matrix as a heatmap (represents initial pheromone distribution)
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_matrix, annot=True, cmap="YlGnBu", fmt=".0f")
        plt.title("Travel Time Matrix (seconds) - Initial Distance Values")
        plt.savefig("distance_matrix_heatmap.png", dpi=300, bbox_inches="tight")
        print("Heatmap saved as distance_matrix_heatmap.png")
        
        return self.distance_matrix
    
    def optimize_route(self, n_ants=10, n_iterations=50, alpha=1.0, beta=2.5, 
                      rho=0.5, q=100, strategy='elitist'):
        """Run ACO to optimize the route"""
        # Create ACO instance with genuine ACO parameters
        aco = ACO(
            distance_matrix=self.distance_matrix,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            strategy=strategy
        )
        
        # Run ACO
        start_node = 0  # First node in the matrix (equivalent to start location)
        
        # Solve using ACO
        best_path, best_distance = aco.solve(start_node)
        
        if best_path:
            # Nodes in the network
            lst_nodes = self.dtf["node"].tolist()
            
            # Convert path indices to actual nodes
            route_nodes = [lst_nodes[i] for i in best_path[:-1]]  # Exclude the last one as it's the return to start
            
            # Convert to route locations
            route_locations = []
            for node in route_nodes:
                location = self.dtf.loc[self.dtf['node'] == node, 'Street Address'].values
                if len(location) > 0:
                    route_locations.append(location[0])
            
            # Print results
            print("\nACO Optimized Route:")
            for i, location in enumerate(route_locations):
                print(f"Stop {i}: {location}")
            
            print(f"\nTotal travel time: {best_distance} seconds ({best_distance/60:.1f} minutes)")
            print(f"Total stops: {len(route_locations)}")
            
            # Create route indices (for visualization compatibility)
            route_indices = best_path[:-1]
            
            return route_nodes, route_indices, route_locations, best_distance
        
        return None, None, None, None
    
    def create_detailed_route_map(self, route_nodes, route_indices, route_locations):
        """Create a detailed route map with the optimized path"""
            
        # Create a new map for the optimized route
        route_map = folium.Map(location=self.start, tiles="cartodbpositron", zoom_start=12)
        
        # Add traffic visualization if needed
        if hasattr(self, 'add_traffic_visualization') and self.add_traffic_visualization:
            from traffic_simulator import TrafficSimulator
            traffic_sim = TrafficSimulator(self.G)
            traffic_sim.visualize_traffic(self.traffic_G, route_map)
        
        # Continue with the original method...
        # Add markers for all locations
        self.dtf.apply(lambda row: folium.CircleMarker(
            location=[row["y"], row["x"]], 
            color=row["border_color"], fill=True, fill_color=row["color"], 
            fill_opacity=0.8, radius=8,
            tooltip=row["Street Address"]
        ).add_to(route_map), axis=1)
        
        # Rest of the original method follows...
        # Add numbered markers for route order
        lst_nodes = self.dtf["node"].tolist()
        for i, idx in enumerate(route_indices):
            node_id = route_nodes[i]
            node_data = self.dtf[self.dtf['node'] == node_id]
            if not node_data.empty:
                folium.Marker(
                    location=[node_data.iloc[0]['y'], node_data.iloc[0]['x']],
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: 12pt; color: white; background-color: black; border-radius: 50%; width: 20px; height: 20px; text-align: center;">{i}</div>'
                    ),
                    tooltip=f"Stop {i}: {node_data.iloc[0]['Street Address']}"
                ).add_to(route_map)
        
        # Plot each path segment on the map with different colors (pheromone trails)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
        
        # Extract all coordinates for the entire route
        all_route_coords = []
        segment_transitions = []
        
        # Generate detailed paths (colony trails)
        def get_path_between_nodes(route_nodes):
            path_segments = []
            for i in range(len(route_nodes) - 1):
                a, b = route_nodes[i], route_nodes[i + 1]
                try:
                    # Use traffic-adjusted graph if available
                    if hasattr(self, 'add_traffic_visualization') and self.add_traffic_visualization:
                        path = nx.shortest_path(self.traffic_G, source=a, target=b, weight='weight')
                    else:
                        path = nx.shortest_path(self.G, source=a, target=b, weight='travel_time')
                    path_segments.append(path)
                except nx.NetworkXNoPath:
                    print(f"No path found between {a} and {b}")
            return path_segments
        
        route_paths = get_path_between_nodes(route_nodes)
        
        for path in route_paths:
            path_coords = []
            for node in path:
                y = self.G.nodes[node]['y']
                x = self.G.nodes[node]['x']
                path_coords.append((y, x))
            
            all_route_coords.extend(path_coords)
            segment_transitions.append(len(all_route_coords) - 1)  # Mark end of each segment
        
        # Add all route segments to the map
        for i, path in enumerate(route_paths):
            # Get color for this segment
            color = colors[i % len(colors)]
            
            # Extract the route coordinates
            route_coords = []
            for node in path:
                y = self.G.nodes[node]['y']
                x = self.G.nodes[node]['x']
                route_coords.append((y, x))
                
            # Add the route line to the map
            folium.PolyLine(
                route_coords,
                color=color,
                weight=4,
                opacity=0.8
            ).add_to(route_map)
            
            # Add distance information
            # Calculate path metrics
            path_length = 0
            path_time = 0
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                # Get all edges between the nodes
                try:
                    # Handle multigraphs (get the first edge if multiple exist)
                    edge_data = None
                    for key in self.G[u][v]:
                        edge_data = self.G[u][v][key]
                        break
                        
                    if edge_data:
                        path_length += edge_data.get('length', 0)
                        # Use traffic-adjusted time if available
                        if hasattr(self, 'add_traffic_visualization') and self.add_traffic_visualization:
                            for key in self.traffic_G[u][v]:
                                traffic_data = self.traffic_G[u][v][key]
                                path_time += traffic_data.get('traffic_time', 0)
                                break
                        else:
                            path_time += edge_data.get('travel_time', 0)
                except:
                    pass
            
            # Get midpoint for the popup
            mid_idx = len(path) // 2
            mid_node = path[mid_idx]
            mid_y, mid_x = self.G.nodes[mid_node]['y'], self.G.nodes[mid_node]['x']
            
            # Add popup with segment info
            folium.Popup(
                f"Segment {i+1}: {route_locations[i]} → {route_locations[i+1] if i+1 < len(route_locations) else route_locations[0]}<br>"
                f"Distance: {path_length/1000:.2f} km<br>"
                f"Travel time: {path_time/60:.1f} min",
                max_width=300
            ).add_to(folium.Marker(
                location=[mid_y, mid_x],
                icon=folium.DivIcon(html="")
            ).add_to(route_map))
        
        # Save the route map
        file_name = "traffic_adjusted_route.html" if hasattr(self, 'add_traffic_visualization') and self.add_traffic_visualization else "aco_optimized_route_map.html"
        route_map.save(file_name)
        print(f"Route map saved as {file_name}")
        
        return route_map, all_route_coords, segment_transitions, route_paths
    
    

    def create_animated_route(self, all_route_coords, segment_transitions, route_locations,route_paths):
        """Create an interactive animated visualization of the route"""
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink'] 
        # Convert Python lists to JavaScript arrays for animation
        route_coords_js = "[" + ",".join(f"[{coord[0]}, {coord[1]}]" for coord in all_route_coords) + "]"
        segment_transitions_js = str(segment_transitions).replace(" ", "")
        route_locations_js = "[" + ",".join(f"'{loc}'" for loc in route_locations) + "]"


        # Create animation control HTML directly
        animation_control_html = """
        <div id="busAnimationControl" style="position: fixed; top: 10px; left: 60px; z-index: 1000; background-color: white; 
            padding: 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.6); border: 2px solid #333;">
            <h4 style="margin: 0 0 8px 0; text-align: center; font-weight: bold;">ACO Bus Route Simulation</h4>
            <button id="startAnimation" style="padding: 8px 15px; margin-right: 8px; background-color: #4CAF50; color: white; 
                    border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px;">▶ Start</button>
            <button id="pauseAnimation" style="padding: 8px 15px; margin-right: 8px; background-color: #FFC107; color: black; 
                    border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px;" disabled>⏸ Pause</button>
            <button id="resetAnimation" style="padding: 8px 15px; background-color: #F44336; color: white; 
                    border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px;">🔄 Reset</button>
            <div id="busStatus" style="margin-top: 8px; font-size: 13px; font-weight: bold; text-align: center;">At: Start</div>
        </div>
        """

        # Create standalone HTML file
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ACO Bus Route Animation</title>
            
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            
            <style>
                body {{ margin: 0; padding: 0; }}
                #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
                .bus-icon {{ font-size: 28px; filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.7)); }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            {animation_control_html}

            <script>
                var map = L.map('map').setView([{self.start[0]}, {self.start[1]}], 12);

                L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                    subdomains: 'abcd',
                    maxZoom: 19
                }}).addTo(map);
        """

        for i, row in self.dtf.iterrows():
            html_content += f"""
                L.circleMarker([{row['y']}, {row['x']}], {{
                    color: '{row['border_color']}',
                    fillColor: '{row['color']}',
                    fillOpacity: 0.8,
                    radius: 8
                }}).bindTooltip("{row['Street Address']}").addTo(map);
            """

        for i, path in enumerate(route_paths):
            color = colors[i % len(colors)]
            path_coords_js = "[" + ",".join(f"[{self.G.nodes[node]['y']}, {self.G.nodes[node]['x']}]" for node in path) + "]"

            html_content += f"""
                L.polyline({path_coords_js}, {{
                    color: '{color}',
                    weight: 4,
                    opacity: 0.8
                }}).addTo(map);
            """

        html_content += f"""
                var routeCoords = {route_coords_js};
                var segmentTransitions = {segment_transitions_js};
                var routeLocations = {route_locations_js};
                var busIcon = '🚌';
                var animationSpeed = 50;
                var busMarker = null;
                var animationRunning = false;
                var currentPosition = 0;
                var animationInterval;

                function createBusMarker() {{
                    if (busMarker) {{
                        map.removeLayer(busMarker);
                    }}
                    busMarker = L.marker([routeCoords[0][0], routeCoords[0][1]], {{
                        icon: L.divIcon({{
                            html: '<div class="bus-icon">' + busIcon + '</div>',
                            iconSize: [35, 35],
                            iconAnchor: [17, 17]
                        }})
                    }});
                    busMarker.addTo(map);
                    return busMarker;
                }}

                function startAnimation() {{
                    if (animationRunning) return;
                    animationRunning = true;
                    document.getElementById('startAnimation').disabled = true;
                    document.getElementById('pauseAnimation').disabled = false;
                    if (!busMarker) busMarker = createBusMarker();
                    
                    animationInterval = setInterval(function() {{
                        if (currentPosition >= routeCoords.length - 1) {{
                            clearInterval(animationInterval);
                            document.getElementById('startAnimation').disabled = false;
                            document.getElementById('pauseAnimation').disabled = true;
                            document.getElementById('busStatus').innerHTML = 'At: Destination';
                            animationRunning = false;
                            return;
                        }}
                        currentPosition++;
                        busMarker.setLatLng([routeCoords[currentPosition][0], routeCoords[currentPosition][1]]);
                        map.panTo([routeCoords[currentPosition][0], routeCoords[currentPosition][1]]);
                    }}, animationSpeed);
                }}

                function pauseAnimation() {{
                    clearInterval(animationInterval);
                    document.getElementById('startAnimation').disabled = false;
                    document.getElementById('pauseAnimation').disabled = true;
                    animationRunning = false;
                }}

                function resetAnimation() {{
                    clearInterval(animationInterval);
                    currentPosition = 0;
                    animationRunning = false;
                    document.getElementById('startAnimation').disabled = false;
                    document.getElementById('pauseAnimation').disabled = true;
                    document.getElementById('busStatus').innerHTML = 'At: Start';
                    if (busMarker) {{
                        busMarker.setLatLng([routeCoords[0][0], routeCoords[0][1]]);
                        map.panTo([routeCoords[0][0], routeCoords[0][1]]);
                    }}
                }}

                document.getElementById('startAnimation').addEventListener('click', startAnimation);
                document.getElementById('pauseAnimation').addEventListener('click', pauseAnimation);
                document.getElementById('resetAnimation').addEventListener('click', resetAnimation);
                createBusMarker();
            </script>
        </body>
        </html>
        """

        with open("aco_animated_route.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print("Animated route visualization saved as aco_animated_route.html")
        webbrowser.open("aco_animated_route.html")

    def incorporate_traffic_data(self):
        """Generate traffic conditions and update the road network"""
        # Import the traffic simulator
        from traffic_simulator import TrafficSimulator
        
        # Create traffic simulator
        traffic_sim = TrafficSimulator(self.G)
        
        # Generate traffic-adjusted network
        self.traffic_G = traffic_sim.generate_traffic_conditions()
        
        # Create traffic visualization map
        self.traffic_map = traffic_sim.visualize_traffic(self.traffic_G, folium.Map(
            location=[self.dtf['y'].mean(), self.dtf['x'].mean()],
            zoom_start=12
        ))
        
        # Add points to the traffic map
        for idx, row in self.dtf.iterrows():
            popup_text = f"{row['Street Address']}<br>ID: {row['id']}"
            folium.Marker(
                [row['y'], row['x']],
                popup=popup_text,
                icon=folium.Icon(color=row['color'], icon='info-sign')
            ).add_to(self.traffic_map)
        
        # Save traffic map
        self.traffic_map.save('traffic_conditions.html')
        print("Traffic conditions map saved as traffic_conditions.html")
        
        # Update the distance matrix based on traffic conditions
        self.compute_traffic_distance_matrix()
        
        return self.traffic_map

    def compute_traffic_distance_matrix(self):
        """Compute distance matrix using traffic-adjusted road network"""
        import networkx as nx
        
        # Store original distance matrix for comparison
        if not hasattr(self, 'original_distance_matrix'):
            self.original_distance_matrix = self.distance_matrix.copy()
        
        nodes = self.dtf["node"].values
        n = len(nodes)
        traffic_matrix = np.zeros((n, n))
        self.traffic_route_paths = {}
        
        print("Computing traffic-adjusted distance matrix...")
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        # Use traffic-adjusted weights for routing
                        route = nx.shortest_path(
                            self.traffic_G, 
                            nodes[i], 
                            nodes[j], 
                            weight='weight'  # 'weight' now includes traffic
                        )
                        
                        # Calculate total travel time
                        travel_time = 0
                        edges = list(zip(route[:-1], route[1:]))
                        
                        for u, v in edges:
                            # Handle multigraphs (get the first edge if multiple exist)
                            edge_data = None
                            for key in self.traffic_G[u][v]:
                                edge_data = self.traffic_G[u][v][key]
                                break
                            
                            if edge_data:
                                travel_time += edge_data.get('traffic_time', 0)
                        
                        # Store travel time for ACO algorithm
                        traffic_matrix[i, j] = travel_time
                        
                        # Store route for visualization
                        self.traffic_route_paths[(i, j)] = route
                    except nx.NetworkXNoPath:
                        print(f"No path found between nodes {nodes[i]} and {nodes[j]}")
                        traffic_matrix[i, j] = float('inf')
        
        # Update the distance matrix with traffic data
        self.traffic_distance_matrix = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
        
        # Visualize the traffic matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.traffic_distance_matrix, annot=True, cmap="YlOrRd", fmt=".0f")
        plt.title("Traffic-Adjusted Travel Time Matrix (seconds)")
        plt.savefig("traffic_matrix_heatmap.png", dpi=300, bbox_inches="tight")
        print("Traffic matrix heatmap saved as traffic_matrix_heatmap.png")
        
        return self.traffic_distance_matrix

        # Add these methods to your RouteOptimizer class
    def incorporate_traffic_data(self):
        """Generate traffic conditions and update the road network"""
        # Import the traffic simulator
        from traffic_simulator import TrafficSimulator
        
        # Create traffic simulator
        traffic_sim = TrafficSimulator(self.G)
        
        # Generate traffic-adjusted network
        self.traffic_G = traffic_sim.generate_traffic_conditions()
        
        # Create traffic visualization map
        self.traffic_map = traffic_sim.visualize_traffic(self.traffic_G, folium.Map(
            location=[self.dtf['y'].mean(), self.dtf['x'].mean()],
            zoom_start=12
        ))
        
        # Add points to the traffic map
        for idx, row in self.dtf.iterrows():
            popup_text = f"{row['Street Address']}<br>ID: {row['id']}"
            folium.Marker(
                [row['y'], row['x']],
                popup=popup_text,
                icon=folium.Icon(color=row['color'], icon='info-sign')
            ).add_to(self.traffic_map)
        
        # Save traffic map
        self.traffic_map.save('traffic_conditions.html')
        print("Traffic conditions map saved as traffic_conditions.html")
        
        # Update the distance matrix based on traffic conditions
        self.compute_traffic_distance_matrix()
        
        return self.traffic_map

    def compute_traffic_distance_matrix(self):
        """Compute distance matrix using traffic-adjusted road network"""
        import networkx as nx
        
        # Store original distance matrix for comparison
        if not hasattr(self, 'original_distance_matrix'):
            self.original_distance_matrix = self.distance_matrix.copy()
        
        nodes = self.dtf["node"].values
        n = len(nodes)
        traffic_matrix = np.zeros((n, n))
        self.traffic_route_paths = {}
        
        print("Computing traffic-adjusted distance matrix...")
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        # Use traffic-adjusted weights for routing
                        route = nx.shortest_path(
                            self.traffic_G, 
                            nodes[i], 
                            nodes[j], 
                            weight='weight'  # 'weight' now includes traffic
                        )
                        
                        # Calculate total travel time
                        travel_time = 0
                        edges = list(zip(route[:-1], route[1:]))
                        
                        for u, v in edges:
                            # Handle multigraphs (get the first edge if multiple exist)
                            edge_data = None
                            for key in self.traffic_G[u][v]:
                                edge_data = self.traffic_G[u][v][key]
                                break
                            
                            if edge_data:
                                travel_time += edge_data.get('traffic_time', 0)
                        
                        # Store travel time for ACO algorithm
                        traffic_matrix[i, j] = travel_time
                        
                        # Store route for visualization
                        self.traffic_route_paths[(i, j)] = route
                    except nx.NetworkXNoPath:
                        print(f"No path found between nodes {nodes[i]} and {nodes[j]}")
                        traffic_matrix[i, j] = float('inf')
        
        # Update the distance matrix with traffic data
        self.traffic_distance_matrix = pd.DataFrame(traffic_matrix, index=nodes, columns=nodes)
        
        # Visualize the traffic matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.traffic_distance_matrix, annot=True, cmap="YlOrRd", fmt=".0f")
        plt.title("Traffic-Adjusted Travel Time Matrix (seconds)")
        plt.savefig("traffic_matrix_heatmap.png", dpi=300, bbox_inches="tight")
        print("Traffic matrix heatmap saved as traffic_matrix_heatmap.png")
        
        return self.traffic_distance_matrix

    def optimize_with_traffic(self, n_ants=10, n_iterations=50, alpha=1.0, beta=2.5, 
                        rho=0.5, q=100, strategy='elitist'):
        """Run optimization with traffic data incorporated"""
        # Generate traffic conditions and update distance matrix
        self.incorporate_traffic_data()
        
        # Store original matrix
        original_matrix = self.distance_matrix.copy()
        
        # Set distance matrix to traffic-adjusted matrix
        self.distance_matrix = self.traffic_distance_matrix
        
        # Run ACO optimization with traffic-adjusted distances
        print("\nRunning ACO optimization with traffic data...")
        route_nodes, route_indices, route_locations, best_distance = self.optimize_route(
            n_ants, n_iterations, alpha, beta, rho, q, strategy
        )
        
        if route_nodes:
            # Create detailed route map with traffic visualization
            self.add_traffic_visualization = True
            route_map, all_route_coords, segment_transitions, route_paths = self.create_detailed_route_map(
                route_nodes, route_indices, route_locations
            )
            
            # Save traffic-adjusted route
            route_map.save('traffic_adjusted_route.html')
            print("Traffic-adjusted route saved as traffic_adjusted_route.html")
            
            # Create comparison map
            self.create_comparison_map(route_indices)
            
            # Reset flag
            self.add_traffic_visualization = False
            
            # Restore original distance matrix
            self.distance_matrix = original_matrix
            
            return route_map, best_distance
        
        # Restore original distance matrix
        self.distance_matrix = original_matrix
        return None, None

    def create_comparison_map(self, traffic_route_indices):
        """Create a map comparing routes with and without traffic"""
        from traffic_simulator import TrafficSimulator
        
        # Calculate optimized route without traffic (using original distance matrix)
        print("\nCalculating standard route for comparison...")
        original_route_nodes, original_route_indices, original_route_locations, original_distance = self.optimize_route(
            n_ants=10, n_iterations=50
        )
        
        # Create a new map for comparison
        center_lat = self.dtf['y'].mean()
        center_lon = self.dtf['x'].mean()
        comp_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add traffic visualization to the map
        traffic_sim = TrafficSimulator(self.G)
        traffic_sim.visualize_traffic(self.traffic_G, comp_map)
        
        # Add routes to map
        if original_route_nodes:
            # Create route without traffic visualization (blue)
            route_paths = []
            for i in range(len(original_route_indices) - 1):
                start_idx = original_route_indices[i]
                end_idx = original_route_indices[i + 1]
                
                # Get node IDs
                start_node = self.dtf.iloc[start_idx]['node']
                end_node = self.dtf.iloc[end_idx]['node']
                
                try:
                    # Calculate path without traffic
                    path = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
                    route_paths.append(path)
                    
                    # Extract coordinates
                    coords = []
                    for node in path:
                        y = self.G.nodes[node]['y']
                        x = self.G.nodes[node]['x']
                        coords.append((y, x))
                    
                    # Add path to map
                    folium.PolyLine(
                        coords,
                        color='blue',
                        weight=4,
                        opacity=0.7,
                        tooltip="Standard route"
                    ).add_to(comp_map)
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node} and {end_node}")
        
        # Add traffic-adjusted route (red)
        if traffic_route_indices:
            for i in range(len(traffic_route_indices) - 1):
                start_idx = traffic_route_indices[i]
                end_idx = traffic_route_indices[i + 1]
                
                # Get node IDs
                start_node = self.dtf.iloc[start_idx]['node']
                end_node = self.dtf.iloc[end_idx]['node']
                
                try:
                    # Calculate path with traffic
                    path = nx.shortest_path(self.traffic_G, start_node, end_node, weight='weight')
                    
                    # Extract coordinates
                    coords = []
                    for node in path:
                        y = self.traffic_G.nodes[node]['y']
                        x = self.traffic_G.nodes[node]['x']
                        coords.append((y, x))
                    
                    # Add path to map
                    folium.PolyLine(
                        coords,
                        color='red',
                        weight=4,
                        opacity=0.7,
                        tooltip="Traffic-adjusted route"
                    ).add_to(comp_map)
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node} and {end_node}")
        
        # Add stop points
        for idx, row in self.dtf.iterrows():
            popup_text = f"{row['Street Address']}<br>ID: {row['id']}"
            folium.Marker(
                [row['y'], row['x']],
                popup=popup_text,
                icon=folium.Icon(color=row['color'], icon='info-sign')
            ).add_to(comp_map)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; 
        background-color: white; padding: 10px; border: 1px solid grey; z-index: 9999">
        <h4>Route Comparison</h4>
        <div><i style="background: blue; width: 15px; height: 15px; display: inline-block;"></i> Standard Route</div>
        <div><i style="background: red; width: 15px; height: 15px; display: inline-block;"></i> Traffic-Adjusted Route</div>
        </div>
        '''
        comp_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Save comparison map
        comp_map.save('route_comparison.html')
        print("Route comparison map saved as route_comparison.html")
        
        # Print comparison statistics
        if original_distance and hasattr(self, 'traffic_distance_matrix'):
            print("\nRoute Comparison:")
            print(f"Standard route travel time: {original_distance:.1f} seconds ({original_distance/60:.1f} minutes)")
            
            # Calculate traffic-adjusted distance
            traffic_distance = 0
            for i in range(len(traffic_route_indices) - 1):
                start_idx = traffic_route_indices[i]
                end_idx = traffic_route_indices[i + 1]
                traffic_distance += self.traffic_distance_matrix.iloc[start_idx, end_idx]
            
            print(f"Traffic-adjusted route travel time: {traffic_distance:.1f} seconds ({traffic_distance/60:.1f} minutes)")
            
            diff = ((traffic_distance - original_distance) / original_distance) * 100
            print(f"Traffic impact: {'Increased' if diff > 0 else 'Decreased'} travel time by {abs(diff):.1f}%")
        
        return comp_map
    
    def calculate_route_statistics(self, standard_route, traffic_route):
        """
        Calculate and display statistics comparing standard and traffic-optimized routes.
        
        Parameters:
        -----------
        standard_route : list
            List of nodes representing the standard optimized route
        traffic_route : list
            List of nodes representing the traffic-optimized route
        """
        print("\n===== Route Statistics =====")
        
        # Calculate route overlap percentage
        common_nodes = set(standard_route).intersection(set(traffic_route))
        overlap_percentage = (len(common_nodes) / len(standard_route)) * 100
        print(f"Route overlap: {overlap_percentage:.1f}%")
        
        # Calculate number of different road segments
        different_segments = len(set(standard_route).symmetric_difference(set(traffic_route)))
        print(f"Different road segments: {different_segments}")
        
        # Calculate additional statistics if needed
        # For example, traffic density comparison, route length comparison, etc.
        
        print("============================")