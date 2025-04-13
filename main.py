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
from route import RouteOptimizer



"""Main function to run the complete optimization process"""
# Sample data for VIT-AP University bus route
data = {
    "id": range(10),
    "City": ["Vijayawada", "Vijayawada", "Vijayawada", "Guntur", "Guntur", 
             "Guntur", "Guntur", "Vijayawada", "Guntur", "Guntur"],
    "Street Address": [
        "Vijayawada Bus Station", 
        "Undavalli Caves (8km SW)",
        "Kondapalli Fort (16km NW)",
        "Amaravathi (35km W)",
        "Tadepalli (12km E)",
        "Mangalagiri Temple (10km SE)",
        "Namburu (18km S)",
        "Gannavaram Airport (20km NE)",
        "Tenali (30km S)", 
        "VIT-AP University (25km W)"
    ],
    "Latitude": [
        16.5069,  # Vijayawada Central
        16.4850,  # Southwest
        16.6167,  # Northwest
        16.5727,  # West (Amaravati)
        16.4667,  # East
        16.4300,  # Southeast
        16.3667,  # South
        16.5333,  # Northeast
        16.2420,  # Far South
        16.4998   # West (University)
    ],
    "Longitude": [
        80.6486,  # Central
        80.6020,  # SW cluster
        80.5333,  # NW cluster
        80.3575,  # Western expansion
        80.6000,  # Eastern suburb
        80.5500,  # SE corridor
        80.5667,  # Southern expansion
        80.8000,  # NE connection
        80.6400,  # Southern terminus
        80.5209   # Academic hub
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create optimizer instance
optimizer = RouteOptimizer(df)

# Prepare data
optimizer.prepare_data()

# Create initial map
optimizer.create_initial_map()

# Generate road network
optimizer.generate_road_network(radius_km=30)

# Find nearest nodes
optimizer.find_nearest_nodes()

# Compute distance matrix
optimizer.compute_distance_matrix()

# Run ACO optimization
route_nodes, route_indices, route_locations, best_distance = optimizer.optimize_route(
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.5,
    rho=0.5,
    q=100,
    strategy='elitist'
)

if route_nodes:
    # Create detailed route map
    route_map, all_route_coords, segment_transitions, route_paths = optimizer.create_detailed_route_map(
        route_nodes, route_indices, route_locations
    )
    
    # Create animated route
    optimizer.create_animated_route(all_route_coords, segment_transitions, route_locations, route_paths)
    
    print("\nOptimization complete! Check the generated HTML files to view the results.")
