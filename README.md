# RouteOptimizerACO
Route Optimization using Ant Colony Optimization
This project implements Ant Colony Optimization (ACO) algorithms to find optimal routes in road networks, with consideration for traffic conditions and distance optimization.
Project Overview
This application simulates ant colony behavior to solve route optimization problems in road networks. It includes:

Basic ACO implementation for finding shortest paths
Traffic-adjusted route optimization
Visualization tools for route comparison
Animated route visualization
Distance and traffic matrices analysis

Files and Components
Core Algorithm Files

aco.py: Core implementation of the Ant Colony Optimization algorithm
route.py: Route calculation and handling
main.py & main2.py: Main program entry points
traffic_simulator.py: Simulation of traffic conditions

Visualization

initial_map.html: Base map visualization
aco_optimized_route_map.html: Map showing optimized routes
aco_animated_route.html: Animated visualization of route discovery
route_comparison.html: Comparison of different routing strategies
traffic_conditions.html: Visualization of traffic conditions
traffic_adjusted_route.html: Routes adjusted for traffic conditions

Data Visualization

distance_matrix_heatmap.png: Heatmap of distances between nodes
pheromone_heatmap.png: Visualization of pheromone concentrations
traffic_matrix_heatmap.png: Heatmap of traffic intensity
aco_convergence.png: Convergence graph of the ACO algorithm
road_network_30km.png: Road network visualization

Getting Started

Ensure you have Python 3.x installed
Install required dependencies:
pip install numpy matplotlib folium pandas networkx

Run the main program:
python main.py


Features

Ant Colony Optimization: Uses pheromone trails to find efficient routes
Traffic Integration: Factors in real-time or simulated traffic conditions
Interactive Visualizations: HTML-based maps for route analysis
Convergence Analysis: Tools to analyze algorithm performance
Comparative Analysis: Compare ACO routes with traditional routing methods

Contributors
    D.Vasudev
    Pavan