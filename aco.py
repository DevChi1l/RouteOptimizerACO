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


class ACO:
    """
    True Ant Colony Optimization for route finding
    Implements the core ACO algorithm with pheromone trails, ant simulation,
    and probabilistic path selection
    """
    def __init__(self, distance_matrix, n_ants=10, n_iterations=50, 
                 alpha=1.0, beta=2.0, rho=0.5, q=100.0, strategy='elitist'):
        """
        Initialize the ACO algorithm
        
        Parameters:
        -----------
        distance_matrix : pandas DataFrame
            Matrix of distances between nodes
        n_ants : int
            Number of ants to use in each iteration
        n_iterations : int
            Number of iterations to run
        alpha : float
            Relative importance of pheromone
        beta : float
            Relative importance of heuristic information (distance)
        rho : float
            Pheromone evaporation rate (0-1)
        q : float
            Pheromone deposit factor
        strategy : str
            'elitist' - only best ant deposits pheromone
            'as' - all ants deposit pheromone proportional to their path quality
        """
        self.distance_matrix = distance_matrix
        self.n_nodes = len(distance_matrix)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.strategy = strategy
        
        # Convert distance matrix to numpy for faster computation
        self.distances = distance_matrix.values
        
        # Initialize pheromone trails
        # Use inverse of distances as initial pheromone values
        with np.errstate(divide='ignore'):
            heuristic = 1.0 / self.distances
        # Replace infinities with very small value
        heuristic[~np.isfinite(heuristic)] = 1e-10
        
        # Initial pheromone is uniform
        self.pheromones = np.ones((self.n_nodes, self.n_nodes)) / self.n_nodes
        
        # Heuristic information (inverse of distance)
        self.heuristic = heuristic
        
        # Best solution found so far
        self.best_path = None
        self.best_distance = float('inf')
        
        # For tracking history of best distances
        self.best_distances_history = []
        
        print("Initialized ACO with:")
        print(f"- {n_ants} ants")
        print(f"- {n_iterations} iterations")
        print(f"- Alpha: {alpha} (pheromone importance)")
        print(f"- Beta: {beta} (distance importance)")
        print(f"- Rho: {rho} (pheromone evaporation rate)")
        print(f"- Strategy: {strategy}")
    
    def _select_next_node(self, ant_path, current_node, unvisited):
        """
        Select the next node for an ant using the ACO probability formula
        """
        if len(unvisited) == 0:
            return None
        
        # If only one node left, return it
        if len(unvisited) == 1:
            return unvisited[0]
        
        # Calculate probabilities for each unvisited node
        pheromone = np.array([self.pheromones[current_node][j] for j in unvisited])
        heuristic = np.array([self.heuristic[current_node][j] for j in unvisited])
        
        # Calculate the probabilities using the ACO formula
        numerator = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
        # Handle case where all values might be very small
        if np.sum(numerator) == 0:
            # If all values are zero, choose randomly
            probabilities = np.ones(len(unvisited)) / len(unvisited)
        else:
            probabilities = numerator / np.sum(numerator)
        
        # Select next node based on probabilities
        next_node = np.random.choice(unvisited, p=probabilities)
        return next_node
    
    def _construct_solutions(self, start_node):
        """
        Construct solutions for all ants in the colony
        """
        ant_paths = []
        ant_distances = []
        
        # For each ant
        for ant in range(self.n_ants):
            # Start at the specified node
            current_node = start_node
            path = [current_node]
            visited = set([current_node])
            unvisited = [i for i in range(self.n_nodes) if i != current_node]
            path_distance = 0
            
            # Construct path by visiting all nodes
            while unvisited:
                next_node = self._select_next_node(path, current_node, unvisited)
                path.append(next_node)
                visited.add(next_node)
                unvisited.remove(next_node)
                
                # Add distance
                path_distance += self.distances[current_node][next_node]
                current_node = next_node
            
            # Complete the tour by returning to start
            path_distance += self.distances[path[-1]][start_node]
            path.append(start_node)  # Return to starting point
            
            ant_paths.append(path)
            ant_distances.append(path_distance)
        
        return ant_paths, ant_distances
    
    def _update_pheromones(self, ant_paths, ant_distances):
        """
        Update pheromone levels on all edges
        """
        # Evaporation
        self.pheromones *= (1 - self.rho)
        
        # Deposit new pheromones based on strategy
        if self.strategy == 'elitist':
            # Only the best ant deposits pheromone
            best_ant = np.argmin(ant_distances)
            best_path = ant_paths[best_ant]
            best_dist = ant_distances[best_ant]
            
            # Update the best solution if this is better
            if best_dist < self.best_distance:
                self.best_path = best_path.copy()
                self.best_distance = best_dist
                print(f"New best distance: {best_dist}")
            
            # Deposit pheromone on the best path
            for i in range(len(best_path) - 1):
                self.pheromones[best_path[i]][best_path[i+1]] += self.q / best_dist
                # Make sure it's symmetric for undirected graph
                self.pheromones[best_path[i+1]][best_path[i]] += self.q / best_dist
                
        elif self.strategy == 'as':
            # All ants deposit pheromone
            for ant in range(self.n_ants):
                path = ant_paths[ant]
                dist = ant_distances[ant]
                
                # Update best solution if better
                if dist < self.best_distance:
                    self.best_path = path.copy()
                    self.best_distance = dist
                    print(f"New best distance: {dist}")
                
                # Each ant deposits pheromone proportional to its path quality
                deposit = self.q / dist
                for i in range(len(path) - 1):
                    self.pheromones[path[i]][path[i+1]] += deposit
                    # Make sure it's symmetric for undirected graph
                    self.pheromones[path[i+1]][path[i]] += deposit
    
    def solve(self, start_node):
        """
        Run the ACO algorithm to find the optimal path
        
        Parameters:
        -----------
        start_node : int
            Index of the starting node
            
        Returns:
        --------
        best_path : list
            Indices of nodes in the best path found
        best_distance : float
            Total distance of the best path
        """
        print(f"\nStarting ACO optimization with {self.n_ants} ants, {self.n_iterations} iterations")
        print(f"Start node: {start_node}")
        
        # Reset best solution
        self.best_path = None
        self.best_distance = float('inf')
        self.best_distances_history = []
        
        # Run for specified number of iterations
        for iteration in range(self.n_iterations):
            # Construct solutions for all ants
            ant_paths, ant_distances = self._construct_solutions(start_node)
            
            # Update pheromones based on ant paths
            self._update_pheromones(ant_paths, ant_distances)
            
            # Store best distance for this iteration
            self.best_distances_history.append(self.best_distance)
            
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                print(f"Iteration {iteration+1}/{self.n_iterations}, Best distance: {self.best_distance:.2f}")
        
        # Plot the convergence curve
        self._plot_convergence()
        
        # Plot the pheromone levels
        self._plot_pheromone_heatmap()
        
        # Convert best path indices to original nodes
        if self.best_path:
            print("\nACO Optimization completed!")
            print(f"Best distance found: {self.best_distance:.2f}")
            return self.best_path, self.best_distance
        else:
            print("No solution found!")
            return None, float('inf')
    
    def _plot_convergence(self):
        """Plot the convergence curve of the ACO algorithm"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_distances_history) + 1), self.best_distances_history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title('ACO Convergence Curve')
        plt.grid(True)
        plt.savefig("aco_convergence.png", dpi=300, bbox_inches="tight")
        print("Convergence curve saved as aco_convergence.png")
        plt.close()
    
    def _plot_pheromone_heatmap(self):
        """Plot the pheromone matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pheromones, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title("Pheromone Levels After Optimization")
        plt.savefig("pheromone_heatmap.png", dpi=300, bbox_inches="tight")
        print("Pheromone heatmap saved as pheromone_heatmap.png")
        plt.close()


