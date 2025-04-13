import random
import numpy as np
import folium
from datetime import datetime

class TrafficSimulator:
    def __init__(self, road_network):
        self.G = road_network
        self.traffic_levels = {
            'light': (0.8, 1.0),    # 0.8-1.0x normal travel time
            'moderate': (1.0, 1.5),  # 1.0-1.5x normal travel time
            'heavy': (1.5, 2.5),     # 1.5-2.5x normal travel time
            'severe': (2.5, 4.0)     # 2.5-4.0x normal travel time
        }
        
    def generate_traffic_conditions(self, seed=None):
        """Generate random traffic conditions for the road network"""
        if seed is None:
            # Use current time as seed for different traffic each run
            seed = int(datetime.now().timestamp())
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Make a copy of the original graph
        traffic_graph = self.G.copy()
        
        # Major roads are more likely to have heavy traffic
        major_roads = [e for e in traffic_graph.edges if 
                      traffic_graph.edges[e].get('highway') in 
                      ['motorway', 'trunk', 'primary', 'secondary']]
        
        # Minor roads have less traffic variation
        minor_roads = [e for e in traffic_graph.edges if e not in major_roads]
        
        # Simulate rush hour pattern in certain areas (more likely near city centers)
        city_center_lat, city_center_lon = 16.5069, 80.6486  # Vijayawada center
        
        # Apply traffic factors to edges
        for edge in traffic_graph.edges:
            u, v, k = edge if len(edge) == 3 else (*edge, 0)
            
            # Get edge attributes
            length = traffic_graph.edges[edge].get('length', 0)
            
            # Calculate distance from city center
            u_lat, u_lon = traffic_graph.nodes[u]['y'], traffic_graph.nodes[u]['x']
            dist_from_center = np.sqrt((u_lat - city_center_lat)**2 + (u_lon - city_center_lon)**2)
            
            # Higher traffic probability closer to center
            center_factor = max(0.1, min(1.0, 0.2 / (dist_from_center + 0.05)))
            
            # Determine traffic level based on road type and location
            if edge in major_roads:
                if random.random() < 0.4 + center_factor:  # Higher chance for traffic on major roads
                    traffic_level = random.choices(
                        ['moderate', 'heavy', 'severe'], 
                        weights=[0.5, 0.3, 0.2]
                    )[0]
                else:
                    traffic_level = 'light'
            else:
                if random.random() < 0.2 + center_factor/2:  # Lower chance on minor roads
                    traffic_level = random.choices(
                        ['light', 'moderate', 'heavy'], 
                        weights=[0.6, 0.3, 0.1]
                    )[0]
                else:
                    traffic_level = 'light'
            
            # Apply traffic multiplier to travel time
            min_factor, max_factor = self.traffic_levels[traffic_level]
            traffic_multiplier = random.uniform(min_factor, max_factor)
            
            # Update edge with traffic information
            traffic_graph.edges[edge]['traffic_level'] = traffic_level
            traffic_graph.edges[edge]['traffic_multiplier'] = traffic_multiplier
            
            # Update travel time based on traffic
            if 'travel_time' in traffic_graph.edges[edge]:
                original_time = traffic_graph.edges[edge]['travel_time']
                traffic_graph.edges[edge]['traffic_time'] = original_time * traffic_multiplier
            else:
                # If no travel_time exists, estimate based on length and speed
                # Assume 30 km/h average speed if not specified
                speed = traffic_graph.edges[edge].get('speed_kph', 30)
                travel_time = (length / 1000) / (speed / 3600)  # in seconds
                traffic_graph.edges[edge]['travel_time'] = travel_time
                traffic_graph.edges[edge]['traffic_time'] = travel_time * traffic_multiplier
        
        # Update edge weights for path calculations
        for edge in traffic_graph.edges:
            traffic_graph.edges[edge]['weight'] = traffic_graph.edges[edge]['traffic_time']
        
        return traffic_graph
    
    def visualize_traffic(self, traffic_graph, folium_map=None):
        """Add traffic visualization to the map"""
        if folium_map is None:
            # Get center coordinates
            center_node = list(traffic_graph.nodes)[0]
            center_coords = (traffic_graph.nodes[center_node]['y'], 
                            traffic_graph.nodes[center_node]['x'])
            folium_map = folium.Map(location=center_coords, zoom_start=13)
        
        # Define colors for traffic levels
        traffic_colors = {
            'light': '#3CB371',      # Green
            'moderate': '#FFD700',   # Yellow
            'heavy': '#FF8C00',      # Orange
            'severe': '#FF0000'      # Red
        }
        
        # Add traffic information to map
        for edge in traffic_graph.edges:
            u, v, k = edge if len(edge) == 3 else (*edge, 0)
            
            # Get edge geometry
            if 'geometry' in traffic_graph.edges[edge]:
                coords = [(lat, lon) for lon, lat in traffic_graph.edges[edge]['geometry'].coords]
            else:
                u_lat, u_lon = traffic_graph.nodes[u]['y'], traffic_graph.nodes[u]['x']
                v_lat, v_lon = traffic_graph.nodes[v]['y'], traffic_graph.nodes[v]['x']
                coords = [(u_lat, u_lon), (v_lat, v_lon)]
            
            # Only visualize major roads to avoid cluttering
            if traffic_graph.edges[edge].get('highway') in ['motorway', 'trunk', 'primary', 'secondary']:
                traffic_level = traffic_graph.edges[edge].get('traffic_level', 'light')
                color = traffic_colors[traffic_level]
                weight = 3 if traffic_level in ['heavy', 'severe'] else 2
                
                folium.PolyLine(
                    coords, 
                    color=color, 
                    weight=weight, 
                    opacity=0.7,
                    tooltip=f"Traffic: {traffic_level.capitalize()} ({traffic_graph.edges[edge].get('traffic_multiplier', 1.0):.1f}x)"
                ).add_to(folium_map)
        
        # Add traffic legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; 
        background-color: white; padding: 10px; border: 1px solid grey; z-index: 9999">
        <h4>Traffic Conditions</h4>
        <div><i style="background: #3CB371; width: 15px; height: 15px; display: inline-block;"></i> Light</div>
        <div><i style="background: #FFD700; width: 15px; height: 15px; display: inline-block;"></i> Moderate</div>
        <div><i style="background: #FF8C00; width: 15px; height: 15px; display: inline-block;"></i> Heavy</div>
        <div><i style="background: #FF0000; width: 15px; height: 15px; display: inline-block;"></i> Severe</div>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))
        
        return folium_map