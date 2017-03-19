import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from scripts.NET.decomposer import hierarchical_decomposition
from scripts.NET.analyzer import analyze_tree, topological_length_for_edge, \
                                 weighted_line_graph, vein_distance_net


### HELPER FUNCTIONS ###

def polygon_area(x, y):
    """
    Calculate the area of an arbitrary polygon.
    """
    return 0.5 * (np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_total_leaf_area(G, cycles):
    """
    This function calculates each invidual basis cycle area, adds all the areas and
    returns the total sum, which corresponds to the leaf area.
    """
    node_positions = nx.get_node_attributes(G,'pos')
    total_leaf_area = 0

    for cycle in cycles:
        x = []
        y = []
        for node in cycle:
            pos = node_positions[node]
            x.append(pos[0])
            y.append(pos[1])
        leaf_area = polygon_area(np.array(x), np.array(y))
        total_leaf_area += leaf_area

    return total_leaf_area


def get_total_vein_length(G):
    sum_vein = 0
    for edge in G.edges():
        sum_vein += G.get_edge_data(*edge)['length']
    return sum_vein


### GEOMETRICAL FEATURES ###

def n_nodes(G, cycles):
    """
    Return number of nodes.
    """
    return nx.number_of_nodes(G)


def n_edges(G, cycles):
    """
    Return number of edges.
    """
    return nx.number_of_edges(G)


def average_node_degree(G, cycles):
    return np.mean(list(G.degree().values()))


def vein_density(G, cycles):
    """
    Return vein length per area.
    """
    total_vein_length = get_total_vein_length(G)
    total_leaf_area = get_total_leaf_area(G, cycles)
    return total_vein_length / total_leaf_area


def areole_area(G, cycles):
    """
    Return mean areole area.
    """
    total_leaf_area = get_total_leaf_area(G, cycles)
    return total_leaf_area / len(cycles)


def areole_density(G, cycles):
    """
    Return number of areoles per area.
    """
    no_cycles = len(cycles)
    total_leaf_area = get_total_leaf_area(G, cycles)
    return no_cycles / total_leaf_area


def weighted_vein_thickness(G, cycles):
    """
    Return average product of length*radius for all edges.
    """
    total_vein_length = get_total_vein_length(G)
    individual_weighted_vein_thickness = 0
    for edge in G.edges():
        individual_weighted_vein_thickness += G.get_edge_data(*edge)['radius']*G.get_edge_data(*edge)['length']
    weighted_vein_thickness = individual_weighted_vein_thickness / total_vein_length
    return weighted_vein_thickness


def vein_distance(G, cycles):
    """
    Return average distance between veins approximated by the radii of
    circles inscribed into the areoles.
    """
    return vein_distance_net(G, cycles)


### TOPOLOGICAL ###

def topological_length(G, cycles):
    """
    Return average tapering length, i.e. the number of nodes one can follow
    from an starting edge one follows the thickest neighboring edge that is
    smaller than then current edge. 'G' has to be clean, i.e. 'clean_graph'
    has been applied to it.
    """
    total_length = 0
    line_graph = weighted_line_graph(G)
    for edge in line_graph.nodes():
        length, _, _ = topological_length_for_edge(line_graph, edge, G)
        total_length += length
    return total_length / (len(line_graph.nodes()))


def nesting_numbers(G, cycles):
    """
    Return the number, i.e. the average left-right asymmetry in the nesting
    tree. 'G' has to be clean, i.e. 'clean_graph' has been applied to it.
    """
    tree, _, _ = hierarchical_decomposition(G)
    tree_asymmetry_weighted, tree_asymmetry_weighted_no_ext, \
    tree_asymmetry_unweighted, tree_asymmetry_unweighted_no_ext = analyze_tree(tree)

    nesting_number_weighted = 1 - tree_asymmetry_weighted
    nesting_number_weighted_no_ext = 1 - tree_asymmetry_weighted_no_ext
    nesting_number_unweighted = 1 - tree_asymmetry_unweighted
    nesting_number_unweighted_no_ext = 1 - tree_asymmetry_unweighted_no_ext

    return nesting_number_weighted, nesting_number_weighted_no_ext, \
           nesting_number_unweighted, nesting_number_unweighted_no_ext
