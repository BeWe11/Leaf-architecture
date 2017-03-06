import numpy as np
import networkx as nx

from scripts.decomposer import hierarchical_decomposition

from scripts.analyzer import analyze_tree


### HELPER FUNCTIONS ###

def polygon_area(x, y):
    """
    Calculate the area of an arbitrary polygon.
    """
    return 0.5 * (np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


#Function recieves the nx.Graph generated
def get_total_leaf_area(G):
    """
    This function calculates each invidual basis cycle area, adds all the areas and
    returns the total sum, which corresponds to the leaf area
    """
    cycle_basis = nx.cycle_basis(G)
    node_positions = nx.get_node_attributes(G,'pos')
    total_leaf_area = 0

    for cycle in cycle_basis:
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
        sum_vein += G.get_edge_data(*edge)['length']   #With * python unpacks the tuple
    return sum_vein


### GEOMETRICAL FEATURES ###

def vein_density(G):
    total_vein_length = get_total_vein_length(G)
    total_leaf_area = get_total_leaf_area(G)
    return total_vein_length / total_leaf_area


def vein_distance(G):
    return


def areole_area(G):
    total_leaf_area = get_total_leaf_area(G)
    cycle_basis = nx.cycle_basis(G)
    return total_leaf_area / len(cycle_basis)   #It is mean areole area


def areole_density(G):
    """
    Individual basic cycles forming G are obtained using nx.cycle_basis
    """
    basis_cycles = nx.cycle_basis(G)   #Each list has node indices representing one basis cycle
    no_basis_cycles = len(basis_cycles)
    total_leaf_area = get_total_leaf_area(G)
    return no_basis_cycles / total_leaf_area


def weighted_vein_thickness(G):
    """
    Weighted vein thickness is calculated as the total sum of the product radius(weight)*length of each
    individual vein segment divided by total vein length
    """
    total_vein_length = get_total_vein_length(G)
    individual_weighted_vein_thickness = 0
    for edge in G.edges():
        individual_weighted_vein_thickness += G.get_edge_data(*edge)['radius']*G.get_edge_data(*edge)['length']  #vein_thickness*vein_length
    weighted_vein_thickness = individual_weighted_vein_thickness / total_vein_length
    return weighted_vein_thickness


### TOPOLOGICAL ###

def nesting_numbers(G):
    """
    Calculate nesting number for a *cleaned graph*, which means that
    'clean_graph' has been applied to G.
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
