import numpy as np
import networkx as nx
import cvxopt as cvx
from cvxopt.modeling import variable, op
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

from scripts.decomposer import hierarchical_decomposition

from scripts.analyzer import analyze_tree, topological_length_for_edge, \
                             weighted_line_graph
from scripts.utility import cycle_basis


### HELPER FUNCTIONS ###

def polygon_area(x, y):
    """
    Calculate the area of an arbitrary polygon.
    """
    return 0.5 * (np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_total_leaf_area(G, cycles):
    """
    This function calculates each invidual basis cycle area, adds all the areas and
    returns the total sum, which corresponds to the leaf area
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
    return nx.number_of_nodes(G)

def n_edges(G, cycles):
    return nx.number_of_edges(G)

def average_node_degree(G, cycles):
    return np.mean(list(G.degree().values()))

def vein_density(G, cycles):
    total_vein_length = get_total_vein_length(G)
    total_leaf_area = get_total_leaf_area(G, cycles)
    return total_vein_length / total_leaf_area


def areole_area(G, cycles):
    total_leaf_area = get_total_leaf_area(G, cycles)
    return total_leaf_area / len(cycles)


def areole_density(G, cycles):
    """
    Individual basic cycles forming G are obtained using nx.cycle_basis
    """
    no_cycles = len(cycles)
    total_leaf_area = get_total_leaf_area(G, cycles)
    return no_cycles / total_leaf_area


def weighted_vein_thickness(G, cycles):
    """
    Weighted vein thickness is calculated as the total sum of the product radius(weight)*length of each
    individual vein segment divided by total vein length
    """
    total_vein_length = get_total_vein_length(G)
    individual_weighted_vein_thickness = 0
    for edge in G.edges():
        individual_weighted_vein_thickness += G.get_edge_data(*edge)['radius']*G.get_edge_data(*edge)['length']
    weighted_vein_thickness = individual_weighted_vein_thickness / total_vein_length
    return weighted_vein_thickness


### TOPOLOGICAL ###

def topological_length(G, cycles):
    total_length = 0
    line_graph = weighted_line_graph(G)
    for edge in line_graph.nodes():
        length, _, _ = topological_length_for_edge(line_graph, edge, G)
        total_length += length
    return total_length / (len(line_graph.nodes()))


def nesting_numbers(G, cycles):
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



def vein_distance(G, cycles):
    """ approximate vein distances by finding the chebyshev
    centers of the areoles, and taking the radii.
"""
    distances = []
    positions = nx.get_node_attributes(G, 'pos')
    for cycle in cycles:
        coords = np.array([positions[node] for node in cycle])
        cvx.solvers.options['show_progress'] = False

        # find convex hull to make approximation
        # possible
        hull = ConvexHull(coords)
        coords = coords[hull.vertices,:]

        # shift to zero center of gravity
        cog = coords.mean(axis=0)

        coords -= cog
        # append last one
        coords = np.vstack((coords, coords[0,:]))

        # Find Chebyshev center
        X = cvx.matrix(coords)
        m = X.size[0] - 1

        # Inequality description G*x <= h with h = 1
        G, h = cvx.matrix(0.0, (m,2)), cvx.matrix(0.0, (m,1))
        G = (X[:m,:] - X[1:,:]) * cvx.matrix([0., -1., 1., 0.],
                (2,2))
        h = (G * X.T)[::m+1]
        G = cvx.mul(h[:,[0,0]]**-1, G)
        h = cvx.matrix(1.0, (m,1))

        # Chebyshev center
        R = variable()
        xc = variable(2)
        lp = op(-R, [ G[k,:]*xc + R*cvx.blas.nrm2(G[k,:]) <= h[k]
            for k in range(m) ] +[ R >= 0] )

        lp.solve()
        R = R.value
        xc = xc.value

        if lp.status == 'optimal':
            distances.append(R[0])

    return np.sum(distances) / len(cycles)
