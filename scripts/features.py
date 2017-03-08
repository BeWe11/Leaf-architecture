import numpy as np
import networkx as nx
import cvxopt as cvx
from cvxopt.modeling import variable, op
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

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

def weighted_line_graph(G, average=False):
    """ Return a line graph of G where edge attributes are propagated
    properly. Node attributes are ignored.
    If average is set to True, perform an averaging over
    conductivities.
    """
    line_graph = nx.line_graph(G)
    line_graph.add_nodes_from((tuple(sorted((u, v))), d)
            for u, v, d in G.edges_iter(data=True))

    # average
    if average:
        new_node_conds = {}
        for n, d in line_graph.nodes_iter(data=True):
            neighbor_conds = mean([line_graph.node[m]['conductivity']
                    for m in line_graph.neighbors(n)])
            new_node_conds[n] = 0.5*(d['conductivity'] +
                    neighbor_conds)

        for n, v in new_node_conds.items():
            line_graph.node[n]['conductivity'] = v

    return line_graph

def topological_length_alternative(line_graph, e, G, mode='lt'):
    """ Find the topological length associated to node e
    in the line graph. Topological length is defined as
    in the comment to topological_length_statistics.
    """
    length = 0
    length_real = 0

    current_width = line_graph.node[e]['conductivity']
    current_node = e
    edges =  [e]

    if mode == 'lt':
        comp = lambda x, y: x < y
    elif mode == 'leq':
        comp = lambda x, y: x <= y

    while True:
        # find neighboring edges
        neighs_below = [(line_graph.node[n]['conductivity'], n)
               for n in line_graph.neighbors(current_node)
               if comp(line_graph.node[n]['conductivity'], current_width)
               and not n in edges]

        # edges in 2-neighborhood
        #neighs_below_2 = [(line_graph.node[n]['conductivity'], n)
        #       for n in decomposer.knbrs(line_graph, current_node, 2)
        #       if line_graph.node[n]['conductivity'] < current_width]

        length += 1
        length_real += G[current_node[0]][current_node[1]]['weight']

        # we're at the end
        if len(neighs_below) == 0:
            break

        # use best bet from both 2 and 1 neighborhood
        max_neighs = max(neighs_below)

        current_width, current_node = max_neighs
        edges.append(current_node)

    # plot edges
    #print edges
    #plt.sca(self.leaf_subplot)
    #plot.draw_leaf_raw(G, edge_list=edges, color='r')
    #raw_input()

    return length, length_real, edges


def topological_length(G):
    total_length = 0
    line_graph = weighted_line_graph(G)
    for edge in line_graph.nodes():
        length, _, _ = topological_length_alternative(line_graph, edge, G)
        total_length += length
    return total_length / (len(line_graph.nodes()))


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



def vein_distance(G):
    """ approximate vein distances by finding the chebyshev
    centers of the areoles, and taking the radii.
"""
    distances = []
    cycle_basis = nx.cycle_basis(G)
    positions = nx.get_node_attributes(G, 'pos')
    for cycle in cycle_basis:
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

    return np.sum(distances) / len(cycle_basis)
