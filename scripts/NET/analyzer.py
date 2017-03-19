#!/usr/bin/env python
"""
    analyzer.py

    Contains functions which analyze tree graphs such as the ones
    obtained from decomposition.py.

    2013 Henrik Ronellenfitsch

    2017 adapted by Benjamin Weigang and Alan Preciado
"""
from numpy import *
import numpy as np
from numpy import ma
import networkx as nx
import cvxopt as cvx
from cvxopt.modeling import variable, op
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def weighted_line_graph(G, average=False):
    """ Return a line graph of G where edge attributes are propagated
    properly. Node attributes are ignored.
    If average is set to True, perform an averaging over
    conductivities.
    """
    line_graph = nx.line_graph(G)
    line_graph.add_nodes_from((tuple(sorted((u, v))), d)
            for u, v, d in G.edges_iter(data=True))

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


def topological_length_for_edge(line_graph, e, G, mode='lt'):
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

        length += 1
        length_real += G[current_node[0]][current_node[1]]['weight']

        # we're at the end
        if len(neighs_below) == 0:
            break

        # use best bet from both 2 and 1 neighborhood
        max_neighs = max(neighs_below)

        current_width, current_node = max_neighs
        edges.append(current_node)

    return length, length_real, edges


def mark_subtrees(tree):
    """ Modifies the given tree by adding node attributes which
    contain the subtree degree (i.e. number of leaf nodes) for
    the subtree anchored at that particular node as well as the
    partition asymmetry.

    We define
        partition-asymmetry:        |r-l|/max(r, l)
        partition-asymmetry-1:      |r-l|/(r+l-1)
        partition-asymmetry-2:      |r-l|/(r+l-2)

    where partition-asymmetry-1,2 are only defined where the denominator
    does not vanish. The respective unweighted asymmetries are
    defined accordingly
    """
    for n in nx.dfs_postorder_nodes(tree):
        succ = tree.successors(n)

        if len(succ) == 0:
            tree.node[n]['subtree-degree'] = 1
            tree.node[n]['partition-asymmetry'] = None
            tree.node[n]['partition-asymmetry-1'] = None
            tree.node[n]['partition-asymmetry-2'] = None
            tree.node[n]['sub-partition-asymmetries'] = []
            tree.node[n]['asymmetry-simple-weights'] = []
            tree.node[n]['asymmetry-simple'] = 0.
            tree.node[n]['asymmetry-unweighted'] = 0.

            tree.node[n]['level'] = 0
        else:
            s0 = tree.node[succ[0]]
            s1 = tree.node[succ[1]]

            r = s0['subtree-degree']
            s = s1['subtree-degree']

            r_parts = s0['sub-partition-asymmetries']
            s_parts = s1['sub-partition-asymmetries']

            r_wts = s0['asymmetry-simple-weights']
            s_wts = s1['asymmetry-simple-weights']

            abs_degree_diff = abs(float(r) - s)
            degree = r + s

            my_part = abs_degree_diff/max(r, s)
            my_part_1 = abs_degree_diff/(degree - 1)

            if r + s > 2:
                my_part_2 = abs_degree_diff/(degree - 2)
            else:
                my_part_2 = None

            asym_simple_wts = r_wts + s_wts + [degree - 1]
            sub_part_asym = r_parts + s_parts + [my_part]

            tree.node[n]['subtree-degree'] = degree
            tree.node[n]['partition-asymmetry'] = my_part
            tree.node[n]['partition-asymmetry-1'] = my_part_1
            tree.node[n]['partition-asymmetry-2'] = my_part_2
            tree.node[n]['asymmetry-simple-weights'] = asym_simple_wts
            tree.node[n]['sub-partition-asymmetries'] = sub_part_asym

            tree.node[n]['asymmetry-simple'] = ma.average(
                    sub_part_asym, weights=asym_simple_wts)
            tree.node[n]['asymmetry-unweighted'] = ma.average(sub_part_asym)

            tree.node[n]['level'] = max(s0['level'], s1['level']) + 1


def remove_external_nodes(tree):
    """ Returns a tree that is equivalent to the given tree,
    but all external nodes are removed.
    """
    no_ext_tree = tree.copy()

    # Remove external nodes except for root which must be kept
    root = no_ext_tree.graph['root']
    no_ext_tree.remove_nodes_from(n for n in tree.nodes_iter()
            if tree.node[n]['external'] and n != root)

    internal_nodes = [n for n in no_ext_tree.nodes() if
            len(no_ext_tree.successors(n)) == 1 \
            and len(no_ext_tree.predecessors(n)) == 1]

    for i in internal_nodes:
        pr = no_ext_tree.predecessors(i)[0]
        su = no_ext_tree.successors(i)[0]

        no_ext_tree.add_edge(pr, su)
        no_ext_tree.remove_node(i)

    # Handle case of root node
    root_succ = no_ext_tree.successors(root)

    if len(root_succ) == 1:
        s = root_succ[0]
        su = no_ext_tree.successors(s)

        no_ext_tree.add_edge(root, su[0])
        no_ext_tree.add_edge(root, su[1])
        no_ext_tree.remove_node(s)

    return no_ext_tree


def analyze_tree(tree):
    # calculate metrics
    horton_strahler = 0
    shreve = 0

    print("Constructing marked trees.")
    marked_tree = tree.copy()
    mark_subtrees(marked_tree)

    tree_no_ext = remove_external_nodes(tree)
    marked_tree_no_ext = tree_no_ext.copy()
    mark_subtrees(marked_tree_no_ext)

    print("Calculating tree asymmetry.")
    tree_asymmetry_weighted = marked_tree.node[
            marked_tree.graph['root']]['asymmetry-simple']
    tree_asymmetry_weighted_no_ext = marked_tree_no_ext.node[
            marked_tree_no_ext.graph['root']]['asymmetry-simple']
    tree_asymmetry_unweighted = marked_tree.node[
            marked_tree.graph['root']]['asymmetry-unweighted']
    tree_asymmetry_unweighted_no_ext = marked_tree_no_ext.node[
            marked_tree_no_ext.graph['root']]['asymmetry-unweighted']

    areas = array([tree_no_ext.node[n]['cycle_area']
        for n in tree_no_ext.nodes_iter()])

    #  return horton_strahler, shreve, marked_tree, tree_no_ext, \
    #          marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
    #          areas

    return tree_asymmetry_weighted, tree_asymmetry_weighted_no_ext, \
           tree_asymmetry_unweighted, tree_asymmetry_unweighted_no_ext, \


def vein_distance_net(G, cycles):
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
