#!/usr/bin/env python
"""
    analyzer.py

    Contains functions which analyze tree graphs such as the ones
    obtained from decomposition.py.

    2013 Henrik Ronellenfitsch

    2017 adapted by Benjamin Weigang and Alan Preciado
"""
from numpy import *
from numpy import ma
import networkx as nx
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


def asymmetry(marked_tree):
    """ Returns the tree asymmetry after Van Pelt using
    the marked tree.
    """
    parts = [marked_tree.node[n]['partition-asymmetry']
            for n, d in marked_tree.degree_iter() if d > 1]

    weights = [marked_tree.node[n]['subtree-degree'] - 1
            for n, d in marked_tree.degree_iter() if d > 1]

    if len(weights) > 0:
        return average(parts, weights=weights)


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


def average_asymmetry(marked_tree, delta, Delta, attr='asymmetry-simple'):
    """ Returns the average asymmetry of all subtrees of tree whose
        degree is within Delta/2 from delta
    """
    asymmetries = array([marked_tree.node[n][attr]
            for n in marked_tree.nodes_iter()
            if abs(marked_tree.node[n]['subtree-degree'] - delta) <=
            Delta/2.])

    if len(asymmetries) > 0:
        return mean(asymmetries)
    else:
        return float('NaN')


def normalized_area_distribution(tree, bins):
    """ Returnes the set of normalized non-external areas associated
    to the hierarchical decomposition,
    the normalized probability distribution P[A = a] and the
    cumulative probability distribution P[A > a]
    """
    areas = array([tree.node[n]['cycle_area'] for n in tree.nodes_iter() \
            if not tree.node[n]['external']])

    areas /= areas.max()

    hist, bin_edges = histogram(areas, bins=bins, density=True)

    normed = hist/bins
    cumul = 1. - cumsum(normed)

    return areas, normed, cumul


def low_level_avg_asymmetries(tree, degree, Delta,
        attr='asymmetry-simple'):
    """ Cuts the tree at given degree level and calculates the
    average asymmetries for the resulting subtrees.
    """
    tree_new = tree.copy()

    nodes_to_rem = [n for n in tree.nodes_iter()
            if tree.node[n]['subtree-degree'] >= degree]
    tree_new.remove_nodes_from(nodes_to_rem)

    roots = [n for n in tree_new.nodes_iter()
            if len(tree_new.predecessors(n)) == 0 and
            len(tree_new.successors(n)) == 2]

    return subtree_asymmetries(tree_new, roots, Delta, attr=attr)


def subtree_asymmetries(tree, roots, Delta, attr='asymmetry-simple'):
    """ Calculates the average asymmetry functions for the
    subtrees rooted at the nodes given in roots.
    """
    subtrees = [nx.DiGraph(tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    reslt = []
    for s, r in zip(subtrees, roots):
        s.graph['root'] = r

        degree = s.node[r]['subtree-degree']
        degrees = array(sorted(list(set([
            s.node[n]['subtree-degree']
            for n in s.nodes_iter()]))))

        reslt.append([degrees,
            [average_asymmetry(s, d, Delta, attr=attr)
                for d in degrees]])

    return reslt


def get_subtrees(tree, roots, mode='all', area=0, degree=0):
    """ Extracts the subtrees rooted at roots from tree.
    If a mode is given, further restricts to a sub-subtree which
    has some desired property.

    Parameters:
        tree: The hierarchical, marked tree we are interested in.

        roots: The root node ids of the subtrees we are interested in

        mode:
            'all': extract full subtree
            'area': extract subtree whose loop area is closest to area
            'degree': extract subtree whose degree is closest to degree

        area: The area for the 'area' mode

        degree: The degree for the 'degree' mode

    Returns:
        subtrees: List of requested subtrees

        roots: List of roots of the requested subtrees
    """
    # Obtain subtrees as subgraphs and properly set root nodes
    subtrees = [nx.DiGraph(tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    if mode == 'area':
        roots = []
        for st in subtrees:
            # Find node with area closest to area
            ar, root = min([(abs(data['cycle_area'] - area), r)
                for r, data in st.nodes_iter(data=True)])
            ar = st.node[root]['cycle_area']

            roots.append(root)

            print("Subtree closest to {} has area {}, degree {}, root {}".format(area,
                    ar, st.node[root]['subtree-degree'], root))

        # Recalculate subtrees
        subtrees = [nx.DiGraph(
            tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    elif mode == 'degree':
        roots = []
        for st in subtrees:
            # Find node with degree closest to degree
            de, root = min([(abs(data['subtree-degree'] - degree), r)
                for r, data in st.nodes_iter(data=True)])
            de = st.node[root]['subtree-degree']

            roots.append(root)

            print("Subtree closest to {} has degree {}, area {}, root {}".format(
                    degree, de, st.node[root]['cycle_area'], root))

        # Recalculate subtrees
        subtrees = [nx.DiGraph(
            tree.subgraph(nx.dfs_tree(tree, r).nodes_iter()))
            for r in roots]

    # Save subtree roots in tree attributes
    for s, r in zip(subtrees, roots):
        s.graph['root'] = r
        s.node[r]['order'] = 0

    return subtrees, roots


def subtree_asymmetries_areas(tree, roots, attr='asymmetry-simple',
        area=0):
    """ Calculates the average asymmetry functions for the subtrees
    rooted at the nodes given in roots.
    Returns a list of lists (one for each subtree) of tuples
    of the form (asymmetry, degree, area)
    as well as the subtrees.

    This is the complete set of raw data from hierarchical
    decomposition.

    if area is equal to zero, returns the full selected subtrees.
    Otherwise, returns for each selected subtree that sub-subtree
    whose area is closest to the given area (in square pixels)
    """

    if area > 0:
        subtrees, roots = get_subtrees(tree, roots, mode='area',
                area=area)
    else:
        subtrees, roots = get_subtrees(tree, roots)

    reslt = []
    for s, r in zip(subtrees, roots):
        dist = [(s.node[n][attr], s.node[n]['subtree-degree'],
            s.node[n]['cycle_area']) for n in s.nodes_iter()]

        dist = [(q, d, a) for q, d, a in dist if a > 0 and d > 0]

        reslt.append(dist)

    return reslt, subtrees


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

    #areas, area_hist, cumul = normalized_area_distribution(tree, 100)
    areas = array([tree_no_ext.node[n]['cycle_area']
        for n in tree_no_ext.nodes_iter()])

    #  return horton_strahler, shreve, marked_tree, tree_no_ext, \
    #          marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
    #          areas

    return tree_asymmetry_weighted, tree_asymmetry_weighted_no_ext, \
           tree_asymmetry_unweighted, tree_asymmetry_unweighted_no_ext, \
