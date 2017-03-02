import networkx as nx
import csv
import os

from scripts.decomposer import sorted_connected_components, \
                       remove_intersecting_edges, \
                       prune_graph, \
                       apply_workaround

from scripts.analyzer import analyze_tree


def clean_graph(graph):
    print("Removing disconnected parts")
    con = sorted_connected_components(graph)
    assert len(con) != 0, 'Graph is empty!'

    graph = con[0]

    print("Removing intersecting edges.")
    remove_intersecting_edges(graph)

    print("Pruning.")
    pruned = prune_graph(graph)

    print("Applying workaround to remove spurious collinear edges.")
    removed_edges = apply_workaround(pruned)

    print("Pruning again.")
    pruned = prune_graph(pruned)

    con = sorted_connected_components(pruned)
    print("Connected components:", len(con))
    assert len(con) != 0, 'Graph is empty!'

    return con[0]


def graph_from_data(node_path, edge_path):
    """
    Create networkX graph for given node and edge files.
    """
    # Extract network id from path
    network_id = os.path.basename(node_path)
    network_id = network_id.split('binary', 1)[0]
    network_id = network_id.split('graph', 1)[0]
    network_id = network_id[:-1]

    # Read node positions from the node file
    nodes = []
    with open(node_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            nodes.append({'node_id': int(row[0]), 'x': row[1], 'y': row[2]})

    # Read edge positions from the edge file
    edges = []
    with open(edge_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            edges.append({
                'node_id1': int(row[0]),
                'node_id2': int(row[1]),
                'radius': row[2],
                'length': row[3]
            })

    # Create the graph with the networkx library
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['node_id'], pos=(node['x'], node['y']))
    for edge in edges:
        G.add_edge(
            edge['node_id1'],
            edge['node_id2'],
            weight=edge['radius'],
            length=edge['length'],
            conductivity=1
        )

    #  return network_id, clean_graph(G)
    return network_id, G


def graph_generator():
    """
    Iterate over all graphs.
    """
    data_list = []
    for subset in ['BronxA', 'BronxB']:
        # Get a list of all files in the BronxA or BronxB directory
        file_list = sorted(os.listdir('data/networks-{}'.format(subset)))

        # The file names result in the node list always following the edge list,
        # which means that all even numbers correspond to edge lists and all odd
        # numbers to node lists.
        for i in range(len(file_list)//2):
            edge_path = os.path.join('data/networks-{}'.format(subset), file_list[2*i])
            node_path = os.path.join('data/networks-{}'.format(subset), file_list[2*i + 1])

            # Everytime the next element of the generator is accessed, the current step in
            # the loop is executed. After the 'yield' line is finished, the generator will pause
            # until the next element is accessed again. So everytime we access the next element,
            # we raise i to i+1, get the paths for the corresponding files and generate the graph
            # for these files.
            yield graph_from_data(node_path, edge_path)


def get_nesting_numbers(tree):
    horton_strahler, shreve, marked_tree, tree_no_ext, \
    marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
    areas = analyze_tree(tree)

    nesting_number = 1 - tree_asymmetry
    nesting_number_no_ext = 1 - tree_asymmetry_no_ext

    return nesting_number, nesting_number_no_ext
