import networkx as nx
import numpy as np
import csv
import os

from scripts.decomposer import sorted_connected_components, \
                       remove_intersecting_edges, \
                       prune_graph, \
                       apply_workaround, \
                       hierarchical_decomposition

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


def get_nesting_numbers(G):
    """
    Calculate nesting number for a *cleaned graph*, which means that
    'clean_graph' has been applied to G.
    """
    tree, _, _ = hierarchical_decomposition(G)
    horton_strahler, shreve, marked_tree, tree_no_ext, \
    marked_tree_no_ext, tree_asymmetry, tree_asymmetry_no_ext, \
    areas = analyze_tree(tree)

    nesting_number = 1 - tree_asymmetry
    nesting_number_no_ext = 1 - tree_asymmetry_no_ext

    return nesting_number, nesting_number_no_ext


def PolyArea(x, y):
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
    basis_cycles = nx.cycle_basis(G, 1)   #Each list has node indices representing one basis cycle
    no_basis_cycles = len(basis_cycles)
    node_positions = nx.get_node_attributes(G,'pos')

    coordinates = np.empty((no_basis_cycles, 0)).tolist()  #Lists of individual cycles containing node positions (x,y)

    i = 0
    for cycle in basis_cycles:
        for node in cycle:
            coordinates[i].append(node_positions[node])  #Append node positions by looking at pos[index] returs tuple (x,y)
        i+=1


    X = np.zeros((no_basis_cycles, 0)).tolist()  #Separate coordinates into X-Y arrays
    Y = np.zeros((no_basis_cycles, 0)).tolist()

    j = 0
    for item3 in coordinates:
        #item4 is a tuple: (x,y) --> item[0],item[1]
        for item4 in item3:
            X[j].append(item4[0])
            Y[j].append(item4[1])
        j+=1


    cycle_areas = np.zeros(no_basis_cycles)   #Store polygon areas


    k=0
    for item5 in cycle_areas:
        cycle_areas[k] = PolyArea(X[k],Y[k])  #Function call: Compute single cycle (polygon) area
        k+=1

    total_leaf_area = sum(cycle_areas)
    return total_leaf_area


def get_total_vein_length(G):
    sum_vein = 0
    for edge in G.edges():
        sum_vein += G.get_edge_data(*edge)['length']   #With * python unpacks the tuple
    return sum_vein


def get_vein_density(G):
    total_vein_length = get_total_vein_length(G)
    total_leaf_area = get_total_leaf_area(G)
    return total_vein_length / total_leaf_area

def get_areole_density(G):
    """
    Individual basic cycles forming G are obtained using nx.cycle_basis
    """
    basis_cycles = nx.cycle_basis(G, 1)   #Each list has node indices representing one basis cycle
    no_basis_cycles = len(basis_cycles)
    total_leaf_area = get_total_leaf_area(G)
    return no_basis_cycles/total_leaf_area

def get_weighted_vein_thickness(G):
    """
    Weighted vein thickness is calculated as the total sum of the product radius(weight)*length of each
    individual vein segment divided by total vein length
    """
    total_vein_length = get_total_vein_length(G)
    individual_weighted_vein_thickness = 0
    for edge in G.edges():
        individual_weighted_vein_thickness += G.get_edge_data(*edge)['weight']*G.get_edge_data(*edge)['length']  #vein_thickness*vein_length
    weighted_vein_thickness = individual_weighted_vein_thickness/total_vein_length
    return weighted_vein_thickness


# It's not necessary to make a function for this, just use this dictionary like
# 'species = species_from_id[network_id]'
species_from_id = {}
for k in [1, 2]:
    with open('data/LABELS_{}_FIXED.txt'.format(k)) as file:   #BronxB LABEL1, BronxA LABEL2
        reader = csv.reader(file, delimiter='\t')  #Works with elements as strings
        for row in reader:
            network_id = row[0].strip()
            spec = row[1]
            species_from_id[network_id] = spec
