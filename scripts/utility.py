"""
Utility functions to work with leaf networks.
"""
import csv
import os
import networkx as nx
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib.path import Path as mplPath

from scripts.decomposer import sorted_connected_components, \
                               remove_intersecting_edges, \
                               prune_graph, \
                               apply_workaround
from scripts.cycle_basis import shortest_cycles


def cycle_basis(G):
    cycle_objects = shortest_cycles(G)
    outer_loop = max([(c.area(), c) for c in cycle_objects])[1]
    cycle_objects.remove(outer_loop)
    return [cycle_object.path for cycle_object in cycle_objects]


def clean_graph(graph):
    #  print("Removing disconnected parts")
    #  con = sorted_connected_components(graph)
    #  assert len(con) != 0, 'Graph is empty!'

    #  graph = con[0]

    print("Removing nodes with same positions.")
    for u, v in graph.edges():
        if np.all(graph.node[u]['pos'] == graph.node[v]['pos']):
            graph.remove_edge(u, v)

    print("Removing intersecting edges.")
    remove_intersecting_edges(graph)

    print("Pruning.")
    pruned = prune_graph(graph)

    print("Applying workaround to remove spurious collinear edges.")
    removed_edges = apply_workaround(pruned)

    print("Pruning again.")
    pruned = prune_graph(pruned)

    con = sorted_connected_components(pruned)
    print("Connected components:", len(con), '\n')
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
        reader = csv.reader(csvfile, delimiter=' ',
                            quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            nodes.append({'node_id': int(row[0]), 'x': row[1], 'y': row[2]})

    # Read edge positions from the edge file
    edges = []
    with open(edge_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',
                            quoting=csv.QUOTE_NONNUMERIC)
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

            # Naming convention of the NET library
            conductivity=edge['radius'],
            weight=edge['length'],

            # Our naming convention
            radius=edge['radius'],
            length=edge['length'],
        )

    print("Removing disconnected parts")
    con = sorted_connected_components(G)
    assert len(con) != 0, 'Graph is empty!'

    G = con[0]
    return network_id, G


def graph_generator(skip_file=''):
    """
    Iterate over all graphs.
    """
    data_list = []
    for subset in ['BronxA', 'BronxB']:
        # Get a list of all files in the BronxA or BronxB directory
        file_list = sorted(os.listdir('data/networks-{}'.format(subset)))

        # The file names result in the node list always following the edge
        # list, which means that all even numbers correspond to edge lists
        # and all odd numbers to node lists.
        for i in range(len(file_list)//2):
            edge_path = os.path.join('data/networks-{}'.format(subset),
                                     file_list[2*i])
            node_path = os.path.join('data/networks-{}'.format(subset),
                                     file_list[2*i + 1])

            # If a skip_file is given, check whether the id of the current
            # network is in there, if it is skip the current network
            if skip_file:
                if os.path.isfile(skip_file):
                    network_id = os.path.basename(node_path)
                    network_id = network_id.split('binary', 1)[0]
                    network_id = network_id.split('graph', 1)[0]
                    network_id = network_id[:-1]
                    # Get whole file content to check for existing entries
                    with open(skip_file, 'r') as file:
                        content = file.read()
                    if network_id in content:
                        continue

            # Everytime the next element of the generator is accessed, the
            # current step in the loop is executed. After the 'yield' line is
            # finished, the generator will pause until the next element is
            # accessed again. So everytime we access the next element, we
            # raise i to i+1, get the paths for the corresponding files and
            # generate the graph for these files.
            yield graph_from_data(node_path, edge_path)


def save_feature(feature_function, skip_existing=True, clean=False):
    feature_name = feature_function.__name__
    if clean:
        file_path = 'features/{}_clean.txt'.format(feature_name)
    else:
        file_path = 'features/{}.txt'.format(feature_name)

    if skip_existing:
        skip_file = file_path
    else:
        skip_file = ''

    # If we want to skip feature calculation for networks which already have
    # a value in the file, we want to append values with mode 'a', otherwise
    # we want to create a new file with 'w'
    if skip_existing:
        write_mode = 'a'
    else:
        write_mode = 'w'

    with open(file_path, write_mode) as file, \
            open('log_{}'.format(feature_name), 'w') as log_file:
        for network_id, G in graph_generator(skip_file):
            print('Saving {} for {}...'.format(feature_name, network_id))
            log_file.write(
                'Saving {} for {}...\n'.format(feature_name, network_id)
            )

            # If anytime during feature calculation there is an error, we catch
            # it, write it to a log file and proceed with the next network
            try:
                if clean:
                    G = clean_graph(G)
                feature_value = feature_function(G)
            except Exception as e:
                print('Exception occured: {}'.format(e))
                log_file.write('Exception occured: {}\n'.format(e))
            else:
                file.write(network_id)
                try:
                    for value in feature_value:
                        file.write('\t' + str(value))
                    file.write('\n')
                except:
                    file.write('\t' + str(feature_value) + '\n')

                print('Saved {} for {}!\n'.format(feature_name, network_id))
                log_file.write(
                    'Saved {} for {}!\n\n'.format(feature_name, network_id)
                )
                file.flush()
                log_file.flush()


species_dict = {}
for k in [1, 2]:
    with open('data/LABELS_{}_FIXED.txt'.format(k)) as file:   # BronxB LABEL1, BronxA LABEL2
        reader = csv.reader(file, delimiter='\t')  # Works with elements as strings
        for row in reader:
            network_id = row[0].strip()
            genus, species = row[1].split(' ', 1)
            species_dict[network_id] = {'genus': genus, 'species': species}


def species_from_id(network_id):
    return species_dict[network_id[:10]]


def read_features():
    feature_names = [
        'topological_length',
        'nesting_numbers',
        'vein_density',
        'vein_distance',
        'areole_area',
        'areole_density',
        'weighted_vein_thickness',
        'average_node_degree',
        'n_edges',
        'n_nodes',
    ]
    data = {}
    for feature_name in feature_names:
        if feature_name == 'nesting_numbers':
            values_list = [{}, {}, {}, {}]
            with open('features/{}.txt'.format(feature_name)) as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    network_id = row[0]
                    for k, value in enumerate([float(x) for x in row[1:]]):
                        values_list[k][network_id] = value
            data['nesting_number_weighted'] = values_list[0]
            data['nesting_number_weighted_no_ext'] = values_list[1]
            data['nesting_number_unweighted'] = values_list[2]
            data['nesting_number_unweighted_no_ext'] = values_list[3]
        else:
            values = {}
            with open('features/{}.txt'.format(feature_name)) as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    network_id = row[0]
                    value = float(row[1])
                    values[network_id] = value
            data[feature_name] = values
    return data


def find_factor(a, product=100):
    """
    Helper function for the grid generation.
    """
    current_product = a
    b = 1
    while current_product < product:
        current_product += a
        b += 1
    return b


class Cell():
    """
    A square cell containing nodes.
    """
    def __init__(self, corners):
        # Convention:   0: bottom left, 1: bottom right, 2: top left, 3: top right
        self.corners = np.array(corners)
        self.nodes = []

    @property
    def area(self):
        return self.length ** 2

    @property
    def length(self):
        return np.linalg.norm(self.corners[1] - self.corners[0])

    @property
    def center(self):
        return np.mean(self.corners, axis=0)

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        try:
            self.nodes.remove(node)
        except:
            pass


class CellGrid():
    """
    The cell grid with which a leaf is segmented.
    """
    def __init__(self, G, n_cells_desired=100, min_cell_length=0,
                 fill_ratio_threshold=0.5):
        self.G = G
        self.node_positions = nx.get_node_attributes(G, 'pos')

        self.find_optimal_cells(n_cells_desired, min_cell_length, fill_ratio_threshold)
        # BUG: some nodes exist multiple times in a cell after
        # remove_partially_filled_cells is called. This good be a consequence
        # of the wrong cycle basis (seems work fine now with correct
        # cycle basis)
        self.remove_partially_filled_cells(fill_ratio_threshold)
        self.repair_cycles()

    @property
    def segments(self):
        node_coords = np.array([self.node_positions[key] for key in self.G.nodes()])
        max_coords = np.max(node_coords, axis=0)
        min_coords = np.min(node_coords, axis=0)

        leaf_center = np.mean(node_coords, axis=0)
        cell_centers = np.array([cell.center for cell in self.cells])
        distances = cell_centers - leaf_center
        relative_distances = []
        for dist in distances:
            if dist[0] < 0:
                rel_x = -dist[0] / min_coords[0]
            else:
                rel_x = dist[0] / max_coords[0]
            if dist[1] < 0:
                rel_y = -dist[1] / min_coords[1]
            else:
                rel_y = dist[1] / max_coords[1]
            relative_distances.append([rel_x, rel_y])


        segments = [nx.Graph(nx.subgraph(self.G, cell.nodes)) for cell in self.cells]
        return segments, relative_distances

    def find_optimal_cells(self, n_cells_desired, min_cell_length, fill_ratio_threshold):
        node_coords = np.array(
            [self.node_positions[key] for key in sorted(self.node_positions.keys())]
        )

        ### Get enscribing rectangle
        max_nodes = np.max(node_coords, axis=0)
        min_nodes = np.min(node_coords, axis=0)

        # Convention:   0: bottom left, 1: bottom right, 2: top left, 3: top right
        outer_rect = np.array([[min_nodes[0], min_nodes[1]],
                               [max_nodes[0], min_nodes[1]],
                               [min_nodes[0], max_nodes[1]],
                               [max_nodes[0], max_nodes[1]]])

        rect_sides = (max_nodes[0] - min_nodes[0],
                      max_nodes[1] - min_nodes[1])
        short_side_index = np.argmin(rect_sides)
        short_side = rect_sides[short_side_index]
        long_side_index = np.argmax(rect_sides)
        long_side = rect_sides[long_side_index]

        ### Get cells
        # Increase amount of cells until the number of kept cells exceeds the
        # the desired number of cells, or when the cell length becomes too small
        # in comparison to the average vein distance
        cells_to_keep = []
        cell_length = 1e20
        min_n_cells = n_cells_desired
        # FIXME: The vein lengths seem to be much to big, the loop always stops
        # after the first iteration (this is due to them being calculated with
        # the wrong cycle basis!)
        while len(cells_to_keep) < n_cells_desired and cell_length > min_cell_length:
            # Reduce cell size until the desired number of cells is reached and
            # the resulting grid fits almost onto the enscribing rectangle.
            n_cells_short = 1
            overlap = 1e21
            while overlap > cell_length:
                n_cells_short += 1
                n_cells_long = find_factor(n_cells_short, product=min_n_cells)
                cell_length = short_side / n_cells_short
                overlap = n_cells_long * cell_length - long_side


            # We try to find the smallest number of cells in the long direction
            # while the total cell length should be larger than the rectangle
            # side length. If the total length becomes smaller, we increase the
            # size in the short direction, so that the cell length in the long
            # direction is equal to the rectangle side length.
            cells_center = np.array([0, 0])
            if overlap < 0:
                cell_length = long_side / n_cells_long
                overlap = n_cells_short * cell_length - short_side
                cells_center[short_side_index] = (rect_sides[short_side_index] + overlap) / 2
                cells_center[long_side_index] = rect_sides[long_side_index] / 2
            else:
                cells_center[short_side_index] = rect_sides[short_side_index] / 2
                cells_center[long_side_index] = (rect_sides[long_side_index] + overlap) / 2

            n_cells = [0, 0]
            n_cells[short_side_index] = n_cells_short
            n_cells[long_side_index] = n_cells_long

            cells = []
            for x in range(n_cells[0]):
                for y in range(n_cells[1]):
                    # Convention:   0: bottom left, 1: bottom right, 2: top left, 3: top right
                    corners = np.array([[x, y], [x+1, y], [x, y+1], [x+1, y+1]])
                    corners = corners * cell_length + (np.mean(outer_rect, axis=0) - cells_center)
                    cells.append(Cell(corners))

            self.cells = cells
            self.fill_cells()

            # Check how many cells we would currently keep
            nodes_per_cell = [len(cell.nodes) for cell in self.cells]
            average_nodes_per_cell = np.mean([nodes for nodes in nodes_per_cell if nodes > 0])
            cell_fill_ratios = nodes_per_cell / average_nodes_per_cell
            cells_to_keep = [cell for cell, ratio in zip(self.cells, cell_fill_ratios)
                               if ratio >= fill_ratio_threshold]

            # If the number of kept cells is not high enough, use at least one
            # cell more next time
            min_n_cells = n_cells_short * n_cells_long + 1

    def fill_cells(self):
        # For every node, find the closest cells and put it in the there
        cells = self.find_closest_cells(self.G.nodes())
        for cell, node in zip(cells, self.G.nodes()):
            cell.add_node(node)

    def find_closest_cells(self, nodes):
        cell_centers = np.array([cell.center for cell in self.cells])
        coords = np.array([self.node_positions[node] for node in nodes])
        coords = np.repeat(coords[:, :, np.newaxis], len(cell_centers), axis=2)
        distances = np.linalg.norm(coords - cell_centers.T, axis=1)
        cell_indices = np.argmin(distances, axis=1)
        cells = [self.cells[index] for index in cell_indices]
        return cells

    def remove_partially_filled_cells(self, fill_ratio_threshold):
        # First remove completely empty nodes
        for cell in self.cells[:]:
            if not cell.nodes:
                self.cells.remove(cell)

        # Now remove all cells which don't have enough nodes
        nodes_per_cell = [len(cell.nodes) for cell in self.cells]
        cell_fill_ratios = nodes_per_cell / np.mean(nodes_per_cell)
        cells_to_remove = [cell for cell, ratio in zip(self.cells, cell_fill_ratios)
                           if ratio < fill_ratio_threshold]
        self.remove_cells(cells_to_remove)

    def remove_cells(self, cells):
        # Allow giving only a single cell
        try:
            iter(cells)
        except TypeError:
            cells = [cells]

        for cell in cells:
            # Remove the cell and distribute its nodes to the closest other
            # cells
            nodes = cell.nodes
            self.cells.remove(cell)
            closest_cells = self.find_closest_cells(nodes)
            for closest_cell, node in zip(closest_cells, nodes):
                closest_cell.add_node(node)

    def repair_cycles(self):
        cycles = cycle_basis(clean_graph(self.G))
        cycle_centers = []
        for cycle in cycles:
            cycle_center = np.mean([self.node_positions[node] for node in cycle], axis=0)
            cycle_centers.append(cycle_center)

        largest_cycle_size = max([len(cycle) for cycle in cycles])
        cycle_nodes_in_cells = np.zeros((len(self.cells), len(cycles), largest_cycle_size), dtype=int)
        for cell_index, cell in enumerate(self.cells):
            print('cell_index: {}'.format(cell_index))
            for cycle_index, cycle in enumerate(cycles):
                node_index = 0
                for node in cell.nodes:
                    if node in cycle:
                        cycle_nodes_in_cells[cell_index, cycle_index, node_index] = node
                        node_index += 1

        counts = (cycle_nodes_in_cells != 0).sum(2)
        max_cell_indices = np.argmax(counts, axis=0)

        n_cycles = len(max_cell_indices)
        for cycle_index, cell_index in enumerate(max_cell_indices):
            print('cycle_index: {} / {}'.format(cycle_index, n_cycles))
            cycle = cycles[cycle_index]
            cycle_center = cycle_centers[cycle_index]

            tree_structures = []
            all_cycle_nodes = list(itertools.chain(*cycles))
            for node in cycle:
                neighbors = self.G.neighbors(node)
                for neighbor in neighbors:
                    if not neighbor in all_cycle_nodes:
                        polygon = mplPath(np.array([self.node_positions[node] for node in cycle]))
                        if polygon.contains_point(self.node_positions[neighbor]):
                            tree_structure = self.traverse_tree_structure(neighbor, node, all_cycle_nodes)
                            tree_structures.append(tree_structure)
            for tree_structure in tree_structures:
                for node in tree_structure:
                    for cell in self.cells:
                        cell.remove_node(node)
                    self.cells[cell_index].add_node(node)

            self.cells[cell_index].nodes = list(set(self.cells[cell_index].nodes + list(cycle)))

    def traverse_tree_structure(self, start_node, root_node, all_cycle_nodes):
        visited_nodes = [root_node, start_node]
        current_nodes = [start_node]
        while len(current_nodes) > 0:
            neighbors = list(itertools.chain(*[self.G.neighbors(node) for node
                                               in current_nodes]))
            new_neighbors = [node for node in neighbors
                             if node not in visited_nodes + all_cycle_nodes]
            current_nodes = []
            visited_nodes += new_neighbors
            current_nodes += new_neighbors
        return visited_nodes[1:]


def normalized_graph(network_id, G):
    angles = {
        'BronxA_009': -71,
        'BronxA_010': -82,
        'BronxA_016': -98,
        'BronxA_019': 152,
        'BronxA_070': 7,
        'BronxA_071': -89,
        'BronxA_088': -85,
        'BronxA_109': 5,
        'BronxA_075': -4,
        'BronxA_119': -2,
    }
    # Shift leaf center of mass to origin
    pos = nx.get_node_attributes(G, 'pos')
    node_coords = np.array([pos[key] for key in sorted(G.nodes())])
    center = np.mean(node_coords, axis=0)
    node_coords -= center

    # Rotate leaf so that the tip points to north, counter-clockwise
    # in comparison to the average vein distance
    phi = -2*np.pi/360 * -angles[network_id]
    rotary_matrix = np.array([[np.cos(phi), -np.sin(phi)],
                              [np.sin(phi), np.cos(phi)]])
    node_coords = np.dot(rotary_matrix, node_coords.T).T

    H = G.copy()
    pos = {node: node_coord for node, node_coord in zip(sorted(G.nodes()), node_coords)}
    nx.set_node_attributes(H, 'pos', pos)
    return H


def partition_graph(network_id, G):
    with open('features/vein_distance.txt') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            id = row[0]
            if id == network_id:
                vein_distance = float(row[1])

    G = normalized_graph(network_id, G)
    cell_grid = CellGrid(
        G,
        n_cells_desired=20,
        min_cell_length=0,
        #  min_cell_length=10*vein_distance,
        fill_ratio_threshold=0.75,
    )
    print('len(cell_grid.cells): {}'.format(len(cell_grid.cells)))

    colors = itertools.cycle(['red', 'green', 'blue', 'yellow', 'brown',
                              'orange', 'purple', 'cyan', 'magenta'])

    segments, relative_distances = cell_grid.segments
    print(relative_distances)

    for seg_number, segment in enumerate(segments):
        nx.write_gpickle(segment, 'data/segments/{}/{:02d}_{}_{}'.format(
            network_id, seg_number, relative_distances[seg_number][0], relative_distances[seg_number][1]))
        #  pos = nx.get_node_attributes(segment, 'pos')
        #  color = next(colors)
        #  nx.draw(segment, pos=pos, node_size=0.2, edge_size=0.1,
        #          node_color=color, edge_color=color, alpha=1.0)

    #  plt.show()

