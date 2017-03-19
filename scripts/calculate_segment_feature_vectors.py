import os
from scripts.utility import *
from scripts.features import *


feature_functions = [
    n_nodes,
    n_edges,
    average_node_degree,
    nesting_numbers,
    topological_length,
    vein_density,
    vein_distance,
    areole_area,
    areole_density,
    weighted_vein_thickness,
]

network_ids = [
    'BronxA_009',
    'BronxA_010',
    'BronxA_016',
    'BronxA_019',
    'BronxA_070',
    'BronxA_071',
    'BronxA_088',
    'BronxA_109',
    'BronxA_075',
    'BronxA_119',
]

for network_id in network_ids:
    data_list = sorted(os.listdir('data/segments/{}'.format(network_id)))
    for data_name in data_list:
        seg_number, rel_x, rel_y = data_name.split('_')
        data_path = 'data/segments/{}/{}'.format(network_id, data_name)
        G = nx.read_gpickle(data_path)
        clean_G = clean_graph(G)
        cycles = cycle_basis(clean_G)
        for feature_function in feature_functions:
            feature_name = feature_function.__name__
            file_path = 'features/segments/{}'.format(feature_name)
            with open(file_path) as file, \
                    open('log_{}'.format(feature_name), 'w') as log_file:
                print('Saving {} for {}_{}...'.format(feature_name, network_id, seg_number))
                log_file.write(
                    'Saving {} for {}_{}...\n'.format(feature_name, network_id, seg_number)
                )
                try:
                    if feature_name in ['nesting_numbers', 'topological_length']:
                        feature_value = feature_function(clean_G, cycles)
                    else:
                        feature_value = feature_function(G, cycles)
                except Exception as e:
                    print('Exception occured: {}'.format(e))
                    log_file.write('Exception occured: {}\n'.format(e))
                else:
                    file.write(network_id)
                    file.write('\t' + seg_number)
                    file.write('\t' + rel_x)
                    file.write('\t' + rel_y)
                    try:
                        for value in feature_value:
                            file.write('\t' + str(value))
                        file.write('\n')
                    except:
                        file.write('\t' + str(feature_value) + '\n')

                    print('Saved {} for {}_{}!\n'.format(feature_name, network_id), seg_number)
                    log_file.write(
                        'Saved {} for {}_{}!\n\n'.format(feature_name, network_id, seg_number)
                    )
                    file.flush()
                    log_file.flush()
