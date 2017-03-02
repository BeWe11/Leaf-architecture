import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_total_vein_length(G):
    sum_vein = 0
    for edge in G.edges():
        sum_vein = G.get_edge_data(*edge)['length'] + sum_vein   #With * python unpacks the tuple
    return sum_vein
