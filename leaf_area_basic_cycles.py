import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

## @author  Alan Preciado - Benjamin Weigang
## @version 1.0, 02/03/17

#This function calculates each invidual basic cycle area, adds all the areas and
# returns the total sum, which corresponds to the leaf area

#Function recieves the nx.Graph generated
def get_total_leaf_area(G):
    basic_cycles = nx.cycle_basis(G,1)   #Each list has node indices representing one basic cycle
    no_basic_cycles = len(basic_cycles)
    pos=nx.get_node_attributes(G,'pos')
    #nx.draw(G, pos=pos, node_size=9)
    #print('node positions',pos)
    #print('node coordinates type',type(pos[3]))
    coordinates = np.empty((no_basic_cycles, 0)).tolist()  #Lists of individual cycles containing node positions (x,y)

    i = 0
    for item1 in basic_cycles:
        for item2 in item1:
            coordinates[i].append( pos[item2] )  #Append node positions by looking at pos[index] returs tuple (x,y)
        i+=1

    #print('Node coordinates', coordinates)

    X = np.zeros((no_basic_cycles, 0)).tolist()  #Separate coordinates into X-Y arrays
    Y = np.zeros((no_basic_cycles, 0)).tolist()

    j = 0
    for item3 in coordinates:
        #item4 is a tuple: (x,y) --> item[0],item[1]
        for item4 in item3:
            X[j].append(item4[0])
            Y[j].append(item4[1])
        j+=1

        #print('X',X)
        #print('Y',Y)

    cycle_areas = np.zeros(no_basic_cycles)   #Store polygon areas

    def PolyArea(x,y):
        return 0.5*(np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))  #Standard Polygon Area Formula

    k=0
    for item5 in cycle_areas:
        cycle_areas[k] = PolyArea(X[k],Y[k])  #Function call: Compute single cycle (polygon) area
    #print('cycle node pos',X[k],Y[k])
    #print('cycle area',PolyArea(X[k],Y[k]))
    #print('counter',k)
        k+=1

    total_leaf_area = sum(cycle_areas)
    #print('Basic cycles areas',cycle_areas)
    #print('Total leaf area',total_leaf_area)
    return total_leaf_area
