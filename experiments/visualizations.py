import networkx as nx
import matplotlib.pyplot as plt
from data.utils import edge_tensor_to_list

colors = ['red', 'orange', 'yellow', 'blue', 'green']
defaul_color = 'black'

def visualize_from_edge_index(edge_index, id_maps = None):
    visualize(edge_tensor_to_list(edge_index), id_maps)

def visualize(edge_list, id_maps = None):
    G = nx.Graph()
    for edge in edge_list:
        G.add_edge(*edge)

    color_map = []
    for node in G:
        color = defaul_color

        if id_maps is not None:
            for i, dict in enumerate(id_maps):
                if node in dict.values():
                    color = colors[i]
                    break
        color_map.append(color)

    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.show()
