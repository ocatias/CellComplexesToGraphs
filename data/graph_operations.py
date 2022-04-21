import gudhi as gd
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx
import torch
from torch import Tensor
import numpy as np
import data.graph_operations as go

def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.
    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.

    FROM: https://github.com/twitter-research/cwn/blob/main/data/utils.py
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st

def build_tables(simplex_tree, size):
    """
    ADAPTED FROM: https://github.com/twitter-research/cwn/blob/main/data/utils.py
    """
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    next_id = size - 1
    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id += 1
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    # Sort id_maps
    id_maps_sorted = []
    id = 0
    for i, entry  in enumerate(id_maps):
        dict = {}
        for key, _ in entry.items():
            dict[key] = id
            id += 1

        id_maps_sorted.append(dict)

    return simplex_tables, id_maps_sorted

def encode_simplex_tree(data, simplex_tree, id_maps, size, max_clique, dim1_features = False, dimk_features = False):
    adjacencies, features, edge_features = data.edge_index, data.x, data.edge_attr 
    edgelistA, edgelistB = adjacencies[0], adjacencies[1]

    complex_dim = simplex_tree.dimension()
    original_vertex_feature_dim = features[0].shape[0]

    dim_edge_features = edge_features.shape[1]
    dim_cell_type_feature = max_clique+1
    # Only use edge_features if requested
    if not dim1_features:
        dim_edge_features = 0

    # Create a hashtable from simplex to vertex number
    simplex_to_vertex = []
    for length in range(0, min(max_clique- 1, complex_dim) + 1):
        simplex_to_vertex.append({})
        for simplex, node in id_maps[length].items():
            simplex_to_vertex[length][tuple(simplex)] = node

    new_features = [None]*sum([len(id_maps[i]) for i in range(min(max_clique, complex_dim+1))])
    for simplex,_ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        simplex_id = id_maps[simplex_dim][tuple(simplex)]

        simplex_feature = torch.zeros(dim_cell_type_feature)
        simplex_feature[simplex_dim] = 1

        # Do not include simplices that are too large
        if simplex_dim + 1 > max_clique:
            continue

        if simplex_dim == 0:
            new_features[simplex_id] = torch.cat([simplex_feature, features[simplex_id]])
        else:
            new_features[simplex_id] = torch.cat([simplex_feature, torch.zeros(original_vertex_feature_dim)])

        # Make space for edge features
        if dim1_features:
            new_features[simplex_id] = torch.cat([new_features[simplex_id], torch.zeros(dim_edge_features)])

        # Do not need to encode simplices that correspond to a single vertex as they are already part of the graph
        if simplex_dim == 0:
            continue

        # Create list of lower adjacent simplices
        lower_simplices = []
        for vertex in simplex:
            lower_simplex =  simplex.copy()
            lower_simplex.remove(vertex)
            lower_simplices.append(lower_simplex)

        # Create edges between these simplices
        for lower_simplex in lower_simplices:
            edgelistA = torch.cat([edgelistA, torch.tensor([simplex_id]), torch.tensor([id_maps[simplex_dim - 1][tuple(lower_simplex)]])])
            edgelistB = torch.cat([edgelistB, torch.tensor([id_maps[simplex_dim - 1][tuple(lower_simplex)]]), torch.tensor([simplex_id])])

    # Add features to cells of dimensions 1
    if dim1_features:
        # print(id_maps[0])
        # print(id_maps[1])
        nr_vertices = len(id_maps[0])
        for edge in id_maps[1]:
            cell_idx = simplex_to_vertex[1][edge]
            idx_in_edge_attr = 2*(cell_idx - nr_vertices)
            new_features[cell_idx][dim_cell_type_feature+original_vertex_feature_dim:] = edge_features[idx_in_edge_attr]


    # Add features to cells of dimensions > 1
    if dimk_features:
        for length in range(2, min(max_clique- 1, complex_dim) + 1):
            for simplex in id_maps[length]:
                node_id = simplex_to_vertex[length][simplex]
                simplex_feature = torch.zeros(original_vertex_feature_dim)
                for vertex in simplex:
                    simplex_feature += new_features[vertex][dim_cell_type_feature:dim_cell_type_feature+original_vertex_feature_dim]

                new_features[node_id][dim_cell_type_feature:dim_cell_type_feature+original_vertex_feature_dim] = simplex_feature

    for length in range(2, min(max_clique- 1, complex_dim) + 1):
        edges_to_create = {}
        for simplex in id_maps[length]:
            lower_dim_simplices = []
            for vertex_to_exclude in list(simplex):
                lower_dim_simplex = list(simplex)
                lower_dim_simplex.remove(vertex_to_exclude)
                lower_dim_simplices.append(tuple(lower_dim_simplex))

            # Create all edges to create (we create one entry for each direction)
            for s1 in lower_dim_simplices:
                for s2 in lower_dim_simplices:
                    if s1 == s2:
                        continue

                    key1 = str([s1, s2])
                    key2 = str([s2, s1])
                    if key1 not in edges_to_create:
                        edges_to_create[key1] = [s1, s2]
                    if key2 not in edges_to_create:
                        edges_to_create[key2] = [s2, s1]

        if len(edges_to_create) == 0:
            continue

        # Turn into edgelist format
        new_edgesA = []
        new_edgesB = []
        for _, value in edges_to_create.items():
            new_edgesA.append(simplex_to_vertex[length-1][value[0]])
            new_edgesB.append(simplex_to_vertex[length-1][value[1]])

        # Add new edges
        edgelistA = torch.cat([edgelistA, torch.tensor(new_edgesA)])
        edgelistB = torch.cat([edgelistB, torch.tensor(new_edgesB)])

    return torch.stack([edgelistA, edgelistB]), torch.stack(new_features)

def simplex_encoding_with_id_maps(data, max_complex, dim1_features = False, dimk_features = False):
    edges = data.edge_index
    nr_nodes = int(torch.max(edges) + 1)
    simplex_tree = go.pyg_to_simplex_tree(edges, nr_nodes)
    # Computes the clique complex up to the desired dim.
    simplex_tree.expansion(max_complex)
    simplex_tables, id_maps = go.build_tables(simplex_tree, nr_nodes)
    edges, features = go.encode_simplex_tree(data, simplex_tree, id_maps, nr_nodes, max_complex, dim1_features, dimk_features)
    data.edge_index = edges
    data.x = features
    return data, id_maps

def simplex_encoding(data, max_complex, dim1_features = False, dimk_features = False):
    data, id_maps = simplex_encoding_with_id_maps(data, max_complex, dim1_features, dimk_features)
    return data

def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)

    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles

    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings

def ring_enc_with_id_maps(data, max_ring_size, dim1_features = False, dimk_features = False, edge_attr_to_higher=False):
    edge_index, features, edge_attr = data.edge_index, data.x, data.edge_attr 
    id_maps = [{},{},{}]
    for idx in range(data.x.shape[0]):
        id_maps[0][(idx,)] = idx
    vertex_id = len(id_maps[0])

    edges_endpoints1, edges_endpoints2 = [], []

    vertex_feat_dim = features.shape[1]
    edge_attr_dim = edge_attr.shape[1]

    # Add (1, 0, 0) features to every vertex
    vertex_identifiers = torch.stack([torch.tensor([1,0,0]) for _ in range(len(id_maps[0]))])
    features = torch.cat((vertex_identifiers, features), dim = 1)

    # If we also encode edge values then add space for them to the features of vertices
    dim_edge_features = 0
    if dim1_features:
        dim_edge_features =  edge_attr.shape[1]
        edge_features = torch.stack([torch.zeros([dim_edge_features]) for _ in range(len(id_maps[0]))])
        features = torch.cat((features, edge_features), dim = 1)

    rings = get_rings(edge_index, max_k=7)

    edge_attr_of_new_edges = {}
    # Add edges as vertices
    for i in range(edge_index.shape[1]):
        p, q = edge_index[0, i], edge_index[1, i]

        # Only add every edge a single time (undirected edges have two entries in edge_index)
        if p > q:
            continue


        vertex_features = torch.zeros([vertex_feat_dim + dim_edge_features])
        if dimk_features:
            vertex_features = (features[p,3:3+vertex_feat_dim]+features[q,3:3+vertex_feat_dim])/2

        # Add (0, 1, 0) to vertex feature to denote that it comes from an edge
        new_feat = torch.cat((torch.tensor([0,1,0]), vertex_features))


        if dim1_features:
            new_feat[3+vertex_feat_dim:] = edge_attr[i,:]

        new_feat = torch.unsqueeze(new_feat, 0)
        features = torch.cat((features, new_feat), 0)

        edges_endpoints1 += [p,vertex_id,q,vertex_id]
        edges_endpoints2 += [vertex_id,p,vertex_id,q]

        #  Add empty features to edge_attr so we have an edge_attr for every edge
        if edge_attr_to_higher:
            new_edge_feat = edge_attr[i,:]
            edge_attr = torch.cat((edge_attr, torch.stack([new_edge_feat, new_edge_feat, new_edge_feat, new_edge_feat])), 0)
            edge_attr_of_new_edges[(vertex_id,)] = new_edge_feat
        else:
            edge_attr = torch.cat((edge_attr, torch.zeros([4,edge_attr_dim])), 0)

        id_maps[1][(int(p),int(q))] = vertex_id
        vertex_id += 1


    
    # Add rings as vertices
    newly_created_edges = {}
    for ring in rings:
        ring = list(ring)
        # Add the first element so we can easily find neighbors by just looking at the next vertex in the list
        ring.append(ring[0])

        # Add (0, 0, 1) to vertex feature to denote that it comes from a ring
        if dimk_features:
            # Aggregate features from vertices
            aggregated_features = torch.zeros([vertex_feat_dim + dim_edge_features])
            for vertex in ring:
                aggregated_features += features[id_maps[0][(vertex,)], 3:]

            aggregated_features = aggregated_features / len(ring)
            new_feat = torch.cat((torch.tensor([0, 0, 1]), aggregated_features))
        else:
            new_feat = torch.cat((torch.tensor([0, 0, 1]), torch.zeros([vertex_feat_dim + dim_edge_features])))

        new_feat = torch.unsqueeze(new_feat, 0)
        features = torch.cat((features, new_feat), 0)

        edge_vertices = []

        # Add edges from edge vertex to ring vertex
        for idx in range(len(ring) - 1):
            p, q = id_maps[0][(ring[idx],)], id_maps[0][(ring[idx+1],)]

            # Ensure that p < q
            if p > q:
                p, q = q, p

            edge_vertex = id_maps[1][(p,q)]
            edges_endpoints1 += [edge_vertex, vertex_id]
            edges_endpoints2 += [vertex_id, edge_vertex]
            edge_vertices.append(edge_vertex)

            #  Add empty features to edge_attr so we have an edge_attr for every edge

            if edge_attr_to_higher:
                lower_dim_edge_feat = edge_attr_of_new_edges[(id_maps[1][(p,q)],)]
                edge_attr = torch.cat((edge_attr, torch.stack([lower_dim_edge_feat,lower_dim_edge_feat])), 0)
            else:
                edge_attr = torch.cat((edge_attr, torch.zeros([2,edge_attr_dim])), 0)

        # Again add first element 
        edge_vertices.append(edge_vertices[0])
        
        # Add edges from edge vertex to edge_vertex if they are in the same ring
        for idx in range(len(edge_vertices) - 1):
            p, q = edge_vertices[idx], edge_vertices[idx+1]

            # Ensure that p < q
            if p > q:
                p, q = q, p

            # Ensure we do not create the same edge twice
            if (p,q) in newly_created_edges or (q,p) in newly_created_edges:
                continue
            else:
                edges_endpoints1 += [p,q]
                edges_endpoints2 += [q,p]
                newly_created_edges[(p,q)] = True
                newly_created_edges[(q,p)] = True

                #  Add empty features to edge_attr so we have an edge_attr for every edge
                if edge_attr_to_higher:
                    lower_dim_edge_feat = (edge_attr_of_new_edges[(p,)]+edge_attr_of_new_edges[(q,)])/2
                    edge_attr = torch.cat((edge_attr, torch.stack([lower_dim_edge_feat,lower_dim_edge_feat])), 0)
                else:
                     edge_attr = torch.cat((edge_attr, torch.zeros([2,edge_attr_dim])), 0)

        id_maps[2][tuple(ring)] = vertex_id
        vertex_id += 1

    # Combine everything
    edges_endpoints1 = torch.tensor(edges_endpoints1)
    edges_endpoints2 = torch.tensor(edges_endpoints2)
    edge_index = torch.cat((edge_index, torch.stack([edges_endpoints1, edges_endpoints2])), 1)

    data.edge_attr = edge_attr
    data.edge_index = edge_index
    data.x = features
    return data, id_maps

def ring_encoding(data, max_ring_size, dim1_features = False, dimk_features = False, edge_attr_to_higher=False):
    data, id_maps = ring_enc_with_id_maps(data, max_ring_size, dim1_features, dimk_features, edge_attr_to_higher)
    return data
