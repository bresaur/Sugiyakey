"""Processing directed graphs"""

import os
import itertools

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

from . import draw, optim

def get_graph_from_df(flow_df):
    """Get directed graph from flow dataframe
    
    Args:
        flow_df: a dataframe with columns source, target and value
        
    Returns:
        a networkx digraph
    """
    dig = nx.DiGraph()
    node_names = list(set(flow_df['source']).union(set(flow_df['target'])))
    dig.add_nodes_from(node_names)
    dig.add_edges_from(zip(flow_df['source'], flow_df['target'],
                          [{'value': val} for val in flow_df['value']]))
    
    # Calculate "node values"
    for n in dig.nodes:
        dig.nodes[n]['in_value'] = 0
        dig.nodes[n]['out_value'] = 0

    for n_src, n_targ, edge_data in dig.edges.data():
        edge_val = edge_data['value']
        dig.nodes[n_targ]['in_value'] += edge_val
        dig.nodes[n_src]['out_value'] += edge_val

    for n in dig.nodes:
        dig.nodes[n]['max_value'] = max(dig.nodes[n]['in_value'],
                                        dig.nodes[n]['out_value'])
    
    return dig

def assign_layers(dig):
    """Assign nodes to layers by length of longest directed path from each node
    
    Args:
        dig: a networkx digraph
        
    Returns:
        None: the digraph is modified in place
    """

    dig_copy = dig.copy()
    
    layer = 0
    while dig_copy.number_of_nodes() > 0:
        nodes_to_remove = [node for node in dig_copy if dig_copy.out_degree(node) == 0]
        print(nodes_to_remove)
        for node in nodes_to_remove:
            dig.nodes[node]['layer'] = layer
        for node in nodes_to_remove:
            dig_copy.remove_node(node)
        layer -= 1

def get_layers(dig):
    """Get layers of directed graph
    
    Args:
        dig: a directed graph
        
    Returns:
        A list of strings
    """
    return sorted(list(set([dig.nodes[n]['layer'] for n in dig.nodes])))

def get_layer_nodes(dig, lay, key_name=None):
    """Get nodes of layer
    
    Args:
        dig: a directed graph
        lay: a string, name of layer
        key_name: an optional string to sort the nodes

    Returns:
        a list of strings, names of nodes
    """
    if key_name is None:
        key_fun = None
    else:
        key_fun = lambda nod: dig.nodes[nod][key_name]
    layer_nodes = sorted([n for n in dig.nodes if dig.nodes[n]['layer']==lay], key=key_fun)
    return layer_nodes

def add_dummy_nodes(dig):
    """Add dummy node(s) for edges bridging more 2 or more layers
    
    Args:
        dig: a directed graph
        
    Returns:
        None
    """
    edges_to_break = []
    for edge in dig.edges:
        v_par, v_chi = edge
        lay_diff = dig.nodes[v_chi]['layer'] - dig.nodes[v_par]['layer']
        if lay_diff == 0:
            raise NotImplementedError('Link on the same layer?')
        elif lay_diff == 1:
            pass
        elif lay_diff < 0:
            raise NotImplementedError('Backward link?')
        else:
            edges_to_break.append(edge)
        
    if len(edges_to_break)>0:
        print(f'We need a dummy node for {edges_to_break}')

    for edge in edges_to_break:
        v_par, v_chi = edge
        edge_val = dig.edges[edge]['value']
        lay_diff = dig.nodes[v_chi]['layer'] - dig.nodes[v_par]['layer']
        for i_dum in range(lay_diff-1):
            dummy_name = f'Dummy {edge} {i_dum}'
            dummy_layer = dig.nodes[v_par]['layer']+i_dum+1
            dig.add_node(dummy_name, layer=dummy_layer, in_value=edge_val, out_value=edge_val, max_value=edge_val)
        
            print(f'Adding {dummy_name}')
            if i_dum == 0:
                prev_node = edge[0]
            else:
                prev_node = f'Dummy {edge} {i_dum-1}'
            dig.add_edge(prev_node, dummy_name, value=edge_val)
            if i_dum == lay_diff-2:
                dig.add_edge(dummy_name, edge[1], value=edge_val)
        dig.remove_edge(v_par, v_chi)

def determine_relative_vertical_position(dig, key_name=None, verbose=False):
    """Assign/initialize relative vertical position (integer) within each layer in the simplest possible way
    
    Args:
        dig
        key_name: None or a string on which to sort nodes of each layer
        
    Returns:
        None
    """

    layers = get_layers(dig)
    if verbose:
        print(layers)

    layer_node_dict = {lay: get_layer_nodes(dig, lay, key_name=key_name) for lay in layers}
    if verbose:
        print(layer_node_dict)

    for n in dig.nodes:
        node_layer = dig.nodes[n]['layer']
        dig.nodes[n]['vertical_position'] = layer_node_dict[node_layer].index(n)
    
    return layers, layer_node_dict

def edges_cross(dig, edge_a, edge_b, verbose=False):
    """Determine if two edges cross"""
    z_a = dig.nodes[edge_a[0]]['vertical_position'], dig.nodes[edge_a[1]]['vertical_position']
    z_b = dig.nodes[edge_b[0]]['vertical_position'], dig.nodes[edge_b[1]]['vertical_position']
    vert_pos_diff_from = (dig.nodes[edge_b[0]]['vertical_position'] - dig.nodes[edge_a[0]]['vertical_position'])
    vert_pos_diff_to = (dig.nodes[edge_b[1]]['vertical_position'] - dig.nodes[edge_a[1]]['vertical_position'])
    cross = ((vert_pos_diff_from*vert_pos_diff_to) < 0)
    if cross and verbose:
        print(f'edge_a: {edge_a} {z_a}. edge_b: {edge_b} {z_b}')
        print(f'Vertical position difference on first layer: {vert_pos_diff_from}')
        print(f'Vertical position difference on second layer: {vert_pos_diff_to}')
    return cross

def count_crossing_edges(dig, verbose=False):
    """Count the number of crossing edge pairs in a directed graph"""
    n_crossing = 0
    layers = get_layers(dig)
    for prev_lay, next_lay in zip(layers[:-1], layers[1:]):
        #print(f'Layer {prev_lay}')
        layer_nodes = get_layer_nodes(dig, prev_lay, key_name='vertical_position') #sorted([n for n in dig.nodes if dig.nodes[n]['layer']==prev_lay], key=lambda nod: dig.nodes[nod]['vertical_position'])
        next_layer_nodes = get_layer_nodes(dig, next_lay, key_name='vertical_position')
        layer_edges = [edge for edge in dig.edges if edge[0] in layer_nodes and edge[1] in next_layer_nodes]
        if verbose:
            print(f'Edges between layers {prev_lay} and {next_lay}: {layer_edges}')
        crossing_edges = [(edge_a, edge_b) for edge_a, edge_b in itertools.combinations(layer_edges, 2) if edges_cross(dig, edge_a, edge_b)]
        if verbose:
            print(f'Crossing edge pairs: {crossing_edges}')
        n_crossing += len(crossing_edges)
    return n_crossing

def apply_barycenter_crossing_reduction_heuristic(dig, from_the_left=True, verbose=False):
    """Apply crossing reduction technique
    
    Heuristic for crossing reduction:
    changing position of nodes in layer one layer l according to the barycenter of the connected nodes in the adjacent layer (l-1 or l+1)
    
    Args:
        from_the_left: a boolean, True if position in a layer is changed on the basis of previous layer (left layer). If so, we start from the leftmost layer
    """
    # 
    layers = get_layers(dig)
    if verbose:
        if from_the_left:
            print('From the left')
        else:
            print('From the right')
    if not from_the_left:
        # 
        layers = layers[::-1]

    for lay, other_lay in zip(layers[1:], layers[:-1]):
        if verbose:
            print(f'Changing vertical position in layer {lay} based on connected nodes in layer {other_lay}')
        layer_nodes = get_layer_nodes(dig, lay, key_name='vertical_position') #sorted([n for n in dig.nodes if dig.nodes[n]['layer']==prev_lay], key=lambda nod: dig.nodes[nod]['vertical_position'])
        other_layer_nodes = get_layer_nodes(dig, other_lay, key_name='vertical_position')
        if from_the_left:
            layer_edges = [edge for edge in dig.edges if edge[1] in layer_nodes and edge[0] in other_layer_nodes]
        else:
            layer_edges = [edge for edge in dig.edges if edge[0] in layer_nodes and edge[1] in other_layer_nodes]
        if verbose:
            print(layer_edges)
        relative_values = []
        for node in layer_nodes:
            if from_the_left:
                connected_nodes = [edge[0] for edge in layer_edges if edge[1] == node]
            else:
                connected_nodes = [edge[1] for edge in layer_edges if edge[0] == node]
            connected_values = [dig.nodes[cn]['vertical_position'] for cn in connected_nodes]
            relative_values.append(np.mean(connected_values))
        # add current vertical position with small weight to avoid equal values
        eps = 0.1 / len(layer_nodes)
        relative_values = np.array(relative_values) + eps * np.array([dig.nodes[node]['vertical_position'] for node in layer_nodes])
        #non_null_rel_values = 
        if verbose:
            print(relative_values)
            print('Relative values: {}'.format(dict(zip(layer_nodes, relative_values))))
            
        # sort values, that is those that are not null
        sorted_values = sorted(relative_values)
        #new_values = [sorted_values.index(x) for x in relative_values ]
        non_null_ids = [i_val for i_val, val in enumerate(relative_values) if ~np.isnan(val)]
        non_null_vals = [val for i_val, val in enumerate(relative_values) if ~np.isnan(val)]
        sorted_values = sorted(non_null_vals)
        new_values = []
        for i, val in enumerate(relative_values):
            if ~np.isnan(val):
                new_value = non_null_ids[sorted_values.index(val)]
            else:
                new_value = i
            new_values.append(new_value)
        new_val_dict = dict(zip(layer_nodes, new_values))
        print('New relative values: {}'.format(new_val_dict))
        for i_node, node in enumerate(layer_nodes):
            dig.nodes[node]['vertical_position'] = new_values[i_node]


def sweep_barycenter_crossing_reduction(dig, n_sweeps=2, verbose=False):
    """Apply barycenter crossing reduction a number of times in each direction
    
    Args:
        dig
        n_sweeps: an int determining the number of times the crossing reduction heuristic is applied in each direction, typically 2
    """
    n_cross = count_crossing_edges(dig)
    print(f'Edge crossing before: {n_cross}')
    for i_sweep in range(n_sweeps):
        try:
            apply_barycenter_crossing_reduction_heuristic(dig, from_the_left=(i_sweep%2==0), verbose=verbose)
            n_cross = count_crossing_edges(dig)
            print(f'Edge crossing after: {n_cross}')
        except:
            print('Failed applying barycenter edge crossing reduction!')
            draw.plot_node_positions(dig, pos_var='vertical_position')
            apply_barycenter_crossing_reduction_heuristic(dig, from_the_left=(i_sweep%2==0), verbose=verbose)

def assign_absolute_vertical_position(dig):
    """Assign vertical position for each node based on relative vertical position"""
    
    layers = get_layers(dig)
    
    layer_node_dict = {lay: get_layer_nodes(dig, lay, key_name='vertical_position') for lay in layers}
    
    for lay in layers:
        y = 0
        layer_nodes = layer_node_dict[lay]
        for i_ln, n in enumerate(layer_nodes):
            if i_ln == 0:
                y = dig.nodes[n]['max_value']/2
            else:
                prev_n = layer_nodes[i_ln-1]
                y += (dig.nodes[prev_n]['max_value'] + dig.nodes[n]['max_value'])
            dig.nodes[n]['y'] = y

def process_directed_graph(dig, method='barycenter_heuristic'):
    """Process directed graph to prepare Sankey diagram"""
    assign_layers(dig)
    add_dummy_nodes(dig)
    
    determine_relative_vertical_position(dig)
    if method=='barycenter_heuristic':
        sweep_barycenter_crossing_reduction(dig)
        assign_absolute_vertical_position(dig)
    elif method=='lp':
        sweep_barycenter_crossing_reduction(dig)
        optim.optimize_absolute_vertical_position_lp(dig)
    elif method=='milp':
        optim.optimize_vertical_position_milp(dig)
    else:
        raise ValueError('Unknown method')