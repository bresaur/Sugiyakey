import os
import itertools

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

def plot_node_positions(dig, pos_var='y'):
    """Plot node positions
    
    Args:
        dig
        pos_var: a string determining which node attribute to use as y-coordinate
    """
    
    plt.figure()
    color_cycle = ['r', 'g', 'b', 'y']

    node_colors = [color_cycle[dig.nodes[node]['layer']] for node in dig.nodes]
    node_positions = {n: (dig.nodes[n]['layer'], dig.nodes[n][pos_var]) for n in dig.nodes}
    nx.draw(dig, node_color=node_colors, pos=node_positions, with_labels=True)

    for n in dig.nodes:
        x = dig.nodes[n]['layer']
        y = dig.nodes[n][pos_var]
        dy = dig.nodes[n]['max_value']
        plt.plot([x, x], [y-dy/2, y+dy/2], color='gray')

def cubic_spline_link(x_from_to=None, y_from_to=None, n=100):
    """Draw a cubic spline with 0 slope from a point to another point
    
    Args:
        x_from_to: a numpy array with two x coordinates
        y_from_to: a numpy array with two x coordinates
        
    Returns:
        two numpy arrays
    """
    if x_from_to is None:
        x_from_to = np.array([0, 1])
    if y_from_to is None:
        y_from_to = np.array([0, 1])

    t = np.linspace(0, 1, n)
    y = y_from_to[0] + (y_from_to[1] - y_from_to[0])* (t**2 * (3 -2*t))
    x = x_from_to[0] + (x_from_to[1] - x_from_to[0])*t
    # (x_from_to[1] - x_from_to[0])
    return x, y

def get_node_position(dig, n):
    """Get node position"""
    x = dig.nodes[n]['layer']
    dy = dig.nodes[n]['max_value']
    y = dig.nodes[n]['y'] - dy/2
    return x, y, dy
    
def plot_sankey_node(dig, ax, n, dx_node=0.2, dx_arrow=0.1, edge_edge_kw=None, edge_fill_kw=None):
    """Plot lines/figure component of a Sankey diagram for one node
    
    Args:
        dig
        n: node identified, typically a str
    """
    x, y, dy = get_node_position(dig, n)

    if 'dummy' not in str(n).lower():
        ax.text(x, dig.nodes[n]['y'], n)

    if dx_node>0:
        # plot "node", if any width
        x_below = [x-dx_node, x+dx_node]
        y_below = [y, y]

        y_above = [y+dig.nodes[n]['max_value'], y+dig.nodes[n]['max_value']]

        out_arrow = dig.out_degree(n) == 0 and dx_arrow>0
        in_arrow = dig.in_degree(n) == 0 and dx_arrow>0

        if out_arrow:
            x_arrow_out = [x+dx_node, x+dx_node+dx_arrow, x+dx_node]
            y_arrow_out = [y, y+dy/2, y+dy]
            ax.plot(x_below + x_arrow_out + x_below[::-1],
                     y_below + y_arrow_out + y_above[::-1], **edge_edge_kw)
        else:
            x_arrow_out = []
            y_arrow_out = []

        if in_arrow:
            x_arrow_in = [x-dx_node, x-dx_node+dx_arrow, x-dx_node]
            y_arrow_in = [y+dy, y+dy/2, y]
            ax.plot(x_below[::-1] + x_arrow_in + x_below,
                     y_above[::-1] + y_arrow_in + y_below, **edge_edge_kw)
        else:
            x_arrow_in = []
            y_arrow_in = []

        if not out_arrow and not in_arrow:
            ax.plot(x_below, y_below, **edge_edge_kw)
            ax.plot(x_below, y_above, **edge_edge_kw)

        x_fill = x_below + x_arrow_out + x_below[::-1] + x_arrow_in
        y_fill = y_below + y_arrow_out + y_above[::-1] + y_arrow_in
        ax.fill(x_fill, y_fill, **edge_fill_kw)
    
def plot_sankey_link(dig, ax, n, n_next, y, dx_node, link_geometry='cubic_spline', edge_edge_kw=None, edge_fill_kw=None):
    """Plot link between two nodes as part of a Sankey diagram"""
    x, y_old, dy = get_node_position(dig, n)
    
    edge_val = dig.edges[n, n_next]['value']
    x_next = dig.nodes[n_next]['layer']
    y_next = dig.nodes[n_next]['y_in']

    x_from_to = np.array([x+dx_node, x_next-dx_node])
    y_from_to = np.array([y, y_next])
    if link_geometry=='line':
        x_below = x_from_to
        y_below = y_from_to
        y_above = y_from_to + edge_val
    elif link_geometry=='cubic_spline':
        x_below, y_below = cubic_spline_link(x_from_to, y_from_to)
        y_above = y_below + edge_val



    ax.plot(x_below, y_below, **edge_edge_kw)
    ax.plot(x_below, y_above, **edge_edge_kw)

    x_fill = np.concatenate([x_below, x_below[::-1]])
    y_fill = np.concatenate([y_below, y_above[::-1]])
    ax.fill(x_fill, y_fill, **edge_fill_kw)

    y+= edge_val
    dig.nodes[n_next]['y_in'] += edge_val
    return y

    
def draw_sankey(dig, ax=None, link_geometry='cubic_spline', 
                dx_node=0.2, dx_arrow=0.1):
    """The actual drawing
    
    Args:
        dig
        ax: a matplotlib axis
    
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    #nx.draw(dig, node_color=node_colors, pos=node_positions, with_labels=True)

    edge_edge_kw = {'color': 'gray', 'ls': '-'}
    edge_fill_kw = {'color': 'deeppink', 'alpha': 0.2, 'lw': 0}
    

    for n in dig.nodes:
        dig.nodes[n]['y_in'] = dig.nodes[n]['y'] - dig.nodes[n]['in_value']/2
        

    for n in sorted(dig.nodes, key=lambda n: (-dig.nodes[n]['layer'], dig.nodes[n]['vertical_position'])):
        
        plot_sankey_node(dig, ax, n, dx_node=dx_node, dx_arrow=dx_arrow, edge_edge_kw=edge_edge_kw, edge_fill_kw=edge_fill_kw)
        x, y, dy = get_node_position(dig, n)

        for n_next in sorted(dig.successors(n), key=lambda n: dig.nodes[n]['vertical_position']):
            y = plot_sankey_link(dig, ax, n, n_next, y, dx_node=dx_node, link_geometry=link_geometry, edge_edge_kw=edge_edge_kw, edge_fill_kw=edge_fill_kw)