"""Optimization-based layout functions"""

import itertools

# Import PuLP modeler functions
from pulp import *
from . import processing

def var_name(node_name):
    """Get a variable name valid for pulp
    
    Args:
        node_name: a string or other hashable type
        
    Returns:
        a modified node name without special characters
    """
    s = str(node_name)
    # illegalChars = "-+[] ->/"
    s = s.replace("-","(hyphen)")
    s = s.replace(",","(comma)")
    s = s.replace("+","(plus)")
    s = s.replace("[","(leftBracket)")
    s = s.replace("]","(rightBracket)")
    s = s.replace(" ","(space)")
    s = s.replace(">","(greaterThan)")
    s = s.replace("/","(slash)")
    return s

def add_bendiness_objective(prob, dig, node_pos_dict):
    """Add bendiness to the objective function"""
    bendiness_vars = []
    for n_from, n_to in dig.edges:
        edge_bendiness = LpVariable('bendiness_'+var_name(n_from)+'_'+var_name(n_to))
        bendiness_vars.append(edge_bendiness)
        prob += (node_pos_dict[n_from] - node_pos_dict[n_to]) <= edge_bendiness, f"y_from - y_to <= Edge bendiness {n_from} {n_to}"
        prob += (node_pos_dict[n_to] - node_pos_dict[n_from]) <= edge_bendiness, f"y_to - y_from <= Edge bendiness {n_from} {n_to}"

    bendiness_sum = LpVariable('sumBendiness')
    prob += bendiness_sum == lpSum(bend_var for bend_var in bendiness_vars)
    return bendiness_sum

def add_distance_to_center(prob, node_pos_dict, n):
    """Add a variable corresponding to the absolute distance to 0"""
    abs_node_pos = LpVariable('abs('+var_name(n)+')')
    prob += node_pos_dict[n] <= abs_node_pos, f"x <= |x| for {n}"
    prob += -node_pos_dict[n] <= abs_node_pos, f"-x <= |x| for {n}"
    return abs_node_pos

def optimize_absolute_vertical_position_lp(dig):
    """Optimize absolute position of nodes, where relative position is already fixed
    
    The formulation minimizes the distances of nodes to center (y=0) and the vertical distances between nodes joined by an edge
    
    Args:
        dig
    """
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("SugiyamaLP", LpMinimize)

    # Variables
    layers = processing.get_layers(dig)

    layer_node_dict = {lay: processing.get_layer_nodes(dig, lay, key_name='vertical_position') for lay in layers}

    node_pos_dict = {}
    center_distances = []
    for lay in layers:
        y = 0
        layer_nodes = layer_node_dict[lay]
        #print(layer_nodes)

        for i_n, n in enumerate(layer_nodes):
            node_pos_dict[n] = LpVariable(var_name(n))
            abs_node_pos = add_distance_to_center(prob, node_pos_dict, n)
            center_distances.append(abs_node_pos)
            # The objective function is added to 'prob' first
            #prob += abs_node_pos, f"Minimize distance from center for {n}"
            #prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"

            if i_n > 0:
                prev_n = layer_nodes[i_n-1]
                prob += (node_pos_dict[prev_n] + dig.nodes[prev_n]['max_value'] <= node_pos_dict[n] - dig.nodes[n]['max_value']/2), f"Minimum distance between nodes {prev_n} {n}"
        #x1 = LpVariable("ChickenPercent", 0, None, LpInteger)
    #x2 = LpVariable("BeefPercent", 0)

    center_dist_sum = LpVariable('sumDistancesToCenter')
    prob += center_dist_sum == lpSum(cent_dist for cent_dist in center_distances)

    bendiness_sum = add_bendiness_objective(prob, dig, node_pos_dict)

    lambda_bendiness = 2.
    prob += center_dist_sum + lambda_bendiness*bendiness_sum, 'Minimize sum of distances to center plus bendiness'

    #print(node_pos_dict)
    # The problem is solved using PuLP's choice of Solver
    #prob.solve()
    prob.solve(PULP_CBC_CMD(msg=1))

    # Each of the variables is printed with it's resolved optimum value
    #for v in prob.variables():
    #    print(v.name, "=", v.varValue)


    for n, opt_var in node_pos_dict.items():
        dig.nodes[n]['y'] = opt_var.varValue
        
        
def optimize_vertical_position_milp(dig, verbose=False):
    """Optimize absolute and relative position of nodes
    
    The formulation minimizes the distances of nodes to center (y=0) and the vertical distances between nodes joined by an edge
    """
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("SugiyamaLP", LpMinimize)

    # Variables
    layers = processing.get_layers(dig)

    layer_node_dict = {lay: processing.get_layer_nodes(dig, lay) for lay in layers}

    node_pos_dict = {}
    node_rel_pos_dict = {}
    #crossing_dict = {}
    crossing_vars = []
    
    center_distances = []
        
        
    for lay in layers:
        y = 0
        layer_nodes = layer_node_dict[lay]
        #print(layer_nodes)

                
        for i_n, n in enumerate(layer_nodes):
            node_pos_dict[n] = LpVariable(var_name(n))
            abs_node_pos = add_distance_to_center(prob, node_pos_dict, n)
            center_distances.append(abs_node_pos)
            # The objective function is added to 'prob' first
            #prob += abs_node_pos, f"Minimize distance from center for {n}"
            #prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"

        for i_n1, i_n2 in itertools.permutations(range(len(layer_nodes)), 2):
            n1 = layer_nodes[i_n1]
            n2 = layer_nodes[i_n2]
            node_rel_pos_dict[(n1, n2)] = LpVariable(var_name(n1)+'_above_'+var_name(n2), cat='Binary')
            
            # if n1 is above n2, yn1 - yn2 should be above 0 and larger than some distance
            prob += (node_pos_dict[n1] - node_pos_dict[n2] - dig.nodes[n1]['max_value']/2 - dig.nodes[n2]['max_value']/2 - 0.5+ 1000*(1-node_rel_pos_dict[(n1, n2)]) >= 0)
            
            
        for i_n1, i_n2 in itertools.combinations(range(len(layer_nodes)), 2):
            n1 = layer_nodes[i_n1]
            n2 = layer_nodes[i_n2]
            prob += (node_rel_pos_dict[(n1, n2)] + node_rel_pos_dict[(n2, n1)] == 1) # n1 above n2 or n2 above n1
            
            
            #if i_n > 0:
            #    prev_n = layer_nodes[i_n-1]
            #    prob += (node_pos_dict[prev_n] + dig.nodes[prev_n]['max_value'] <= node_pos_dict[n] - dig.nodes[n]['max_value']/2), f"Minimum distance between nodes {prev_n} {n}"
        
    # crossings
    for prev_lay, next_lay in zip(layers[:-1], layers[1:]):
        layer_nodes = processing.get_layer_nodes(dig, prev_lay) #sorted([n for n in dig.nodes if dig.nodes[n]['layer']==prev_lay], key=lambda nod: dig.nodes[nod]['vertical_position'])
        next_layer_nodes = processing.get_layer_nodes(dig, next_lay)
        layer_edges = [edge for edge in dig.edges if edge[0] in layer_nodes and edge[1] in next_layer_nodes]
        for i_e1, i_e2 in itertools.combinations(range(len(layer_edges)), 2):
            n1a, n1b = layer_edges[i_e1]
            n2a, n2b = layer_edges[i_e2]
            
            if not (n1a == n2a) and not (n1b == n2b):
                #print((e1, e2))
                cross_var = LpVariable(var_name(n1a)+'_'+var_name(n1b)+'_crosses_'+var_name(n2a)+'_'+var_name(n2b), cat='Binary') # 1 if the two edges cross
                prob += (cross_var >= node_rel_pos_dict[(n1a, n2a)] + node_rel_pos_dict[(n2b, n1b)] -1)
                prob += (cross_var >= node_rel_pos_dict[(n2a, n1a)] + node_rel_pos_dict[(n1b, n2b)] -1)
                crossing_vars.append(cross_var)

    center_dist_sum = LpVariable('sumDistancesToCenter')
    prob += center_dist_sum == lpSum(cent_dist for cent_dist in center_distances)
    
    n_crossings = LpVariable('numberOfEdgeCrossings')
    prob += n_crossings == lpSum(cr for cr in crossing_vars)

    bendiness_sum = add_bendiness_objective(prob, dig, node_pos_dict)

    lambda_bendiness = 2.
    lambda_crossings = 20.
    prob += center_dist_sum + lambda_bendiness*bendiness_sum + lambda_crossings*n_crossings, 'Minimize sum of crossings plus distances to center plus bendiness'

    #print(node_pos_dict)
    # The problem is solved using PuLP's choice of Solver
    #prob.solve()
    prob.solve(PULP_CBC_CMD(msg=1))

    if verbose:
        # Each of the variables is printed with it's resolved optimum value
        for v in prob.variables():
            print(v.name, "=", v.varValue)



    for n, opt_var in node_pos_dict.items():
        dig.nodes[n]['y'] = opt_var.varValue
        dig.nodes[n]['vertical_position'] = 0

    for lay in layers:
        layer_nodes = layer_node_dict[lay]
        for i_n1, i_n2 in itertools.permutations(range(len(layer_nodes)), 2):
            n1 = layer_nodes[i_n1]
            n2 = layer_nodes[i_n2]
            dig.nodes[n1]['vertical_position'] += node_rel_pos_dict[(n1, n2)].varValue
            if verbose:
                print((n1, n2, node_rel_pos_dict[(n1, n2)].varValue))
        #dig.nodes[n]['vertical_position']
        
    return prob
