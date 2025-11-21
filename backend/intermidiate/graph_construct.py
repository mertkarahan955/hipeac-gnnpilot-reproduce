from utils import *
import networkx as nx
import matplotlib.pyplot as plt
import copy

# to do: duplicate output variables

# G: dependency graph of variables
G = nx.DiGraph()
# Go: dependency graph of operators
Go = nx.DiGraph()
num_nodes = 0
node_dict = {}
op_level_list = []

def register_new_node(var):
    # register a new variable node
    G.add_node(var)
    # G.add_node(num_nodes)
    node_dict[var] = 1
    # num_nodes += 1

def construct_var_graph(gnn_op_list):
    for op_id, gnn_op in enumerate(gnn_op_list):
        var_output = gnn_op.var_output
        if (not var_output.name in node_dict):
            register_new_node(var_output.name)

        for var_input in gnn_op.expr.var_input:
            if (not var_input.name in node_dict):
                register_new_node(var_input.name)
            if (not G.has_edge(var_input.name, var_output.name)):
                G.add_edge(var_input.name, var_output.name, weight=op_id)
                #print(G.edges(data=True))
            else:
                inter_warning("Duplicated edges!")
                # G.add_edge(var_input.name, var_output.name, weight=op_id + 1)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig("dep_var_graph.png")

def construct_op_graph(gnn_op_list):
    for edge in G.edges(data=True):
        kname = edge[2]['weight']
        if (not Go.has_node(kname)):
            Go.add_node(kname)
    
    for edge in G.edges(data=True):
        node = edge[2]['weight']
        src_op = gnn_op_list[node]
        for dst_edge in G.edges(data=True):
            dst_node = dst_edge[2]['weight']
            dst_op = gnn_op_list[dst_node]
            for var in dst_op.expr.var_input:
                if (var.name == src_op.var_output.name and not Go.has_edge(node, dst_node)):
                    Go.add_edge(node, dst_node, label=var.name)

    plt.cla()
    pos = nx.spring_layout(Go)
    nx.draw(Go, pos, with_labels = True)
    nx.draw_networkx_edge_labels(Go, pos)
    plt.savefig("dep_op_graph.png")


def construct_graph(gnn_op_list):
    construct_var_graph(gnn_op_list)
    construct_op_graph(gnn_op_list)

def topo_sort(gnn_op_list):
    while (len(G.edges) > 0):
        wavefront_node_list = []
        wavefront_edge_list = []
        for node in G.nodes:
            if (G.in_degree[node] == 0):
                wavefront_node_list.append(node)
                for edge in G.out_edges(node, data=True):
                    op_id = edge[2]['weight']
                    if (op_id not in wavefront_edge_list):
                        wavefront_edge_list.append(op_id)

        current_level_list = []
        for op_id in wavefront_edge_list:
            flag = True
            for var_input in gnn_op_list[op_id].expr.var_input:
                if (not var_input.name in wavefront_node_list):
                    flag = False
            if (flag):
                current_level_list.append(op_id)
                tmp_edge_list = copy.deepcopy(G.edges(data=True))
                for edge in tmp_edge_list:
                    if (edge[2]['weight'] == op_id):
                        G.remove_edge(edge[0], edge[1])
        
        for node in wavefront_node_list:
            if (G.out_degree[node] == 0):
                G.remove_node(node)
        op_level_list.append(current_level_list)

    return op_level_list
