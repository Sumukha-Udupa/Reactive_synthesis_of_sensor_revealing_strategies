
class Graph(object):

    def __init__(self, gdict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if gdict == None:
            gdict = {}
        self.gdict = gdict
        self.node_dict = {} # this is not used for general graphs.

    def edges(self, vertice):
        """ returns a list of all the edges of a vertice"""
        return self.gdict[vertice]

    def add_node(self, node, states_in_node=None):
        # each node is associated with a name "node" and contains a set of states "states_in_node"
        self.node_dict[node] = states_in_node
        if node not in self.gdict:
            self.gdict[node] = []


    def add_edge(self, edge):
        # each edge is a tuple object including the starting and ending nodes of the edge.
        node1, node2 = edge
        if node1 in self.gdict:
            self.gdict[node1].add(node2)
        else:
            self.gdict[node1] = [node2]  # initialize the edge set starting from x.

    def get_desc(self, node):
        """obtain the descendent of a node in the graph"""
