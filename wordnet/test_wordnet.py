from nltk.corpus import wordnet as wn
import matplotlib
import networkx as nx


def travers(graph, start, node):
    graph.depth[node.name()] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name(), child.name())
        travers(graph, start, child)

def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    travers(G, start, start)
    return G

def graph_draw(graph):
    nx.draw(graph, node_size=[5 * graph.degree(n) for n in graph], node_color=[graph.depth[n] for n in graph],
    with_labels=True)
    matplotlib.pyplot.show()


#computer = wn.synset('book.n.02')
test_word = wn.synset('candle.n.02')
#graph = hyponym_graph(computer)
#print(computer.definition())
print(test_word.definition())
#graph_draw(graph)

'''
Identification d'un mot via NLTK Wordnet

"mot".type.synsets_nb.lemmas

'''