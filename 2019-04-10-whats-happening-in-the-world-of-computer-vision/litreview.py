import json
import itertools
from operator import itemgetter
from time import time

import arxivpy
import networkx as nx
from networkx.algorithms import community


def fetch_recent_cv_papers(filename, num=65536):
    papers = arxivpy.query(search_query=['cs.CV'], start_index=0,
                           max_index=num-1, results_per_iteration=128,
                           wait_time=2.0, sort_by='submittedDate')

    # Normalise articles
    for paper in papers:
        # Dates as strings
        paper['publish_date'] = paper['publish_date'].isoformat()
        paper['update_date'] = paper['update_date'].isoformat()

    with open(filename, 'w') as f:
        f.write(json.dumps(papers))


def load_papers(filename, earliest_date=None):
    papers = json.loads(open(filename, 'r').read())
    if earliest_date:
        papers = [p for p in papers if p["publish_date"] > earliest_date]

    # Normalise
    for paper in papers:
        # Authors as a list
        if type(paper['authors']) is str:
            paper['authors'] = paper['authors'].split(", ")

        # Terms as a list
        if type(paper['terms']) is str:
            paper['terms'] = paper['terms'].split("|")

        # Remove triple spaces from line breaks in titles
        paper['title'] = paper['title'].replace("   ", " ")

    return papers


def get_titles_to_papers(papers):
    t2p = {}
    for paper in papers:
        t2p[paper['title']] = paper
    return t2p


def get_author_to_titles(papers):
    a2t = {}
    for paper in papers:
        for author in paper['authors']:
            a2t[author] = []

    for paper in papers:
        for author in paper['authors']:
            a2t[author].append(paper['title'])

    return a2t


def get_heaviest_edge(g):
    u, v, w = max(g.edges(data='weight'), key=itemgetter(2))
    return u, v


def get_collaboration_graph(papers, min_node_weight=None,
                            min_edge_weight=None, add_communities=False):
    """ Create a collaboration graph from parsed arxiv papers

    The graph edges are weighted inversely to the number of authors on
    each paper. Self-edges are allowed to correctly attribute papers to
    solo authors.

    :param list[dict[str,Any]] papers: a dictionary of metadata for each
        paper, containing at least "authors", a list of strings corresponding
        to the name of each author
    """
    # Get edge weights
    node_weights = {}
    edge_weights = {}
    for paper in papers:
        authors = paper['authors']
        n_authors = len(authors)
        weight = 1.0/n_authors
        for i in range(n_authors):
            # Add node weights
            if authors[i] in node_weights.keys():
                node_weights[authors[i]] += weight
            else:
                node_weights[authors[i]] = weight

            # Add edge weights
            for j in range(i, n_authors):
                edge = [authors[i], authors[j]]
                edge.sort()
                edge = tuple(edge)
                if edge in edge_weights.keys():
                    edge_weights[edge] += weight
                else:
                    edge_weights[edge] = weight

    # Drop light nodes
    if min_node_weight:
        node_weights = {n: w for n, w in node_weights.items()
                        if w > min_node_weight}

    # Add nodes
    g = nx.Graph()
    for node, weight in node_weights.items():
        g.add_node(node, weight=weight)

    # Drop light edges
    if min_edge_weight:
        edge_weights = {e: w for e, w in edge_weights.items()
                        if w > min_edge_weight}

    # Add remaining edges
    for edge, weight in edge_weights.items():
        g.add_edge(edge[0], edge[1], weight=weight)
    g.add_weighted_edges_from([(edge[0], edge[1], weight)
                              for edge, weight in edge_weights.items()])

    # Add communities
    if add_communities:
        n_community_levels = 1
        start = time()
        comms_gen = community.girvan_newman(g, most_valuable_edge=get_heaviest_edge)
        for i_level, comms in enumerate(itertools.islice(comms_gen, n_community_levels)):
            comms = sorted(comms, key=len, reverse=True)
            node_comms = {}
            for i_comm, comm in enumerate(comms):
                for node in comm:
                    node_comms[node] = i_comm
            nx.set_node_attributes(g, node_comms, "comm_level_{}".format(i_level))
        print("{:1.2f} s to compute communities.".format(time() - start))

    # Return
    return g


def extract_subgraph(g, nodes):
    """ Extract a subgraph based on provided nodes

    The networkx Graph.subgraph implementation is inexplicably slow, as is
    the suggested implementation here: https://networkx.github.io/documentation/latest/reference/classes/generated/networkx.Graph.subgraph.html

    Instead, we implement a naive shallow copy followed by node removal, which
    is nearly instantaneous.
    """
    sg = g.copy()
    nodes_to_remove = set(g.nodes).difference(set(nodes))
    for node in nodes_to_remove:
        sg.remove_node(node)
    return sg


def extract_highest_weight_nodes(g, max_nodes):
    if max_nodes >= g.number_of_nodes():
        return g

    node_weights = nx.get_node_attributes(g, "weight")
    node_weights_sorted = sorted(node_weights.items(), key=lambda kv: kv[1],
                                 reverse=True)
    heaviest_nodes = [node for node, weight in node_weights_sorted[:max_nodes]]
    return extract_subgraph(g, heaviest_nodes)


def extract_attribute_match(g, attr_name, attr_val):
    node_vals = nx.get_node_attributes(g, attr_name)
    member_nodes = [node for node, val in node_vals.items()
                    if val == attr_val]
    return extract_subgraph(g, member_nodes)


def get_community_papers(g, author_to_titles, titles_to_papers):
    titles = set()
    for author in g.nodes():
        titles.update(author_to_titles[author])
    papers = [titles_to_papers[title] for title in list(titles)]
    papers = sorted(papers, key=lambda p: p['publish_date'], reverse=True)
    return papers
