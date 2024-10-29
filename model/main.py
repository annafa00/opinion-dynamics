import networkx as nx
import random
from collections import Counter
from networkx.algorithms.community import louvain_communities
from constants import OPINION_RANGE, TRUST_LEVEL, PATH_TO_NETWORKS, NUM_STEPS, GLOBAL_PERIOD, GLOBAL_INFLUENCE, GLOBAL_TRUST_LEVEL
from plots import plot_similarity_heatmaps, plot_opinions_evolution

def initialize_opinions(communities):
    opinions = {}
    for nodes in communities:
        base_opinion = random.uniform(OPINION_RANGE[0], OPINION_RANGE[1])
        for node in nodes:
            opinions[node] = base_opinion + 0.2 * (random.uniform(OPINION_RANGE[0], OPINION_RANGE[1]) - base_opinion)

    return opinions


def initialize_trust_levels(graph):
    trust_levels = {}
    for edge in graph.edges():
        trust_levels[edge] = TRUST_LEVEL

    return trust_levels

def local_interaction_update(graph, opinions, trust_levels, node):
    neighbors = graph.neighbors(node)
    
    total_trust = sum(TRUST_LEVEL for n in neighbors)
    new_opinion = opinions[node] + (sum([trust_levels[(node, neighbor)] * (opinions[neighbor] - opinions[node]) for neighbor in neighbors]) / total_trust)
    opinions[node] = new_opinion
    return opinions

def global_interaction_update(graph, opinions):
    for node in graph.nodes():
        opinions[node] += GLOBAL_TRUST_LEVEL * (GLOBAL_INFLUENCE - opinions[node])
    return opinions

def all_values_equal(my_dict):
    return all(value == next(iter(my_dict.values())) for value in my_dict.values())

def continue_simulation(opinions, step):
    if step == NUM_STEPS:
        return False
    elif all_values_equal(opinions):
        return False
    else:
        return True
    
def length_of_equal_values(my_dict):
    # Count occurrences of each value
    value_counts = Counter(my_dict.values())
    
    # Find the maximum occurrence count
    max_occurrence = max(value_counts.values())
    
    # Count how many values have this maximum occurrence
    equal_values_count = sum(1 for count in value_counts.values() if count == max_occurrence)
    
    return equal_values_count

def main():
    network_path = f"{PATH_TO_NETWORKS}/synthetic_network_N_300_blocks_5_prr_0.84_prs_0.02.net"

    network = nx.Graph(nx.read_pajek(network_path))

    # Find communities
    communities = louvain_communities(network)

    # Initialize opinions and trust levels
    opinions = initialize_opinions(communities)
    trust_levels = initialize_trust_levels(network)

    step = 0
    nodes_opinions = {}
    while continue_simulation(opinions, step):
        step += 1

        # Local interaction
        node = random.choice(list(network.nodes()))
        opinions = local_interaction_update(network, opinions, trust_levels, node)

        # Global interaction
        if step % GLOBAL_PERIOD == 0:
            opinions = global_interaction_update(network, opinions)

        # Save nodes opinion
        nodes_opinions[step] = opinions

    # Analyse
    equal_opinions = {}
    for step, opinions in nodes_opinions.items():
        equal_opinions[step] = length_of_equal_values(opinions)
    
    plot_opinions_evolution(equal_opinions)
    plot_similarity_heatmaps(nodes_opinions)


if __name__ == '__main__':
    main()