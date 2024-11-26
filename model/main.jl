include("./constants.jl")
using PyCall
@pyimport networkx as nx
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib.cm as cm
@pyimport pyvis.network as pyvis
using Random
using Statistics
using LinearAlgebra
#-----------------------------------------------------------------------------------------------------------------------------------
function plot_network(graph, communities, opinions, name)
    G = pyvis.Network(height="1000px", width="100%")
    pos = nx.spring_layout(graph)
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta", "yellow", "black"]
    j = 1
    for comm in communities
        for i in comm
            G.add_node(i, x=pos[i][1]*10000, y=pos[i][2]*10000, size=150, physics=false, color=colors[j], label=opinions[i])
        end
        j += 1
    end
    for i in graph.edges
        G.add_edge(i[1], i[2], physics=false, width=0.5, color="gray")     
    end
    G.save_graph("$PATH_TO_PLOTS/$name.html")
end
#-----------------------------------------------------------------------------------------------------------------------------------
function compute_general_variance(opinion_history, name)
    steps = size(opinion_history, 1)
    general_variances = [var(opinion_history[t, :]) for t in 1:steps]

    plt.figure(figsize=(10, 6))
    plt.plot(1:steps, general_variances, label="Network Variance", color="midnightblue")
    plt.xlabel("Time Step")
    plt.ylabel("Network Variance")
    plt.legend()
    plt.savefig("$PATH_TO_PLOTS/general_variance_$name.png")
end
#-----------------------------------------------------------------------------------------------------------------------------------
function compute_convergence_time(opinion_history, communities, name)
    steps = size(opinion_history, 1)
    community_variances = []

    for comm in communities
        variances = []
        for t in 1:steps
            opinions_comm = [opinion_history[t, parse(Int, node)] for node in comm]
            push!(variances, var(opinions_comm))
        end
        push!(community_variances, variances)
    end

    # Plot community variances over time
    plt.figure(figsize=(10, 6))
    for i in 1:length(communities)
        plt.plot(1:steps, community_variances[i], label="Community $i", linestyle="--", color=cm.Blues(i/ length(communities)))
    end
    plt.xlabel("Time Step")
    plt.ylabel("Community Variance")
    plt.legend()
    plt.savefig("$PATH_TO_PLOTS/community_variances_$name.png")
end
#-----------------------------------------------------------------------------------------------------------------------------------
function plot_opinion_evolution(opinion_history, name)
    steps = size(opinion_history, 1)
    avg_opinions = [mean(opinion_history[t, :]) for t in 1:steps]

    plt.figure(figsize=(10, 6))
    plt.plot(1:steps, avg_opinions, label="Network Average Opinion", color="midnightblue")
    plt.xlabel("Time Step")
    plt.ylabel("Average Opinion")
    plt.legend()
    plt.savefig("$PATH_TO_PLOTS/opinion_evolution_$name.png")
end
#-----------------------------------------------------------------------------------------------------------------------------------
function plot_community_trends(opinion_history, communities, name)
    steps = size(opinion_history, 1)
    community_avg = [mean([opinion_history[t, parse(Int, node)] for node in comm]) for comm in communities, t in 1:steps]

    plt.figure(figsize=(10, 6))
    for i in 1:length(communities)
        plt.plot(1:steps, community_avg[i, :], label="Community $i", linestyle="--", color=cm.Blues(i/ length(communities)))
    end
    plt.xlabel("Time Step")
    plt.ylabel("Community Average Opinion")
    plt.legend()
    plt.savefig("$PATH_TO_PLOTS/community_trends_$name.png")
end
#-----------------------------------------------------------------------------------------------------------------------------------
function initialize_opinions(communities)
    opinions = Dict{String, Float64}()
    for community in unique(communities)
        majority_opinion = rand(OPINION_VALUE)
        for node in community
            if rand() > 0.3
                opinions[node] = majority_opinion
            else
                opinions[node] = rand(OPINION_VALUE)
            end
        end
    end
    return opinions
end
#-----------------------------------------------------------------------------------------------------------------------------------
function initialize_trust_levels(graph)
    positions = nx.spring_layout(graph)
    maximum_distance = maximum([norm(positions[i] - positions[j]) for i in graph.nodes(), j in graph.nodes()]) # or radious

    # Pre-allocate trust levels matrix
    trust_levels = Dict(node => Dict() for node in graph.nodes())

    for node in graph.nodes()
        neighbors = [n for n in graph.neighbors(node)]
        for neighbor in neighbors
            normalized_distance = norm(positions[node] - positions[neighbor]) / maximum_distance
            trust_levels[node][neighbor] = normalized_distance
        end
    end   
    return trust_levels
end
#-----------------------------------------------------------------------------------------------------------------------------------
function MVM(communities, opinions, node)
    Pi = 0
    community_opinion = 0.0

    # Compute community opinion
    for comm in communities
        if node in comm
            community_opinion = sum([opinions[l] for l in comm if l != node]) / (length(comm) - 1)
            break
        end
    end

    if community_opinion == 0.0
        error("Node $node not found in any community")
    end

    # Compute the opinion distance between the node and the community
    opinion_distance = abs(opinions[node] - community_opinion)


    # Probability of node i adopting the state of a neighbor
    Pi = 1 / (1 + exp((1 - opinion_distance) / λ))
    return Pi
end
#-----------------------------------------------------------------------------------------------------------------------------------
function local_interaction_update!(graph, opinions, node, communities)
    # Local Interaction based on Multiscale Voter Model

    neighbors = [n for n in graph.neighbors(node)]
    
    neighbor = rand(neighbors)
    Pi = MVM(communities, opinions, node)

    #trust_level = trust_levels[node][neighbor]
    if Pi > TRUST_LEVEL
        opinions[node] = opinions[neighbor]
    end
end
#-----------------------------------------------------------------------------------------------------------------------------------
function global_influence_update!(graph, opinions, communities)
    global_opinion = GLOBAL_INFLUENCE

    for node in graph.nodes()
        Pi = MVM(communities, opinions, node)
        if Pi > TRUST_LEVEL
            opinions[node] = global_opinion
        end
    end
end
#-----------------------------------------------------------------------------------------------------------------------------------
function different_opinions(opinions)
    return length(unique(opinions)) > 1
end
#-----------------------------------------------------------------------------------------------------------------------------------
function main()
    name = "synthetic_network_N_300_blocks_5_prr_0.12_prs_0.02"

    # Convert the graph to a NetworkX graph
    nx_graph = nx.Graph(nx.read_pajek("$PATH_TO_NETWORKS/$name.net"))
    size = nx_graph.number_of_nodes()

    # Detect communities using the Louvain method
    communities = nx.algorithms.community.louvain_communities(nx_graph)

    # Initialize opinions and trust levels
    opinions = initialize_opinions(communities)
    #trust_levels = initialize_trust_levels(nx_graph)

    # Plot the graph with the detected communities
    plot_network(nx_graph, communities, opinions, name)

    # Store opinions over time for analysis
    opinion_history = zeros(NUM_STEPS, size)

    for step in 1:NUM_STEPS
        if step != NUM_STEPS && different_opinions(opinions)
            # Local interactions
            node = rand(collect(nx_graph.nodes()))
            local_interaction_update!(nx_graph, opinions, node, communities)
    
            # Global influence (periodically)
            if step % GLOBAL_PERIOD == 0
                global_influence_update!(nx_graph, opinions, communities)
            end
    
            # Record opinions
            opinion_values = [opinions[string(i)] for i in 1:size]
            opinion_history[step, :] .= opinion_values
        else
            break
        end
    end

    # Compute and visualize convergence time
    compute_general_variance(opinion_history, name)
    compute_convergence_time(opinion_history, communities, name)

    # Plot evolution and trends
    plot_opinion_evolution(opinion_history, name)
    plot_community_trends(opinion_history, communities, name)

    println("Final opinions: ", opinions)
end
#-----------------------------------------------------------------------------------------------------------------------------------
main()