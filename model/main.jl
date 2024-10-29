include("./constants.jl")
using PyCall
pyimport("pip").main(["install", "networkx"])
pyimport("pip").main(["install", "python-louvain"])
const nx = pyimport("networkx")
const community = pyimport("community")
using GraphPlot
using GraphIO
using Graphs
using Random

function initialize_opinions(graph, communities)
        n = Graphs.nv(graph)
    opinions = zeros(n)
    for community in unique(communities)
        nodes = findall(x -> x == community, communities)
        base_opinion = Random.rand(OPINION_RANGE)
        for node in nodes
            opinions[node] = base_opinion + 0.1 * Random.randn()
        end
    end
    return opinions
end

function initialize_trust_levels(graph)
    n = Graphs.nv(graph)
    trust_levels = fill(TRUST_LEVEL, n)
    return trust_levels
end

function local_interaction_update!(graph, opinions, trust_levels, node)
    neighbors = Graphs.neighbors(graph, node)
    if length(neighbors) == 0
        return
    end
    
    total_trust = sum(trust_levels[neighbors])
    new_opinion = sum(trust_levels[neighbors] .* opinions[neighbors]) / total_trust
    opinions[node] .= new_opinion
end

function global_influence_update!(opinions, trust_levels)
    global_opinion = GLOBAL_INFLUENCE
    opinions .+= trust_levels .* (global_opinion .- opinions)
end

function load_network(file_path)
    graph = Graph()  # Initialize an empty graph
    
    open(file_path, "r") do file
        for line in eachline(file)
            if startswith(line, "*edges")
                break
            end
        end

        for line in eachline(file)
            src, dst, _ = split(line)
            Graphs.add_edge!(graph, parse(Int, src), parse(Int, dst))
        end
    end

    return graph
end

#-----------------------------------------------------------------------------------------------------------------------------------
function main()
    network = load_network("$PATH_TO_NETWORKS/synthetic_network_N_300_blocks_5_prr_0.84_prs_0.02.net")

    # Convert the graph to a NetworkX graph
    nx_graph = nx.Graph(nx.read_pajek("$PATH_TO_NETWORKS/synthetic_network_N_300_blocks_5_prr_0.84_prs_0.02.net"))

    # Detect communities using the Louvain method
    partition = nx.algorithms.community.louvain_communities(nx_graph)

    # Convert partition to a Julia dictionary and ensure 1-based indexing
    communities = Dict{Int, Int}()
    for (node, comm) in partition
        communities[node + 1] = comm
    end

    # Plot the network with community colors
    community_vector = [communities[i] for i in 1:Graphs.nv(network)]
    gplot(network, nodefillc=community_vector, nodelabel=1:nv(network), nodelabeldist=1.5)

    # Initialize opinions and trust levels
    opinions = initialize_opinions(network, community_vector)
    trust_levels = initialize_trust_levels(network)
    
    for step in 1:NUM_STEPS
        # Local interactions
        node = rand(1:Graphs.nv(network))
        local_interaction_update!(network, opinions, trust_levels, node)

        # Global influence (periodically)
        if step % GLOBAL_PERIOD == 0
            global_influence_update!(opinions, trust_levels)
        end
    end

    println("Final opinions: ", opinions)
end

main()