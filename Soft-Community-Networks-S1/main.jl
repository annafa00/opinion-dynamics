using Random
using Distributions
using LinearAlgebra

include("./constants/constants.jl")
#-------------------------------------------------------------------------------------------
function geometric_preferential_attachment()
    # Initialize angular positions
    θ = zeros(N)

    for i in 1:N
        # Sample i candidate angular positions
        φ = 2π * rand(i)

        # Compute attractiveness for each candidate position
        A = zeros(i)
        for l in 1:i
            for s in 1:(i-1)
                if abs(θ[s] - φ[l]) < 2 / (s^(1/(γ-1)) * i^((γ-2)/(γ-1)))
                    A[l] += 1
                end
            end
        end

        # Compute probabilities for each candidate position
        probabilities = (A .+ Λ) ./ sum(A .+ Λ)

        # Assign angular position to node i
        θ[i] = φ[findfirst(cumsum(probabilities) .>= rand())]
    end
    return θ
end
#-------------------------------------------------------------------------------------------
function assign_hidden_degrees(θ)
    # Generate a set of N target degrees from a power-law distribution with exponent γ. P(k) ~ k^(-γ)
    power_dist = Pareto(γ-1)
    ktar = sort(rand(power_dist, N), rev=true)
    μ = mean(ktar)
    println("μ = $(μ)")

    # Initialize hidden degrees
    κ = copy(ktar)

    function expected_degree(i)
        sum(1 / (1 + ((abs(θ[j] - θ[i]) * R)/(μ * κ[i] * κ[j]))^β) for j in 1:N if j != i)
    end

    # Initialize relative deviation
    ϵ = 2*η
    while ϵ > η
        print("ϵ = $(ϵ) > η = $(η). Reassigning hidden degrees...\n")
        for _ in 1:N
            i = rand(1:N)
            k̄i = expected_degree(i)
            δ = rand() * 0.1
            κ[i] = abs(κ[i] + (ktar[i] - k̄i) * δ)
        end
        
        # Compute all relative deviations
        ϵ = maximum(abs(ktar[i] - expected_degree(i)) / ktar[i] for i in 1:N)
    end

    return κ, ktar, μ
end
#-------------------------------------------------------------------------------------------
function generate_network(θ, k, μ)
    # Initialize adjacency matrix
    A = zeros(N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                # Calculate the connection probability
                pij = 1 / (1 + ((abs(θ[j] - θ[i]) * R) / (μ * k[i] * k[j]))^β)
                # Connect nodes i and j with probability pij
                if rand() < pij
                    A[i, j] = 1
                    A[j, i] = 1 
                end
            end
        end
    end

    return A
end
#-------------------------------------------------------------------------------------------
function save_adjacency_matrix(A, filename)
    open("./networks/$(filename).csv", "w") do f
        for i in 1:N
            for j in 1:N
                write(f, string(A[i, j], ","))
            end
            write(f, "\n")
        end
    end
end
#-------------------------------------------------------------------------------------------
function save_positions(θ, ktar, μ, filename)

    # Calculate radial coordinates
    ktar_min = minimum(ktar)
    R = 2 * log(N / (π * μ * (ktar_min)^2))
    r = R .- 2 * log.(ktar ./ ktar_min)

    # Convert polar coordinates to Cartesian coordinates
    x = r .* cos.(θ)
    y = r .* sin.(θ)

    # Save the positions
    open("./networks/$(filename)_positions.csv", "w") do f
        for i in 1:N
            write(f, string(i, x[i], ",", y[i], ",", θ[i], ",", r[i], ",", ktar[i], "\n"))
        end
    end
end
#-------------------------------------------------------------------------------------------
function create_folder(path)
    if !isdir(path)
        mkdir(path)
    end
end
#-------------------------------------------------------------------------------------------
function main()
    # Create folder if it does not exist
    create_folder("./networks")

    filename = "network_γ$(γ)_Λ$(Λ)_N$(N)"
    println("Generating network with S1 model... filename: $(filename)")

    # Get the angular positions of the nodes
    println("Generating angular positions...")
    θ = geometric_preferential_attachment()

    # Assign the hidden degrees to the nodes
    println("Assigning hidden degrees...")
    k, ktar, μ = assign_hidden_degrees(θ)

    # Generate the network with S1 model
    println("Generating network...")
    A = generate_network(θ, k, μ)

    # Save the adjacency matrix
    println("Saving adjacency matrix...")
    save_adjacency_matrix(A, filename)

    # Plot the network
    println("Saving nodes positions...")
    save_positions(θ, ktar, μ, filename)
    
    println("Done!")
end

main()