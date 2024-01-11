using Random
using BenchmarkTools
function random_setup(nodes, N, Neighbourhood_size)
    # Create a random graph and return the adjacency matrix
    graph = rand(0:1, nodes, nodes)
    
    #graph = triu(graph)
    graph=triu!(trues(size(graph)))#[graph[row,col] for col in 1:n for row in 1:col]
    graph = graph + transpose(graph)
    
    for i in 1:nodes
        graph[i, i] = 0
        while sum(graph[i, :]) > Neighbourhood_size
            a = rand(1:nodes)
            graph[i, a] = 0
            graph[a, i] = 0
        end
    end
    
    # Assign random fitness
    fitness = rand(0:1, nodes, N)

    return graph, fitness
end
@btime random_setup(15, 5, 4)
#now call the function
graph, fitness = random_setup(15, 5, 4)

#print graph
println(graph)

