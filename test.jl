using Random
using BenchmarkTools
using LinearAlgebra
using PyCall
using Pkg
using StatsBase 

#println(PyCall.python) 
#Pkg.build("PyCall")

using Random

function generate_symmetric_matrix_no_diag(node_num, neighbor_num)
    if neighbor_num > node_num - 1
        error("N cannot be greater than n - 1 for a zero diagonal")
    end

    matrix = zeros(Int, node_num, node_num)

    for i in 1:node_num
        positions = setdiff(1:node_num, i)
        print(positions)
        exit()
        selected_positions = sample(positions, neighbor_num, replace=false)

        for pos in selected_positions
            matrix[i, pos] = 1
            matrix[pos, i] = 1  # Ensuring symmetry
        end
    end

    return matrix
end

function print_matrix(matrix)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            print("$(matrix[i, j]) ")
        end
        println()
    end
end

mat=generate_symmetric_matrix_no_diag(10,2)
print_matrix(mat)
