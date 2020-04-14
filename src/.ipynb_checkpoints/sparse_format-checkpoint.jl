using SparseArrays

# Define the type of the sparse factorized graph
if isdefined(Base, :Factorized) == false
  #typealias Factorized{T} Dict{Int,SparseMatrixCSC{T,Int}}
  Factorized{T} = Dict{Int, SparseMatrixCSC{T,Int}}
end

# Define the type of the count vectors
if isdefined(Base, :Count) == false
  #typealias Count Dict{Int,Array{Int,1}}
  const Count = Dict{Int, Array{Int,1}}
end

# Define the type of the affinity vectors
if isdefined(Base, :Affinity) == false
  #typealias Affinity Dict{Int,Array{Float64,1}}
  const Affinity = Dict{Int, Array{Float64,1}}
end

"""
  Create a dict to store the sparse factorization of the graph

  Prameters
    n: (int) Size of the graph
    K: (int) number of features

  Returns:
    factorized: (dict) factorized[k] is an empty sparse matrix of size nxn
"""
function init_factorized(n::Int,
                        K::Int)
  factorized = Factorized{Int}()
  for k in 1:K
    factorized[k] = sparse([1],[1],[0],n,n)
  end
  return factorized
end

"""
  Construct the complete adjacency matrix from factorization

  Prameters
    Z: (factorized) Factorized graph

  Returns:
    complete_Z: (nxn Sparse array) Complete adjacency matrix
"""
function complete_graph(Z::Factorized{Int})
  return sum(values(Z))
end


"""
  Find the non zero index corresponding to the element (i,j)
  in the sparse matrix A

  Paramters:
    A: (Sparse matrix) Sparse matrix with integer valued entries
    i, j: (Int) (i,j) is the entry of interest
"""
function get_nz_index(A::SparseMatrixCSC{Int,Int},
                      i::Int,
                      j::Int)
  if !(1 <= i <= A.m && 1 <= j <= A.n)
     throw(BoundsError())
  end
  r1 = Int(A.colptr[j])
  r2 = Int(A.colptr[j+1]-1)
  if (r1 > r2)
    error(string("No corresponding entry ",i," ",j))
  end
  r1 += searchsortedfirst(A.rowval[r1:r2], i ) - 1
end
