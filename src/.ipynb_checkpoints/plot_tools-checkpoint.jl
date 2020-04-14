using PyPlot
using Random



function spy_sparse(spMatrix,directed=true)
  i_idx, j_idx = findnz(spMatrix)
  PyPlot.figure()
  PyPlot.scatter(i_idx,j_idx,s=1.,color="C0")
  if directed == false
    PyPlot.scatter(j_idx,i_idx,s=1.,color="C0")
  end
end

function spy_sparse_order(spMatrix,order_idx,markersize=1.,directed=true,file_name="clustering.png";save=true)
  n = maximum(size(spMatrix))
  if directed == false
    spMatrix = dropzeros(spMatrix+transpose(spMatrix))
  end
  i_idx, j_idx = findnz(spMatrix)
  rank_mat = zeros(Int,length(order_idx))
  for k in 1:length(order_idx)
    rank_mat[order_idx[k]] = k
  end
  i_idx = [rank_mat[x] for x in i_idx]
  j_idx = [rank_mat[x] for x in j_idx]
  PyPlot.figure(figsize=(15.,14.))
  PyPlot.xlim(0,n)
  PyPlot.ylim(0,n)
  PyPlot.scatter(i_idx,j_idx,s=markersize,color="C0")
  if directed == false
    PyPlot.scatter(j_idx,i_idx,s=markersize,color="C0")
  end
  PyPlot.xlabel("nodes")
  PyPlot.ylabel("nodes")
  #title("Adjacency matrix with clustered nodes")
  if save == true
      PyPlot.savefig(file_name,bbox_inches="tight")
  end
end

function spy_sparse_den(spMatrix,cl_alloc,directed=true,file_name="clustering density.png")
  n = maximum(size(spMatrix))
  if directed == false
    spMatrix = dropzeros(spMatrix+transpose(spMatrix))
  end
  nodes_order = sortperm(cl_alloc)
  spMatrix = spMatrix[nodes_order,nodes_order]
  cl_alloc = cl_alloc[nodes_order]
  cl_unique = unique(cl_alloc)
  K = length(cl_unique)
  start_idx = zeros(Int64,K)
  end_idx = zeros(Int64,K)
  for k in 1:K
    cl_ = cl_unique[k]
    start_idx[k] = findfirst(cl_alloc.==cl_)
    end_idx[k] = findlast(cl_alloc.==cl_)
  end
  block_den = zeros(K,K)
  for i in 1:K
    for j in 1:K
      i_ind = start_idx[i]:end_idx[i]
      j_ind = start_idx[j]:end_idx[j]
      block_den[i,j] = sum(spMatrix[i_ind,j_ind].>0)/(length(i_ind)*length(j_ind))
    end
  end
  PyPlot.figure(figsize=(15.,14.))
  PyPlot.xlim(0,n)
  PyPlot.ylim(0,n)
  for i in 1:K
    if start_idx[i] > 1
      PyPlot.plot(1:n,start_idx[i]*ones(n),color="black",alpha=.8)
      PyPlot.plot(start_idx[i]*ones(n),1:n,color="black",alpha=.8)
    end
    for j in 1:K
      PyPlot.axhspan(start_idx[i], end_idx[i], start_idx[j]/n,end_idx[j]/n,
                     alpha=0.8*block_den[i,j]/maximum(block_den))
    end
  end
  PyPlot.xlabel("nodes")
  PyPlot.ylabel("nodes")
  #title("Adjacency matrix with clustered nodes")
  PyPlot.savefig(file_name,bbox_inches="tight")
  return block_den
end

function cluster_communities(activities, affinities)
  k = length(activities)
  n = length(affinities[1])
  max_idx = argmax(activities)
  #order_idx = sortperm(affinities[max_idx])
  order_idx = randperm(n)
  clusters = Dict{Int,Array{Int64,1}}()
  for x in order_idx
    #weights = [sqrt(activities[c])*affinities[c][x] for c in 1:k]
    weights = [activities[c]*affinities[c][x] for c in 1:k]
    c_x = argmax(weights)
    if haskey(clusters,c_x) == false
      clusters[c_x] = Int[]
    end
    push!(clusters[c_x],x)
  end
  final_order = []
  for c in keys(clusters)
    final_order = vcat(final_order,clusters[c])
  end
  return final_order, clusters
end

function plot_communitites_degree(sentAndReceived::Count,nbins::Int64=30)
  k_n = length(sentAndReceived)
  t = 1
  degrees = zeros(Int64,k_n)
  for (k,sAr) in sentAndReceived
    degrees[k] = 1/2*sum(sAr)
    t += 1
  end
  figure()
  plt[:hist](degrees,nbins)
  return degrees
end
