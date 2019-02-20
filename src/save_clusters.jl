
using HDF5, JLD
order,clusters = cluster_communities(R_,V_)
save("clusters.jld","clusters",clusters)

# Load the data using load("clusters.jld")["clusters"]
