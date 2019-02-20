include("src/main.jl")

using HDF5, JLD
using LightGraphs

# Parameters
n=700;kappa=2.5;tau=.2;sigma=0.2;alpha=.1;beta=1.
data_name = "data_for_sigma_700"

println("Parameters set to")
println(string("  n = ", n))
println(string("  kappa = ", kappa))
println(string("  tau = ", tau))
println(string("  sigma = ", sigma))
println(string("  alpha = ", alpha))
println(string("  beta = ", beta))
println(string("  data name when saving = ", data_name))
println()

# Sample a graph from the model
println("Generating model")
@time R, V, Z = generate_model_(n,kappa,tau,sigma,alpha,beta)
A = complete_graph(Z)
G = DiGraph(A)
V_ = Affinity()
for k in 1:length(R)
  V_[k] = V[:,k]
end
println(string("  Number of edges: ", ne(G)))
println(string("  Maximal entry: ", maximum(A)))
println()


# Print vector of activities
println("Vector of activities")
println(sort(R,rev=true))
println()

# Compute the local clustering coefficients (vector of size n)
println("Computing local clustering coefficients")
@time loc_clus = local_clustering_coefficient(G,1:n)
filter!(x->x!=0, loc_clus)
var_p = var(loc_clus)
quant2_p = quantile(loc_clus,.2)
quant8_p = quantile(loc_clus,.8)
println(string("  Variance: ", var_p))
println(string("  20% Quantile: ", quant2_p))
println(string("  80% Quantile: ", quant8_p))
println()

# Plot the adjacency matrix of the graph, reordered to make
# the communities appear
order, clusters = cluster_communities(R,V_)
spy_sparse_order(A,order,2.)

# Save the data for further tests
main_dir = pwd()
while true
  println()
  println("Save data in .jld ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    results_path = string("data/",data_name,"/")
    mkpath(results_path)
    cd(results_path)
    # Store the variables
    save("data.jld","R_",R,
                    "V_",V_,
                    "Z",Z)
    # Save main parameters
    open("info.txt","w") do f
      write(f,"Dataset:\n")
      write(f,string("  name = ", data_name,"\n"))
      write(f,string("  number of nodes = ", n,"\n"))
      write(f,string("  number of edges = ", sum(sparse_data),"\n"))
      write(f,string("  directed = true","\n"))
      write(f,string("  weighted = false","\n\n"))

      write(f,"Initial parameters:\n")
      write(f,string("  kappa = ", kappa,"\n"))
      write(f,string("  tau = ", tau,"\n"))
      write(f,string("  sigma = ", sigma,"\n"))
      write(f,string("  alpha = ", alpha,"\n"))
      write(f,string("  beta = ", beta,"\n\n"))

    end
    # Save the unweighted graph
    cd(main_dir)
    I_,J_,Val_ = findnz(A)
    writedlm(string("data/",data_name,".txt"),hcat(I_,J_))
    break
  end
end
cd(main_dir)
