#-------------------------------------------------------------------------------
# Carry on an interupted run
#   In the 'Set Parameters' section, give the name of the dataset, its properties
#   as well as the folder where the interupted run is stored (current_dir will
#   be the time of the run, look in the result folder).
#-------------------------------------------------------------------------------


using HDF5, JLD
include("src/main.jl")

using ProgressMeter
using HDF5, JLD


#-------------------------------------------------------------------------------
# Set parameters
#-------------------------------------------------------------------------------

# Data name, it has to be in the data/ folder as a .txt file
data_name = "deezer_RO_und"

# Comments about the run
comments = "Alpha fixed to free\n new update of mu tilde \n new measure unobs \n sigma max 0.25"

# Main directory
main_dir = pwd()

# Directory from which we load interupted run
result_folder = "sigma 025"
current_dir = "28-11-18_11-23-54"
load_dir = string("results/",data_name,"/",result_folder,"/",current_dir,"/variables/")
println("Current directory: "*result_folder*"/"*current_dir)
println()

# Properties of the dataset
weighted = false
directed = false
# Are there self edges
self_edge = true

# Proportion of entries to mask
pred_ratio = 0.

# Saving clustering
save_clustering = true
# Number of clustering to save
n_clusterings = 10
n_clusterings = (save_clustering ? n_clusterings : 2)


# Set to true to display the current value of sigma every 5% of the progress. It also
# allows to save the current state of the run every 5%. Then use continue_.jl if the script
# is stopped before the end of the sampler to carry from last save.
monitoring_sigma = true


# Number of updates of the hyperparameters per iteration
n_steps_hyper = 10

# Set to true if the parameter is fixed by user (to the initial value)
FIXED_KAPPA = false
FIXED_SIGMA = false
FIXED_TAU = false
FIXED_ALPHA = false
FIXED_BETA = true


#-------------------------------------------------------------------------------
# Load dataset and variables of the interupted run
#-------------------------------------------------------------------------------

# Load dataset
file_name = string("data/",data_name,".txt")
println(string("Reading the file ",file_name))
@time A = readdlm(file_name)
println()

# Create a sparse representation of the graph
println("Creating sparse matrix")
i_data = [Int(x) for x in A[:,1]]
j_data = [Int(x) for x in A[:,2]]
val_data = ones(Int, length(i_data))
if directed==false
  for t in 1:length(i_data)
    if i_data[t] < j_data[t]
      val_data[t] = 0
    end
  end
end
n = max(maximum(i_data),maximum(j_data))
sparse_data = sparse(i_data,j_data,val_data,n,n)
sparse_data = dropzeros(sparse_data)

println()


# Sparse matrix of observed entries and the ones to predict
I_all,J_all,V_all = findnz(sparse_data+speye(Int64,n))
all_ind_mat = dropzeros(sparse(I_all,J_all,ones(Int,length(I_all)),n,n))
Z_tilde = sparse_data
to_predict=sparse(zeros(Int64,1,1))
pred_vect=zeros(Float64,1)

# Load variables
variables = load(string(load_dir,"variables.jld"))
activities_list = variables["activities_list"]
n_active_list = variables["n_active_list"]
n_iter = variables["n_iter"]
n_active_list = variables["n_active_list"]
kappa_list = variables["kappa_list"]
sigma_list = variables["sigma_list"]
tau_list = variables["tau_list"]
alpha_list = variables["alpha_list"]
beta_list = variables["beta_list"]

K = size(activities_list)[2]
# Number of iterations as burn-in
burn = floor(Int,3.*n_iter/4)

last_iter = findfirst(n_active_list.==0)-2
if last_iter < 0
  last_iter = n_iter
end
partition_ = variables["partition_"]
sentAndReceived_ = variables["sentAndReceived_"]
c_kappa = kappa_list[last_iter]
c_sigma = sigma_list[last_iter]
c_tau = tau_list[last_iter]
c_alpha = alpha_list[last_iter]
c_beta = beta_list[last_iter]
if haskey(variables,"skip")
  skip = variables["skip"]
else
  skip = 50
end

# Hyperparameters for the MH update
if haskey(variables,"prior_params")
  prior_params = variables["prior_params"]
  prop_params = variables["prop_params"]
else
  prior_params = Dict()
  prior_params["kappa"] = (.2,.1)
  prior_params["sigma"] = (.1,.1)
  prior_params["tau"] = (.2,.1)
  prior_params["alpha"] = (.1,.1)
  prior_params["beta"] = (.1,.1)

  prop_params = Dict()
  prop_params["kappa"] = 0.04
  prop_params["sigma"] = 0.08
  prop_params["tau"] = 0.04
  prop_params["alpha"] = 0.02
  prop_params["beta"] = 0.04
end


# Load clusterings variables
# Iterations for which we save the clustering
ind_clusterings = variables["ind_clusterings"]
# Save the assignation of each node to its cluster
# clusterings[i,clus_idx] = idx of cluster of node i
clusterings = variables["clusterings"]
# Initialize index of current clustering
idx_clustering = findfirst(ind_clusterings.==0)

if idx_clustering == 0
  idx_clustering = 1
end

# Debug Variables
PRINT_ = false
plot_true = false
warm_start = false

#-------------------------------------------------------------------------------
# Starting the MCMC iterations
#-------------------------------------------------------------------------------
println(string("Starting Gibbs sample with ",n_iter," steps"))
tic()
@showprogress for i in (last_iter+1):n_iter
    # Update measure
    R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

    # Update partition
    partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)

    # Update hyperparameters
    for t in 1:n_steps_hyper
      c_kappa,c_sigma,c_tau,c_alpha,c_beta = update_parameters_neg2(c_kappa,
                                                                  c_sigma,
                                                                  c_tau,
                                                                  c_alpha,
                                                                  c_beta,
                                                                  prior_params,
                                                                  prop_params,
                                                                  R_,
                                                                  V_,
                                                                  sentAndReceived_,
                                                                  s_min)
    end
    PRINT_ = false
    for bob in 1:(skip-1)
        # Update measure
        R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

        # Update partition
        partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)

        # Update hyperparameters
        for t in 1:n_steps_hyper
            c_kappa,c_sigma,c_tau,c_alpha,c_beta = update_parameters_neg2(c_kappa,
                                                                      c_sigma,
                                                                      c_tau,
                                                                      c_alpha,
                                                                      c_beta,
                                                                      prior_params,
                                                                      prop_params,
                                                                      R_,
                                                                      V_,
                                                                      sentAndReceived_,
                                                                      s_min)
        end

    end


    # Store values
    n_active_list[i] = n_observed
    sort_idx = sortperm(R_,rev=true)
    sorted_R_ = R_[sort_idx]
    for j in 1:min(n_observed,K)
      activities_list[i,j] = sorted_R_[j]*sum(V_[sort_idx[j]])
    end

    # Store clustering
    if i > burn && (i-burn)%floor(Int,(n_iter-burn)/n_clusterings) == 0 && idx_clustering <= n_clusterings
      ind_clusterings[idx_clustering] = i
      for node_idx in 1:n
        weights = [sqrt(R_[c])*V_[c][node_idx] for c in sort_idx[1:min((2*K),length(R_))]]
        clusterings[node_idx,idx_clustering] = indmax(weights)
      end
      idx_clustering += 1
    end


    print_each = floor(Int,n_iter/20)
    # If monitoring sigma
    if monitoring_sigma && i%print_each == 0
      println(string("Progress: ", 100.*i/n_iter,"%"))
      println(string("Current sigma: ",c_sigma))
      println(string("Length R: ",length(R_)))

      # Save Variables
      results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/variables/")
      mkpath(results_path)
      cd(results_path)

      # Store the variables
      save("variables.jld","activities_list",activities_list,
                          "n_active_list",n_active_list,
                          "kappa_list",kappa_list,
                          "sigma_list",sigma_list,
                          "tau_list",tau_list,
                          "alpha_list",alpha_list,
                          "beta_list",beta_list,
                          "partition_",partition_,
                          "sentAndReceived_",sentAndReceived_,
                          "clusterings",clusterings,
                          "ind_clusterings",ind_clusterings,
                          "n_iter",n_iter,
                          "skip",skip,
                          "prop_params",prop_params,
                          "prior_params",prior_params)
      cd(main_dir)

      PRINT_ = true
    end

    kappa_list[i] = c_kappa
    sigma_list[i] = c_sigma
    tau_list[i] = c_tau
    alpha_list[i] = c_alpha
    beta_list[i] = c_beta

end

# Compute partition and corresponding active communities of last iteration
R_t,V_t,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)
R_ = R_t[1:n_observed]
V_ = Affinity()
for k in 1:n_observed
  V_[k] = V_t[k]
end
elapsed_time = toc()

#-------------------------------------------------------------------------------
# Saving main variables
#-------------------------------------------------------------------------------

results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/variables/")
mkpath(results_path)
cd(results_path)

# Store the variables
save("variables.jld","activities_list",activities_list,
                    "n_active_list",n_active_list,
                    "kappa_list",kappa_list,
                    "sigma_list",sigma_list,
                    "tau_list",tau_list,
                    "alpha_list",alpha_list,
                    "beta_list",beta_list,
                    "partition_",partition_,
                    "sentAndReceived_",sentAndReceived_,
                    "clusterings",clusterings,
                    "ind_clusterings",ind_clusterings,
                    "n_iter",n_iter,
                    "skip",skip,
                    "prop_params",prop_params,
                    "prior_params",prior_params)
cd(main_dir)
# Load the data using
#   load("prediction.jld")[name of the variable]




#-------------------------------------------------------------------------------
# Plotting results
#-------------------------------------------------------------------------------

while true
  println()
  println("Plot and save results ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    plot_true = false
    results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/img/")
    mkpath(results_path)
    cd(results_path)
    include(main_dir*"/src/plot_results.jl")
    break
  end
end
cd(main_dir)


#-------------------------------------------------------------------------------
# Plotting clusters
#-------------------------------------------------------------------------------
while true
  println()
  println("Plot clusters ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/img/")
    mkpath(results_path)
    cd(results_path)
    order,clusters = cluster_communities(R_,V_)
    clusters_o = zeros(n)
    for (k,cl) in clusters
      clusters_o[cl] = k
    end
    ioff()
    spy_sparse_order(sparse_data,order,2.,directed)
    spy_sparse_den(sparse_data,clusters_o)
    PyPlot.close()
    ion()
    break
  end
end
cd(main_dir)

#-------------------------------------------------------------------------------
# Saving clusters
#-------------------------------------------------------------------------------
while true
  println()
  println("Save clusters in .jld ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/clusters/")
    mkpath(results_path)
    cd(results_path)
    include("src/save_clusters.jl")
    break
  end
end
cd(main_dir)


#-------------------------------------------------------------------------------
# Saving edge allocation
#-------------------------------------------------------------------------------
order,clusters = cluster_communities(R_,V_)
while true
  println()
  println("Save edge allocation ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    break
  end
  if continue_ == "y"
    # Save partition with posterior order
    results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/img/edge_partition_/")
    mkpath(results_path)
    cd(results_path)
    ioff()
    for k in keys(partition_)
      spy_sparse_order(partition_[k],order,2.,directed,"Feature $k.png")
      PyPlot.close()
    end
    cd(main_dir)
    # Save partition with natural order
    results_path = string("results/",data_name,"/",result_folder,'/',current_dir,"/img/edge_partition/")
    mkpath(results_path)
    cd(results_path)
    for k in keys(partition_)
      spy_sparse_order(partition_[k],1:n,2.,directed,"Feature $k.png")
      PyPlot.close()
    end
    ion()
    break
  end
end
PyPlot.close("all")
cd(main_dir)
