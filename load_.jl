#-------------------------------------------------------------------------------
# Plot and save the results of a saved run (finished or not)
#   In the 'Set Parameters' section, give the name of the dataset, its properties
#   as well as the folder where the interupted run is stored (current_dir will
#   be the time of the run, look in the result folder).
#-------------------------------------------------------------------------------

using HDF5, JLD
include("src/main.jl")



#-------------------------------------------------------------------------------
# Set parameters
#-------------------------------------------------------------------------------
# Data name, it has to be in the data/ folder as a .txt file
data_name = "wikipedia_3000"

# Main directory
main_dir = pwd()
# Directory from which we load the run
current_dir = "sigma 1/01-02-19_16-24-51"
load_dir = string("results/",data_name,"/",current_dir,"/variables/")


#-------------------------------------------------------------------------------
# Load dataset and variables of the interupted run
#-------------------------------------------------------------------------------
warm_start = false
PRINT_ = false

file_name = string("data/",data_name,".txt")
println(string("Reading the file ",file_name))
@time A = readdlm(file_name)
println()

println("Creating sparse matrix")
i_data = [Int(x) for x in A[:,1]]
j_data = [Int(x) for x in A[:,2]]
val_data = ones(Int, length(i_data))
n = max(maximum(i_data),maximum(j_data))
sparse_data = sparse(i_data,j_data,val_data,n,n)
println()

# Sparse matrix of observed entries and the ones to predict
I_all,J_all,V_all = findnz(sparse_data+speye(Int64,n))
all_ind_mat = dropzeros(sparse(I_all,J_all,ones(Int,length(I_all)),n,n))


variables = load(string(load_dir,"variables.jld"))
activities_list = variables["activities_list"]
n_active_list = variables["n_active_list"]
n_iter = findfirst(n_active_list.==0)-2
n_iter = (n_iter == -2 ? length(n_active_list) : n_iter)
n_active_list = variables["n_active_list"][1:n_iter]
kappa_list = variables["kappa_list"][1:n_iter]
sigma_list = variables["sigma_list"][1:n_iter]
tau_list = variables["tau_list"][1:n_iter]
alpha_list = variables["alpha_list"][1:n_iter]
beta_list = variables["beta_list"][1:n_iter]
if haskey(variables,"R_")
  R_ = variables["R_"]
  V_ = variables["V_"]
else
  partition_ = variables["partition_"]
  sentAndReceived_ = variables["sentAndReceived_"]
  c_kappa = kappa_list[n_iter]
  c_sigma = sigma_list[n_iter]
  c_tau = tau_list[n_iter]
  c_alpha = alpha_list[n_iter]
  c_beta = beta_list[n_iter]
  if haskey(variables,"skip")
    skip = variables["skip"]
  else
    skip = 60
  end
  R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)
end
K = size(activities_list)[2]
plot_true = false
pred_ratio = 0.


#-------------------------------------------------------------------------------
# Save plots
#-------------------------------------------------------------------------------
results_path = string("results/",data_name,"/",current_dir,"/imgV/")
mkpath(results_path)
cd(results_path)
include("src/plot_results.jl")
cd(main_dir)


results_path = string("results/",data_name,"/",current_dir,"/imgV/")
mkpath(results_path)
cd(results_path)
order,clusters = cluster_communities(R_,V_)
clusters_o = zeros(n)
for (k,cl) in clusters
  clusters_o[cl] = k
end
ioff()
spy_sparse_order(sparse_data,order,1.)
spy_sparse_den(sparse_data,clusters_o)
PyPlot.close()
ion()
cd(main_dir)
