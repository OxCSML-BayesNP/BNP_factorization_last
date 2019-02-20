using HDF5, JLD
include("src/main.jl")
data_name = "wikipedia_3000"
warm_start = false
PRINT_ = true

main_dir = pwd()
current_dir = "sigma 1/01-02-19_16-24-51"
load_dir = string("results/",data_name,"/",current_dir,"/variables/")


file_name = string("data/",data_name,".txt")
println(string("Reading the file ",file_name))
@time A = readdlm(file_name)
println()

println("Creating sparse matrix")
#i_data = [Int(x+1) for x in A[:,1]] #enron, email eu
#j_data = [Int(x+1) for x in A[:,2]] #enron, email-eu
i_data = [Int(x) for x in A[:,1]] #polnlogs, wikipedia_3000, Protein230, NIPS234, NIPS12
j_data = [Int(x) for x in A[:,2]] #polnlogs, wikipedia_3000, Protein230, NIPS234, NIPS12
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

# Polblogs histogram
cd(results_path)
if data_name == "polblogs_undirected"
  party_aff = zeros(length(clusters))
  weights_v = zeros(length(clusters))
  t = 1
  for (k,c) in clusters
    for c_i in c
      if c_i > 759
        party_aff[t] += 1.
      end
    end
    party_aff[t] = party_aff[t]/length(c)
    weights_v[t] = length(c)
    t += 1
  end

  PyPlot.figure(figsize=(30.,24.))
  PyPlot.plt[:hist](party_aff,30,weights = weights_v)
  title("Clusters historgram")
  legend()
  PyPlot.savefig("Clusters historgram.png",bbox_inches="tight")
end
cd(main_dir)
