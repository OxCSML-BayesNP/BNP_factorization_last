using HDF5, JLD
using MCMCDiagnostics
include("src/main.jl")




warm_start = false

pred_ratio = 0.

compute_cluster = false

data_name = "data_for_sigma_800/alpha free"
data_txt = "data_for_sigma_800"

#data_name = "polblogs/alpha=02"
#data_txt = "polblogs"

main_dir = pwd()
current_dir = Dict()
current_dir[1] = "26-10-18_09-35"
current_dir[2] = "24-10-18_07-11"
current_dir[3] = "24-10-18_02-01"
save_dir = "3 chains 18-2-19 burn"
skip = 50

activities_list = Dict()
n_active_list = Dict()
kappa_list = Dict()
sigma_list = Dict()
tau_list = Dict()
alpha_list = Dict()
beta_list = Dict()
clusterings_list = Dict()

M = length(current_dir)
clus_size = 0
n = 0
R_ = 0
V_ = 0
for m in 1:M
  load_dir = string("results/",data_name,"/",current_dir[m],"/variables/")
  variables = load(string(load_dir,"variables.jld"))
  activities_list[m] = variables["activities_list"]
  n_active_list[m] = variables["n_active_list"]
  kappa_list[m] = variables["kappa_list"]
  sigma_list[m] = variables["sigma_list"]
  tau_list[m] = variables["tau_list"]
  alpha_list[m] = variables["alpha_list"]
  beta_list[m] = variables["beta_list"]
  if compute_cluster == true
    clusterings_list[m] = variables["clusterings"]
    n = size(clusterings_list[m])[1]
  end
end


if compute_cluster == true
  clusterings = clusterings_list[1]
  for m in 2:M
    clusterings = hcat(clusterings,clusterings_list[m])
  end
  clusterings_size = size(clusterings)[2]

  # Load data
  file_name = string("data/",data_txt,".txt")
  println(string("Reading the file ",file_name))
  @time A = readdlm(file_name)
  println()
  # Create sparse matrix
  println("Creating sparse matrix")
  #i_data = [Int(x+1) for x in A[:,1]] #enron, email eu
  #j_data = [Int(x+1) for x in A[:,2]] #enron, email-eu
  i_data = [Int(x) for x in A[:,1]] #polnlogs, wikipedia_3000, Protein230, NIPS234, NIPS12
  j_data = [Int(x) for x in A[:,2]] #polnlogs, wikipedia_3000, Protein230, NIPS234, NIPS12
  val_data = ones(Int, length(i_data))
  sparse_data = sparse(i_data,j_data,val_data,n,n)
  println()
end



n_iter = length(kappa_list[1])
burn_start = (warm_start ? 1 : Int(3*n_iter/4))
K = size(activities_list[1])[2]

burn_start = trunc(Int,n_iter/2) # DELETE

plot_true = false # DELETE
kappa_true = 1. # DELETE
alpha_true = 0.05 # DELETE
sigma_true = 0.2 # DELETE
tau_true = 0.15 # DELETE
kn_true = 42 # DELETE



results_path = string("results/",data_name,"/",save_dir,"/")
mkpath(results_path)

open(results_path*"info.txt","w") do f
  write(f,"Runs:\n")
  for m in 1:M
    write(f,string("  Chain $m = ", current_dir[m],"\n"))
  end
end




cd(results_path)

rc("font",size=24.)

ioff()

#-------------------------------------------------------------------------------
# Plot and save number of communities
#-------------------------------------------------------------------------------
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = n_active_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(kn_true*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of the number of active communities")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Active communities")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of the number of active communities.png",bbox_inches="tight")
PyPlot.close()

# Plot histogram
hist_values = vcat(n_active_list[1][burn_start:n_iter],
                  n_active_list[2][burn_start:n_iter],
                  n_active_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,15,alpha=0.7,histtype="barstacked")
PyPlot.xlabel("Active communities")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of the number of active communities.png",bbox_inches="tight")
PyPlot.close()

# Save in txt
conv_coeff = potential_scale_reduction(n_active_list[1][burn_start:n_iter],
                                      n_active_list[2][burn_start:n_iter],
                                      n_active_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,"\n")
  write(f,"Convergence:\n")
  write(f,string("   n active comm : R hat= ", conv_coeff,"\n"))
end


#-------------------------------------------------------------------------------
# Plot and save parameters
#-------------------------------------------------------------------------------

# Kappa Parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = kappa_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(kappa_true*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Kappa")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("kappa")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of kappa.png",bbox_inches="tight")
PyPlot.close()

conv_coeff = potential_scale_reduction(kappa_list[1][burn_start:n_iter],
                                      kappa_list[2][burn_start:n_iter],
                                      kappa_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,string("   kappa : R hat = ", conv_coeff,"\n"))
end

# Plot histogram
hist_values = vcat(kappa_list[1][burn_start:n_iter],
                  kappa_list[2][burn_start:n_iter],
                  kappa_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,20,alpha=0.7)
PyPlot.xlabel("Kappa")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of kappa.png",bbox_inches="tight")
PyPlot.close()

# Log Kappa Parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = kappa_list[m][burn_start:n_iter]
  PyPlot.plot(log.(chain),label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(log(kappa_true)*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Kappa")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("log kappa")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of log kappa.png",bbox_inches="tight")
PyPlot.close()

# Plot histogram
hist_values = vcat(log.(kappa_list[1][burn_start:n_iter]),
                  log.(kappa_list[2][burn_start:n_iter]),
                  log.(kappa_list[3][burn_start:n_iter]))
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,30,alpha=0.7)
PyPlot.xlabel("Log Kappa")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of log kappa.png",bbox_inches="tight")
PyPlot.close()




# Tau parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = tau_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(tau_true*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Tau")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("tau")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of tau.png",bbox_inches="tight")
PyPlot.close()

conv_coeff = potential_scale_reduction(tau_list[1][burn_start:n_iter],
                                      tau_list[2][burn_start:n_iter],
                                      tau_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,string("   tau : R hat = ", conv_coeff,"\n"))
end

# Plot histogram
hist_values = vcat(tau_list[1][burn_start:n_iter],
                  tau_list[2][burn_start:n_iter],
                  tau_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,30,alpha=0.7)
PyPlot.xlabel("Tau")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of tau.png",bbox_inches="tight")
PyPlot.close()


# Tau parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = tau_list[m][burn_start:n_iter]
  PyPlot.plot(log.(chain),label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(log(tau_true)*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Tau")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("log tau")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of log tau.png",bbox_inches="tight")
PyPlot.close()

# Plot histogram
hist_values = vcat(log.(tau_list[1][burn_start:n_iter]),
                  log.(tau_list[2][burn_start:n_iter]),
                  log.(tau_list[3][burn_start:n_iter]))
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,30,alpha=0.7)
PyPlot.xlabel("Log Tau")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of log tau.png",bbox_inches="tight")
PyPlot.close()



# Sigma Parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = sigma_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(sigma_true*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Sigma")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Sigma")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of sigma.png",bbox_inches="tight")
PyPlot.close()

conv_coeff = potential_scale_reduction(sigma_list[1][burn_start:n_iter],
                                      sigma_list[2][burn_start:n_iter],
                                      sigma_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,string("   sigma : R hat = ", conv_coeff,"\n"))
end

# Plot histogram
hist_values = vcat(sigma_list[1][burn_start:n_iter],
                  sigma_list[2][burn_start:n_iter],
                  sigma_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,22,alpha=0.7)
PyPlot.xlabel("Sigma")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of sigma.png",bbox_inches="tight")
PyPlot.close()


# Alpha Parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = alpha_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
# DELETE
if plot_true
  PyPlot.plot(alpha_true*ones(n_iter-burn_start+1),label="True values", linewidth = 4)
end
# END DELETE
#PyPlot.title("MCMC estimation of Alpha")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Alpha")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of alpha.png",bbox_inches="tight")
PyPlot.close()

conv_coeff = potential_scale_reduction(alpha_list[1][burn_start:n_iter],
                                      alpha_list[2][burn_start:n_iter],
                                      alpha_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,string("   alpha : R hat = ", conv_coeff,"\n"))
end

# Plot histogram
hist_values = vcat(alpha_list[1][burn_start:n_iter],
                  alpha_list[2][burn_start:n_iter],
                  alpha_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,30,alpha=0.7)
PyPlot.xlabel("Alpha")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of alpha.png",bbox_inches="tight")
PyPlot.close()


# Beta Parameter
PyPlot.figure(figsize=(15.,12.))
for m in 1:M
  chain = beta_list[m][burn_start:n_iter]
  PyPlot.plot(chain,label="Chain $m",alpha=.7)
end
#PyPlot.title("MCMC estimation of Beta")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Beta")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of beta.png",bbox_inches="tight")
PyPlot.close()

conv_coeff = potential_scale_reduction(beta_list[1][burn_start:n_iter],
                                      beta_list[2][burn_start:n_iter],
                                      beta_list[3][burn_start:n_iter])
open("info.txt","a+") do f
  write(f,string("   beta : R hat = ", conv_coeff,"\n"))
end

# Plot histogram
hist_values = vcat(beta_list[1][burn_start:n_iter],
                  beta_list[2][burn_start:n_iter],
                  beta_list[3][burn_start:n_iter])
PyPlot.figure(figsize=(15.,12.))
PyPlot.plt[:hist](hist_values,30,alpha=0.7)
PyPlot.xlabel("Beta")
PyPlot.ylabel("Frequency")
PyPlot.legend()
PyPlot.savefig("Histogram of beta.png",bbox_inches="tight")
PyPlot.close()


#-------------------------------------------------------------------------------
# Plot and save the K largest activities
#-------------------------------------------------------------------------------
for r_idx in 1:K
  PyPlot.figure(figsize=(15.,12.))
  for m in 1:M
    estimated_r_list = activities_list[m][:,r_idx][burn_start:n_iter]
    PyPlot.plot(estimated_r_list,label="Chain $m",alpha=.7)
  end
  #PyPlot.title("MCMC estimation of $r_idx largest activity")
  PyPlot.xlabel("MCMC iteration (x$skip)")
  PyPlot.ylabel("Activity")
  PyPlot.legend()
  PyPlot.savefig("MCMC estimation of $r_idx largest activity")
  PyPlot.close()
end





#-------------------------------------------------------------------------------
# Find clustering maximizing bayes criteria (see notes)
#-------------------------------------------------------------------------------
order_star = []
if compute_cluster
  # Compute clusters
  clusterings_crit = zeros(clusterings_size)
  for i in 1:n
    for j in (i+1):n
      post_mean = 0.
      for m in 1:clusterings_size
        if clusterings[i,m] == clusterings[j,m]
          post_mean += 1.
        end
      end
      post_mean /= clusterings_size
      for m in 1:clusterings_size
        if clusterings[i,m] == clusterings[j,m]
          clusterings_crit[m] = 1-2*post_mean
        end
      end
    end
  end
  clustering_ = clusterings[:,indmax(clusterings_crit)]
  idx_perm = randperm(n)
  clustering_star = clustering_
  left_p = ones(n)
  left_p[1:759] = -1.
  left_p_z = zeros(n)
  left_p_z[1:759] = -1.
  #left_p = -left_p
  if data_txt == "polblogs"
    n_clusters = trunc(Int,maximum(clustering_star))
    clusters_o = zeros(n)
    prop_democrates = zeros(n_clusters)
    for cl_i in 1:n_clusters
      nodes_in_cl = zeros(n)
      nodes_idx = find( x->(x==cl_i),clustering_star )
      nodes_in_cl[nodes_idx] = 1.
      clusters_o[nodes_idx] = transpose(nodes_in_cl)*left_p/sqrt(sum(nodes_in_cl))
      prop_democrates[cl_i] = transpose(nodes_in_cl)*left_p_z/sum(nodes_in_cl)
    end
    order_star = idx_perm[sortperm(clusters_o[idx_perm])]
  elseif data_txt == "wikipedia_3000"
    n_clusters = trunc(Int,maximum(clustering_star))
    clusters_o = zeros(n)
    range_arr = Array((1:n).')
    for cl_i in 1:n_clusters
      nodes_in_cl = zeros(n)
      nodes_idx = find( x->(x==cl_i),clustering_star )
      nodes_in_cl[nodes_idx] = 1.
      clusters_o[nodes_idx] = sum(range_arr*nodes_in_cl)/(sum(nodes_in_cl))
    end
    order_star = idx_perm[sortperm(clusters_o[idx_perm])]
  else
    order_star = idx_perm[sortperm(clustering_star[idx_perm])]
    clusters_o = clustering_star
    #order_star = idx_perm[order_]
  end
  ioff()
  spy_sparse_order(sparse_data,order_star,3.)
  spy_sparse_den(sparse_data,clusters_o)
  PyPlot.close("all")
  ion()
  if data_name == "polblogs"
    writedlm("Proportion of democrates per cluster.txt",prop_democrates)
  end
end



#=
true_order = 1:n
partition_ = Factorized{Bool}()
sortperm_R_ = sortperm(R_,rev=true)
K=15
ioff()
cd(main_dir)
mkpath(results_path*"edge partition/")
mkpath(results_path*"edge partition ordered/")
for k_ in 1:min(length(R_),K)
  k = sortperm_R_[k_]
  partition_[k] = spzeros(n,n)
  for t in 1:length(i_data)
    i_ = i_data[t]
    j_ = j_data[t]
    #partition_[k][i_,j_] = min(1,rand(Poisson(R_[k]*V_[k][i_]*V_[k][j_])))
    partition_[k][i_,j_] = R_[k]*V_[k][i_]*V_[k][j_]
  end
  cd(main_dir)
  cd(results_path*"edge partition ordered/")
  spy_sparse_order(partition_[k],order_star,2.,true,"Feature $k ordered.png")
  cd(main_dir)
  cd(results_path*"edge partition/")
  spy_sparse_order(partition_[k],true_order,2.,true,"Feature $k.png")
  PyPlot.close()
end
ion()


n = 1490
cd(main_dir)
load_path = "results/polblogs/alpha=02"
cd(load_path)
partition_ = load("partition_.jld")["partition_"]
dem_idx = 1:759
rep_idx = 760:1490
for k in keys(partition_)
  den_block = zeros(2,2)
  part = partition_[k]
  den_block[1,1] = sum(part[dem_idx,dem_idx].>0)/sum(part.>0)
  den_block[1,2] = sum(part[dem_idx,rep_idx].>0)/sum(part.>0)
  den_block[2,1] = sum(part[rep_idx,dem_idx].>0)/sum(part.>0)
  den_block[2,2] = sum(part[rep_idx,rep_idx].>0)/sum(part.>0)
  writedlm("feature $k.txt",den_block)
end
=#




PyPlot.close("all")
ion()
cd(main_dir)
