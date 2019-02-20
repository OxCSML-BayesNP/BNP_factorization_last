include("src/main.jl")

using PyPlot
using ProfileView
using ProgressMeter
using HDF5, JLD

# Parameters
n=400;kappa=1.5;tau=.6;sigma=0.17;alpha=.015;beta=.05
#beta = 1.*alpha

weighted = true
self_edge= true
directed = true

data_name = string("Synthetic/ n=",n,
                  " kappa = ", kappa,
                  " tau = ",tau,
                  " sigma = ",sigma,
                  " alpha = ", alpha,
                  " beta = ",beta,
                  " weighted = ", weighted)




warm_start = true

pred_ratio = 0.#20

# Plot true values
plot_true = true



# Gibbs sampler

# Number of iterations
n_iter = (warm_start ? 1000 : 150000)
# Number of communities to store
K = 10

# Hyperparameters for the MH updates
prior_params = Dict()
prior_params["kappa"] = (.1,.1)
prior_params["sigma"] = (.1,.1)
prior_params["tau"] = (1.,1.)
prior_params["alpha"] = (.1,.1)
prior_params["beta"] = (.1,.1)

prop_params = Dict()
prop_params["kappa"] = 0.08
prop_params["sigma"] = 0.08
prop_params["tau"] = 0.08
prop_params["alpha"] = 0.08
prop_params["beta"] = 0.08

# Set to true if user wants to fix the parameter to true value
FIXED_KAPPA = false
FIXED_SIGMA = false
FIXED_TAU = false
FIXED_ALPHA = false
FIXED_BETA = true


c_kappa = (FIXED_KAPPA ? kappa : 2.)
c_sigma = (FIXED_SIGMA ? sigma : 0.)
c_tau = (FIXED_TAU ? tau : .8)
c_alpha = alpha
c_beta = beta

# Number of updates of the hyperparameters per iteration
n_steps_hyper = 10

println("Parameters set to")
println(string("  n = ", n))
println(string("  kappa = ", kappa))
println(string("  tau = ", tau))
println(string("  sigma = ", sigma))
println(string("  alpha = ", alpha))
println(string("  beta = ", beta))
println(string("  warm_start = ", warm_start))
println(string("  weighted = ", weighted))
println()

# Generate from model
println("Generating model")
@time R, V, Z = generate_model_(n,kappa,tau,sigma,alpha,beta)
println()




# Inference on the community with higher activity
println("Inference on the community with higher activity")
c = indmax(R)
Z_c = Z[c]
sent = vec(sum(Z_c,2))
received = vec(sum(Z_c,1))
@time rand_observed_community(sent+received,
                              tau,sigma,alpha,beta)

r,v = rand_observed_community(sent+received,
                              tau,sigma,alpha,beta)
println()


# Update the GGP and slice variables
println("One step update of the variables")
Z_ = complete_graph(Z)
I,J = findnz(Z_)
println(string("  Number of edges: ", length(I)))
println(string("  Maximal entry: ", maximum(Z_)))
println()


println("Initialize variables")
partition_ = Factorized{Bool}()
sentAndReceived_ = Count()

@time for k in keys(Z)
  I_,J_,V_ = findnz(Z[k])
  V_s = [(v_ > 0) for v_ in V_]
  partition_[k] = sparse(I_,J_,V_s,n,n)
  sentAndReceived_[k] = reshape(sum(Z[k],1),n) + reshape(sum(Z[k],2),n)
end

# Update measure
println("Update measure and slice variables")
@time R_,V_,n_observed,slice_matrix = update_measure(partition_,sentAndReceived_,Z_,kappa,tau,sigma,alpha,beta)
println()


# Update partition weighted graph
println("Update partition weighted")
@time partition2,sentAndReceived2 = update_partition(R_,V_,slice_matrix,Z_)
println()



# Update partition unweighted
println("Update partition unweighted")
@time partition2,sentAndReceived2 = update_partition_unweighted(R_,V_,slice_matrix,Z_)
println()


# Print vector of activities
println("Vector of activities")
println(sort(R,rev=true))

# Construct observed adjacency matrix
Z_tilde = copy(Z_)


# Select entries to predict
println("Masking entries to predict")
n_to_predict = Int(pred_ratio*n^2)
println("Select $n_to_predict indices to predict")
@time to_predict = sparse(rand(1:n,n_to_predict),rand(1:n,n_to_predict),ones(Int64,n_to_predict),n,n)
I_pred,J_pred = findnz(to_predict)
n_to_predict = length(I_pred)
v_true = Array{Int64}(n_to_predict)
println("Mask corresponding entries")
@time for t in 1:n_to_predict
  i_pred = I_pred[t]
  j_pred = J_pred[t]
  v_true[t] = Z_[i_pred,j_pred]
  if weighted == false
    v_true[t] = min(v_true[t],1)
  end
  Z_tilde[i_pred,j_pred] = 0
end
I_tilde,J_tilde,V_tilde = findnz(Z_tilde)
Z_tilde = sparse(I_tilde,J_tilde,V_tilde,n,n)
# Vecor with integer predictions
pred_vect = ones(Float64,n_to_predict)
# Vector of posterior mean
pred_average_vect = zeros(Float64,n_to_predict)
# Sparse matrix of observed entries and the ones to predict
I_all,J_all,V_all = findnz(Z_tilde+to_predict)
all_ind_mat = sparse(I_all,J_all,V_all,n,n)
println()




# Gibbs sampler Weigted

println()

# Ask if the user wants to continue
println(string("Gibbs sampler with ",n_iter," iterations"))
while true
  println("Continue ? [y/n]")
  continue_ = chomp(readline())
  if continue_ == "n"
    error("Code stopped")
  end
  if continue_ == "y"
    break
  end
end

# List of the number of active communities
n_active_list = zeros(Int,n_iter)
# List of the top K activites
activities_list = zeros(n_iter,K)

# List of l2 errors for prediction
error_list = zeros(n_iter)
error_mean_list = zeros(n_iter)

kappa_list = zeros(n_iter)
sigma_list = zeros(n_iter)
tau_list = zeros(n_iter)
alpha_list = zeros(n_iter)
beta_list = zeros(n_iter)
s_min_list = zeros(n_iter)



s_min = 0.

println()

# If we observe the full adjacency matrix (weighted graph)

if warm_start == false
  partition_ = Factorized{Bool}()
  sentAndReceived_ = Count()
  I_,J_,V_ = findnz(Z_tilde)
  V_s = [(v_ > 0) for v_ in V_]
  partition_[1] = sparse(I_,J_,V_s,n,n)
  sentAndReceived_[1] = reshape(sum(Z_tilde,1),n) + reshape(sum(Z_tilde,2),n)
else
  K_init = trunc(Int,active_feature_mean(n, c_kappa, c_tau, c_sigma, c_alpha, c_beta))+1
  s_min_init = Inf
  r_dist_init = Gamma(1.-c_sigma,1./c_tau)
  R_ = zeros(K_init)
  V_ = Affinity()
  for k in 1:K_init
    R_[k] = rand(r_dist_init)
    V_[k] = c_alpha/c_beta*ones(n)
    s_min_init = min(s_min_init,R_[k])
  end
  slice_matrix = sparse(I_all,J_all,rand()*s_min_init*ones(length(I_all)),n,n)
  partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect)
end

@showprogress for i in 1:n_iter
  # Update measure
  R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

  # Update partition
  partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect)

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
  for bob in 1:49
    # Update measure
    R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

    # Update partition
    partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect)

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
  sorted_R_ = sort(R_,rev=true)
  for j in 1:min(n_observed,K)
    activities_list[i,j] = sorted_R_[j]
  end

  kappa_list[i] = c_kappa
  sigma_list[i] = c_sigma
  tau_list[i] = c_tau
  alpha_list[i] = c_alpha
  beta_list[i] = c_beta

end


main_dir = pwd()
current_dir = Dates.format(now(),"dd-mm-yy_HH-MM")

results_path = string("results/",data_name,"/",current_dir,"/weighted/variables/")
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
                    "R_",R_,
                    "V_",V_)
cd(main_dir)

#-------------------------------------------------------------------------------
# Plotting results
#-------------------------------------------------------------------------------

results_path = string("results/",data_name,"/",current_dir,"/weighted/img/")
mkpath(results_path)
cd(results_path)
include("src/plot_results.jl")

cd(main_dir)


# Plotting clusters
cd(results_path)
order,clusters = cluster_communities(R_,V_)
spy_sparse_order(Z_,order,1.,directed)
PyPlot.close()
cd(main_dir)



#############################################
# Gibs sample unweighted
#############################################
I_,J_,V_ = findnz(Z_)
V_s = [(v_ > 0) ? 1 : 0 for v_ in V_]
Z_tilde = sparse(I_,J_,V_s,n,n)

# List of the number of active communities
n_active_list = zeros(Int,n_iter)
# List of the top K activites
activities_list = zeros(n_iter,K)

# List of l2 errors for prediction
error_list = zeros(n_iter)
error_mean_list = zeros(n_iter)

kappa_list = zeros(n_iter)
sigma_list = zeros(n_iter)
tau_list = zeros(n_iter)
alpha_list = zeros(n_iter)
beta_list = zeros(n_iter)
s_min_list = zeros(n_iter)



s_min = 0.

println()

if warm_start == false
  partition_ = Factorized{Bool}()
  sentAndReceived_ = Count()
  partition_[1] = copy(Z_tilde) #sparse(I_,J_,V_s,n,n)
  sentAndReceived_[1] = reshape(sum(Z_tilde,1),n) + reshape(sum(Z_tilde,2),n)
else
  K_init = trunc(Int,active_feature_mean(n, c_kappa, c_tau, c_sigma, c_alpha, c_beta))+1
  s_min_init = Inf
  r_dist_init = Gamma(1.-c_sigma,1./c_tau)
  R_ = zeros(K_init)
  V_ = Affinity()
  for k in 1:K_init
    R_[k] = rand(r_dist_init)
    V_[k] = c_alpha/c_beta*ones(n)
    s_min_init = min(s_min_init,R_[k])
  end
  slice_matrix = sparse(I_all,J_all,s_min_init*ones(length(I_all)),n,n)
  partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
end

@showprogress for i in 1:n_iter
  # Update measure
  R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

  # Update partition
  partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect)

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

  for bob in 1:49
    # Update measure
    R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

    # Update partition
    partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect)

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
  sorted_R_ = sort(R_,rev=true)
  for j in 1:min(n_observed,K)
    activities_list[i,j] = sorted_R_[j]
  end

  kappa_list[i] = c_kappa
  sigma_list[i] = c_sigma
  tau_list[i] = c_tau
  alpha_list[i] = c_alpha
  beta_list[i] = c_beta

end

results_path = string("results/",data_name,"/",current_dir,"/unweighted/variables/")
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
                    "R_",R_,
                    "V_",V_)
cd(main_dir)

#-------------------------------------------------------------------------------
# Plotting results
#-------------------------------------------------------------------------------

results_path = string("results/",data_name,"/",current_dir,"/unweighted/img/")
mkpath(results_path)
cd(results_path)
include("src/plot_results.jl")

cd(main_dir)


# Plotting clusters
cd(results_path)
order,clusters = cluster_communities(R_,V_)
spy_sparse_order(Z_,order,1.,directed)
PyPlot.close()
cd(main_dir)
