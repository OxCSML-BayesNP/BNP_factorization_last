#-------------------------------------------------------------------------------
# Test the code on real data
#     In the 'Data' section, chose the name of the dataset of interest in
#   "data_name". The file must then be in data/$data_name.txt.
#   The results will then be stored in results/$data_name/$results_path/time_of_run
#     Set the parameters of the Gibbs sampler in the 'Set Parameters'
#   section.
#-------------------------------------------------------------------------------

include("src/main.jl")

using ProgressBars
using HDF5, JLD
using DelimitedFiles
using Dates
using ProfileView
using Profile
using ArgParse

Id = SparseArrays.I

# Parse arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "--data_name"
        help = "name of the dataset"
        default = "data_for_kn_small"

    "--result_folder"
        help = "Name of the folder where to store results"
        default = "fixed sigma 0"

    "--const_sigma"
        help = "Set to true to fix sigma to 0"
        action = :store_true

    "--directed"
        help = "Set to true if the dataset is directed"
        arg_type = Bool
        default = true

    "--weighted"
        help = "Set to true if the dataset is weighted"
        arg_type = Bool
        default = true

    "--n_iter"
        help = "Number of iterations"
        arg_type = Int
        default = 2000
end

parsed_args = parse_args(ARGS, s)

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
#data_name = "Email-Enron"
#data_name = "polblogs"
#data_name = "wikipedia_3000"
#data_name = "NIPS234"
#data_name = "deezer_RO_und"
#data_name = "data_for_sigma_800"
#data_name = "sbm_600"
#data_name = "sbm_1200"
data_name = parsed_args["data_name"]

# Result folder name
result_folder = "sigma 025"
result_folder = parsed_args["result_folder"]

# Comments about the run
comments = "Alpha free\n new update of mu tilde \n new measure unobs \n sigma max 0.25"


#-------------------------------------------------------------------------------
# Set parameters
#-------------------------------------------------------------------------------
kappa=1.;tau=5.;sigma=0.;alpha=.5;beta=alpha
warm_start = true
weighted = parsed_args["weighted"]
directed = parsed_args["directed"]

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
# allows to save the current state of the run every 5%. Then use continue_real.jl if the script
# is stopped before the end of the sampler to carry from last save.
monitoring_sigma = true

# Number of iterations for the Gibbs sampler
n_iter = parsed_args["n_iter"]
# Thinning the chain, store only every skip iteration
skip = 50
# Number of iterations as burn-in
burn = floor(Int,2*n_iter/4)
# Number of activities to store
K = 10

# Number of updates of the hyperparameters per iteration
n_steps_hyper = 20

# Hyperparameters for the MH update
prior_params = Dict()
prior_params["kappa"] = (.1,.1)
prior_params["sigma"] = (.1,.1)
prior_params["tau"] = (.1,.1)
prior_params["alpha"] = (.1,.1)
prior_params["beta"] = (1.,1.)

prop_params = Dict()
prop_params["kappa"] = 0.02
prop_params["sigma"] = 0.02
prop_params["tau"] = 0.02
prop_params["alpha"] = 0.02
prop_params["beta"] = 0.02

# Set to true if the parameter is fixed by user (to the initial value)
FIXED_KAPPA = false
FIXED_SIGMA = parsed_args["const_sigma"]
FIXED_TAU = false
FIXED_ALPHA = false
FIXED_BETA = true


#-------------------------------------------------------------------------------
# Importing data
#-------------------------------------------------------------------------------
file_name = string("data/",data_name,".txt")
println(string("Reading the file ",file_name))
@time A = readdlm(file_name,Float64)
println()

println(comments)
println()

# Convert in sparse matrix format
println("Creating sparse matrix")
i_data = [trunc(Int,x) for x in A[:,1]]
j_data = [trunc(Int,x) for x in A[:,2]]
if weighted
  val_data = [trunc(Int,x) for x in A[:,3]]
else
  val_data = ones(Int, length(i_data))
end
n = max(maximum(i_data),maximum(j_data))
sparse_data = sparse(i_data,j_data,val_data,n,n)

# If the data is undirected, keep only below diagonal
if directed==false
  for t in 1:length(i_data)
    i_ = i_data[t]
    j_ = j_data[t]
    if i_ < j_
      if weighted
        sparse_data[j_,i_] += sparse_data[i_,j_]
        sparse_data[i_,j_] = 0
      else
        sparse_data[j_,i_] = 1
        sparse_data[i_,j_] = 0
      end
    end
  end
end

# Make sure there are no self edges if self_edge is set to true
if self_edge == false
  for t in 1:length(i_data)
    i_ = i_data[t]
    j_ = j_data[t]
    if i_ == j_
      sparse_data[i_,j_] = 0
    end
  end
end

sparse_data = dropzeros(sparse_data)

println()

#-------------------------------------------------------------------------------
# Save folder
#-------------------------------------------------------------------------------
# Path to the folder where to store the information
main_dir = pwd()
current_dir = Dates.format(now(),"dd-mm-yy_HH-MM-SS")

# Saving main informations about the run in a txt file
println("Saving in "*current_dir)
println()
results_path = string("results/",data_name,"/",result_folder,'/',current_dir,'/')
mkpath(results_path)
open(results_path*"info.txt","w") do f
  write(f,"Dataset:\n")
  write(f,string("  name = ", data_name,"\n"))
  write(f,string("  number of nodes = ", n,"\n"))
  write(f,string("  number of edges = ", sum(sparse_data),"\n"))
  write(f,string("  directed = ", directed,"\n"))
  write(f,string("  weighted = ", weighted,"\n\n"))

  write(f,"Initial parameters:\n")
  write(f,string("  kappa = ", kappa,"\n"))
  write(f,string("  tau = ", tau,"\n"))
  write(f,string("  sigma = ", sigma,"\n"))
  write(f,string("  alpha = ", alpha,"\n"))
  write(f,string("  beta = ", beta,"\n"))
  write(f,string("  warm_start = ", warm_start,"\n\n"))

  write(f,"Hyper parameters:\n")
  write(f,string("  kappa = ", prior_params["kappa"],"\n"))
  write(f,string("  tau = ", prior_params["tau"],"\n"))
  write(f,string("  sigma = ", prior_params["sigma"],"\n"))
  write(f,string("  alpha = ", prior_params["alpha"],"\n"))
  write(f,string("  beta = ", prior_params["beta"],"\n\n"))

  write(f,string("Number of iterations = ", n_iter*skip,"\n"))

  write(f,string("Comments : ", comments,"\n"))
end


#-------------------------------------------------------------------------------
# Initializing parameters
#-------------------------------------------------------------------------------
n=first(size(sparse_data))

# List of the number of active communities
n_active_list = zeros(Int,n_iter)
# List of the top K activites
activities_list = zeros(n_iter,K)

# Lists of values of the parameters at each step
kappa_list = zeros(n_iter)
sigma_list = zeros(n_iter)
tau_list = zeros(n_iter)
alpha_list = zeros(n_iter)
beta_list = zeros(n_iter)

# Monitoring s_min through the run
s_min_list = zeros(n_iter)


# List of l2 errors for prediction
error_list = zeros(n_iter)
error_mean_list = zeros(n_iter)
error_mean_list_burn = zeros(n_iter)

s_min = 0.

# Initializing variables
R_ = Array{Float64,1}()
V_ = Affinity()

# Initialize clusterings variables
# Initialize index of current clustering
idx_clustering = 1
# Iterations for which we save the clustering
ind_clusterings = zeros(Int,n_clusterings)
# Save the assignation of each node to its cluster
# clusterings[i,clus_idx] = idx of cluster of node i
clusterings = zeros(Int,n,n_clusterings)

# Initialize current values of parameters
c_kappa = kappa
c_sigma = sigma
c_tau = tau
c_alpha = alpha
c_beta = beta

# Debugging variables
plot_true = false
PRINT_ = false

#-------------------------------------------------------------------------------
# Initializing variables for predictions
#-------------------------------------------------------------------------------

# Observed matrix
Z_tilde = copy(sparse_data)

# Select entries to predict
println("Masking entries to predict")
if directed
  n_to_predict = trunc(Int,pred_ratio*(n^2-n))
else
  n_to_predict = trunc(Int,pred_ratio/2*(n^2-n))
end
println("Select $n_to_predict indices to predict")

# First select more than needed entries, since we can't hide
# any entry
if directed
  n_to_predict = trunc(Int,1.15*pred_ratio*n^2)
else
  n_to_predict = trunc(Int,1.15*pred_ratio/2*n^2)
end
I_pred,J_pred = rand(1:n,n_to_predict),rand(1:n,n_to_predict)
if directed == false
  for t in 1:n_to_predict
    i_pred = I_pred[t]
    j_pred = J_pred[t]
    if i_pred > j_pred
      I_pred[t] = j_pred
      J_pred[t] = i_pred
    end
  end
end
@time to_predict = sparse(I_pred,J_pred,ones(Int64,n_to_predict),n,n)
I_pred,J_pred = findnz(to_predict)
n_to_predict = length(I_pred)
v_true = Array{Int64, n_to_predict}
is_test = zeros(Int64,n_to_predict)
println("Mask corresponding entries")
@time for t in 1:n_to_predict
  i_pred = I_pred[t]
  j_pred = J_pred[t]
  if i_pred != j_pred
    if Z_tilde[i_pred,j_pred] == 0 || ( sum(Z_tilde[i_pred,:]) > 1 && sum(Z_tilde[:,j_pred])  > 1 )
      is_test[t] = 1
      v_true[t] = sparse_data[i_pred,j_pred]
      if weighted == false
        v_true[t] = min(v_true[t],1)
      end
      Z_tilde[i_pred,j_pred] = 0
    end
  end
  if sum(is_test) > pred_ratio/2*(n^2-n) && directed == false
    break
  end
  if sum(is_test) > pred_ratio*(n^2-n)
    break
  end
end
n_to_predict = sum(is_test)
to_predict = sparse(I_pred[findall(is_test)], J_pred[findall(is_test)], ones(Int64,n_to_predict),n,n)
I_pred, J_pred = findnz(to_predict)
v_true = (n_to_predict > 0) ? v_true[findall(is_test)] : []
I_tilde,J_tilde,V_tilde = findnz(Z_tilde)
Z_tilde = sparse(I_tilde,J_tilde,V_tilde,n,n)
# Vecor with integer predictions
pred_vect = ones(Float64,n_to_predict)
# Vector of posterior mean
pred_average_vect = zeros(Float64,n_to_predict)
pred_average_vect_burn = zeros(Float64,n_to_predict)

# Sparse matrix of observed entries and the ones to predict
I_all,J_all,V_all = findnz(Z_tilde+to_predict+sparse(Id, n, n))
all_ind_mat = dropzeros(sparse(I_all,J_all,ones(Int,length(I_all)),n,n))
println()


if warm_start == false
  global partition_ = Factorized{Bool}()
  global sentAndReceived_ = Count()
  partition_[1] = Z_tilde
  sentAndReceived_[1] = reshape(sum(Z_tilde,dims=1),n) + reshape(sum(Z_tilde,dims=2),n)
else
  K_init = trunc(Int,active_feature_mean(n, c_kappa, c_tau, c_sigma, c_alpha, c_beta))+1
  local s_min_init = Inf
  r_dist_init = Gamma(1-c_sigma,1/c_tau)
  R_ = zeros(K_init)
  for k in 1:K_init
    R_[k] = rand(r_dist_init)
    V_[k] = c_alpha/c_beta*ones(n)
    s_min_init = min(s_min_init, R_[k])
  end
  slice_matrix = s_min_init*all_ind_mat
  if weighted
    partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
  else
    partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
  end
end

#-------------------------------------------------------------------------------
# Starting the MCMC iterations
#-------------------------------------------------------------------------------
println(string("Starting Gibbs sample with ",n_iter," steps"))

start = time_ns()

Profile.clear()
#@showprogress for i in 1:n_iter
for i in tqdm(1:n_iter)

    #FIXED_SIGMA = (i < n_iter/4)

    global partition_, sentAndReceived_, all_ind_mat
    global c_kappa, c_tau, c_sigma, c_alpha, c_beta
    global n_active_list, activities_list
    global kappa_list, sigma_list, tau_list, alpha_list, beta_list
    global s_min_list, error_list, error_mean_list, error_mean_list_burn
    global pred_average_vect
    global idx_clustering

    # Update measure
    R_,V_,n_observed,slice_matrix,s_min = update_measure(partition_,sentAndReceived_,all_ind_mat,c_kappa,c_tau,c_sigma,c_alpha,c_beta)

    # Update partition
    if weighted
      partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
    else
      partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
    end
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
        if weighted
          partition_,sentAndReceived_ = update_partition(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
        else
          partition_,sentAndReceived_ = update_partition_unweighted(R_,V_,slice_matrix,Z_tilde,to_predict,pred_vect,directed,self_edge)
        end

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
        clusterings[node_idx,idx_clustering] = argmax(weights)
      end
      idx_clustering += 1
    end


    pred_average_vect = (i*pred_average_vect+pred_vect)/(i+1)

    # Compute auc error (l2 error in comments)
    if n_to_predict > 0
        error_list[i] = auc_pr(pred_vect, v_true)#norm(v_true-pred_vect)
        error_mean_list[i] = auc_pr(pred_average_vect,v_true)#norm(v_true-pred_average_vect)
    end
    i_b = i-burn
    #if i_b >= 0
    #  pred_average_vect_burn = (i_b*pred_average_vect_burn+pred_vect)/(i_b+1)
    #  error_mean_list_burn[i] = auc_pr(pred_average_vect_burn,v_true)#norm(v_true-pred_average_vect)
    #end

    print_each = floor(Int,n_iter/20)
    # If monitoring sigma
    if monitoring_sigma && i%print_each == 0
      println(string("Progress: ", 100*i/n_iter,"%"))
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

      #PRINT_ = true
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
elapsed_time = time_ns() - start

#-------------------------------------------------------------------------------
# Saving main variables
#-------------------------------------------------------------------------------

# Saving main informations about the run
results_path = string("results/",data_name,"/",result_folder,'/',current_dir,'/')
mkpath(results_path)
open(results_path*"info.txt","w") do f
  write(f,"Dataset:\n")
  write(f,string("  name = ", data_name,"\n"))
  write(f,string("  number of nodes = ", n,"\n"))
  write(f,string("  number of edges = ", sum(sparse_data),"\n"))
  write(f,string("  directed = ", directed,"\n"))
  write(f,string("  weighted = ", weighted,"\n\n"))

  write(f,"Initial parameters:\n")
  write(f,string("  kappa = ", kappa,"\n"))
  write(f,string("  tau = ", tau,"\n"))
  write(f,string("  sigma = ", sigma,"\n"))
  write(f,string("  alpha = ", alpha,"\n"))
  write(f,string("  beta = ", beta,"\n"))
  write(f,string("  warm_start = ", warm_start,"\n\n"))

  write(f,string("Number of iterations = ", n_iter*skip,"\n"))
  write(f,string("Time  = ", elapsed_time," s \n\n"))

  write(f,string("Comments : ", comments,"\n"))
end

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
