
include("sparse_format.jl")
include("sampling_tools.jl")
include("plot_tools.jl")
include("sample_model.jl")
include("auc.jl")



"""
  Sample the parameters (activity and affinity vector) of an observed community c.
  We will use the emails example developped in the notes to illustrate the parameters

  Parameters
  sentAndReceived: (n vector) sentAndReceived[i] is the number of emails, related to
                              community k, sent and received by i
  tau,sigma,alpha,beta: (float) Parameters od the model (see notes)

  Returns
  r: (flaot) Activity of community c
  v: (n vector) Vector of affinities of each node to community c
"""
function rand_observed_community(sentAndReceived::Array{Int,1},
                                tau::Float64 = .1,
                                sigma::Float64 = 0.25,
                                alpha::Float64 = .5,
                                beta::Float64 = 1.)
  # Number of nodes
  n = length(sentAndReceived)

  # Sample sum_v = |v|_1 from posterior
  # total_weight is n_c in the notes
  total_weight = 1/2*sum(sentAndReceived)
  log_scaled_target(x) = (2*total_weight-2*sigma)*log(x)

  log_scaled_prop(x) = (total_weight-sigma)*log(tau+x^2)
  q(x) = exp(log_scaled_target(x)-log_scaled_prop(x))
  sum_v,niter = try rand_rejection(q,n*alpha+2*sigma,beta)
  catch
    println("Degree of the community $total_weight")
    error("Maximal number of iterations reached")
  end

  # Sample r from posterior (activity of the community)
  r_dist = Gamma(total_weight-sigma,1/(sum_v^2+tau))
  r = rand(r_dist)

  # Sample v from posterior
  affinity_param = alpha*ones(n)+sentAndReceived
  v_dist = Dirichlet(affinity_param)
  v = sum_v*rand(v_dist)

  return r,v,niter
end




"""
  Step 1 from the Gibbs sampler (see notes)
  Sample the partitions for a weighted observation

  Parameters:
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    slice_matrix: (nxn sparse array) Slice variables
    adjacency: (nxn sparse Array) Adajcency matrix of the observed weighted graph

  Returns;
    partition: (factorized) partition[k][i,j] = 1 if there is at least one email sent from
                            i to j (corresponding to community k)
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
"""

function update_partition(activities::Array{Float64,1},
                          affinities::Affinity,
                          slice_matrix::SparseMatrixCSC{Float64,Int},
                          adjacency::SparseMatrixCSC{Int,Int},
                          to_predict::SparseMatrixCSC{Int,Int}=sparse(zeros(Int64,1,1)),
                          pred_vect::Array{Float64,1}=zeros(Float64,1),
                          directed::Bool=true,
                          self_edge::Bool = true)
  # Number of nodes in the graph
  n = first(size(adjacency))
  # Initilalize partition
  partition = Factorized{Bool}()
  partition_I_ = Count()
  partition_J_ = Count()
  # Find non zero entries of the adjacency matrix
  i_idx,j_idx,val = findnz(adjacency)
  # Sample the partition for each non zero entry
  sentAndReceived = Count()
  # Sort activities
  sorted_activities_idx = sortperm(activities,rev=true)
  # Initialize mass functions
  k_p = length(activities)
  mass_min = zeros(k_p)
  p_ = zeros(k_p)
  @inbounds for t in 1:length(i_idx)
    i_ = i_idx[t]
    j_ = j_idx[t]
    z_ij = val[t]
    s_ij = slice_matrix[i_,j_]
    partition_weighted_edge(i_,
                            j_,
                            z_ij,
                            s_ij,
                            activities,
                            affinities,
                            sorted_activities_idx,
                            mass_min,
                            p_,
                            sentAndReceived,
                            partition_I_,
                            partition_J_,
                            directed)
  end
  # Indices of entries to predict
  i_idx_pred,j_idx_pred = findnz(to_predict)
  @inbounds for t in 1:length(i_idx_pred)
    i_ = i_idx_pred[t]
    j_ = j_idx_pred[t]
    s_ij = slice_matrix[i_,j_]
    pred_vect[t] = partition_to_predict_edge(i_,
                                            j_,
                                            s_ij,
                                            activities,
                                            affinities,
                                            sorted_activities_idx,
                                            mass_min,
                                            p_,
                                            sentAndReceived,
                                            partition_I_,
                                            partition_J_,
                                            directed)
  end
  # Predict self edges if not observed
  if self_edge == false
    @inbounds for i_ in 1:n
      if to_predict[i_,i_] == 0
        s_ii = slice_matrix[i_,i_]
        partition_to_predict_edge(i_,
                                  i_,
                                  s_ii,
                                  activities,
                                  affinities,
                                  sorted_activities_idx,
                                  mass_min,
                                  p_,
                                  sentAndReceived,
                                  partition_I_,
                                  partition_J_,
                                  directed)
      end
    end
  end
  for k in keys(sentAndReceived)
    partition[k] = sparse(partition_I_[k],partition_J_[k],trues(length(partition_I_[k])),n,n)
  end
  return partition,sentAndReceived
end





"""
  Step 1 from the Gibbs sampler with unweighted observations (see notes)
  Sample the partitions with binary observations

  Parameters:
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    slice_matrix: (nxn sparse array) Slice variables
    adjacency: (nxn sparse Array) Adajcency matrix of the observed unweighted graph

  Returns;
    partition: (factorized) partition[k][i,j] = 1 if there is at least one email sent from
                            i to j (corresponding to community k)
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
"""

function update_partition_unweighted(activities::Array{Float64,1},
                                    affinities::Affinity,
                                    slice_matrix::SparseMatrixCSC{Float64,Int},
                                    adjacency::SparseMatrixCSC{Int,Int},
                                    to_predict::SparseMatrixCSC{Int,Int}=sparse(zeros(Int64,1,1)),
                                    pred_vect::Array{Float64,1}=zeros(Float64,1),
                                    directed::Bool = true,
                                    self_edge::Bool = true)
  # Number of nodes in the graph
  n = first(size(adjacency))
  # Initilalize partition
  partition = Factorized{Bool}()
  partition_I_ = Count()
  partition_J_ = Count()
  # Find non zero entries og the adjacency matrix
  i_idx,j_idx,val = findnz(adjacency)
  # Sample the partition for each non zero entry
  sentAndReceived = Count()
  # Sort activities
  sorted_activities_idx = sortperm(activities,rev=true)
  # Initialize mass functions
  k_p = length(activities)
  mass_min = zeros(k_p)
  p_ = zeros(k_p)
  @inbounds for t in 1:length(i_idx)
    i_ = i_idx[t]
    j_ = j_idx[t]
    s_ij = slice_matrix[i_,j_]

    partition_unweighted_edge(i_,
                              j_,
                              s_ij,
                              activities,
                              affinities,
                              sorted_activities_idx,
                              mass_min,
                              p_,
                              sentAndReceived,
                              partition_I_,
                              partition_J_,
                              directed)
  end
  # Indices of entries to predict
  i_idx_pred,j_idx_pred = findnz(to_predict)
  @inbounds for t in 1:length(i_idx_pred)
    i_ = i_idx_pred[t]
    j_ = j_idx_pred[t]
    s_ij = slice_matrix[i_,j_]
    pred_edge = partition_to_predict_edge(i_,
                                          j_,
                                          s_ij,
                                          activities,
                                          affinities,
                                          sorted_activities_idx,
                                          mass_min,
                                          p_,
                                          sentAndReceived,
                                          partition_I_,
                                          partition_J_,
                                          directed)
    pred_vect[t] = min(pred_edge,1)
  end
  # Predict self edges if not observed
  if self_edge == false
    @inbounds for i_ in 1:n
      if to_predict[i_,i_] == 0
        s_ii = slice_matrix[i_,i_]

        partition_to_predict_edge(i_,
                                  i_,
                                  s_ii,
                                  activities,
                                  affinities,
                                  sorted_activities_idx,
                                  mass_min,
                                  p_,
                                  sentAndReceived,
                                  partition_I_,
                                  partition_J_,
                                  directed)
      end
    end
  end
  for k in keys(sentAndReceived)
    partition[k] = sparse(partition_I_[k],partition_J_[k],trues(length(partition_I_[k])),n,n)
  end
  return partition,sentAndReceived
end





"""
  Step 2 from Gibbs sampler (see notes)
  Sample the slice variables and the points of the latent communities process
  with large enough activities

  Parameters:
    partition: (factorized) partition[k][i,j] is equal to 1 if there is at least one email
                            sent from i to j
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
    adjacency: (nxn sparse Array) Adajcency matrix of the observed graph
    tau,sigma,alpha,beta: (float) Parameters od the model (see notes)

  Returns:
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    n_observed: (int) Number of observed communities
    slice_matrix: (nxn array) Slice variables
"""
function update_measure(partition::Factorized{Bool},
                        sentAndReceived::Count,
                        adjacency::SparseMatrixCSC{Int,Int},
                        kappa::Float64 = 1.,
                        tau::Float64 = .1,
                        sigma::Float64 = 0.25,
                        alpha::Float64 = .5,
                        beta::Float64 = 1.)
  # First sample the activities and affinity vectors of the observed communities
  new_index = Dict{Int,Int}()
  activities = Array{Float64,1}()
  affinities = Affinity()
  k = 1
  # Initialize slice variables
  n = first(size(adjacency))
  i_idx,j_idx = findnz(adjacency)
  s_val = zeros(length(i_idx))
  for c in keys(sentAndReceived)
    new_index[c] = k
    r_,v_ = rand_observed_community(sentAndReceived[c],tau,sigma,alpha,beta)
    push!(activities,r_)
    i_c_idx, j_c_idx = findnz(partition[c])
    j_ = 0
    row_array = []
    r1 = 0
    r2 = 0
    for t in 1:length(i_c_idx)
      # Find the nzindex of i_, j_ in the sparse matrix
      # adjacency efficiently using the fact that (i_,j_)
      # come ordered
      i_ = i_c_idx[t]
      if j_ != j_c_idx[t]
        j_ = j_c_idx[t]
        r1 = Int(adjacency.colptr[j_])
        r2 = Int(adjacency.colptr[j_+1]-1)
        if (r1 > r2)
          error(string("No corresponding entry ",i_," ",j_))
        end
        row_array = adjacency.rowval[r1:r2]
      end
      t_s = r1 + searchsortedfirst(row_array, i_ ) - 1
      s_ = s_val[t_s]
      if s_ > 0.
        s_val[t_s]= min(s_,r_)
      else
        s_val[t_s] = r_
      end
    end
    affinities[k] = v_
    k += 1
  end
  # For prediction
  for t_s in 1:length(i_idx)
    if s_val[t_s] == 0
      s_val[t_s] = 1
    end
  end
  n_observed = k-1
  # Sample slice variables
  map!(x->x*rand(),s_val,s_val)
  slice_matrix = sparse(i_idx,j_idx,s_val,n,n)
  # Sample the non observed communities
  #expected_Ncomm = active_feature_mean(n,kappa,tau,sigma,alpha,beta)
  s_min = minimum(slice_matrix.nzval)

  # DELETE
  #s_min = maximum([s_min, 1e-12])
  # DELETE

  #var_activities = rnd_GGP_jumps((sigma*expected_Ncomm+kappa*tau^sigma),1.,sigma,tau*s_min) # Test change 08/10/18 (Gamma)
  #v_dist = Dirichlet(alpha*ones(n))
  #for v_act in var_activities
  #  log_scaled_target(x) = sigma*log(tau+x^2)
  #  log_scaled_prop(x) = 2*sigma*log(x)
  #  q(x) = exp(log_scaled_target(x)-log_scaled_prop(x))
  #  sum_v,niter = rand_rejection(q,n*alpha+2*sigma,beta)
  #  if v_act/(sum_v^2+tau) > s_min
  #    push!(activities,v_act/(sum_v^2+tau))
  #    affinities[k] = sum_v*rand(v_dist)
  #    k += 1
  #  end
  #end
  # Compute expected number of iterations with change of variables
  # strategy (see notes)
  expected_Ncomm = active_feature_mean(n,kappa,tau,sigma,alpha,beta)
  kappa_tilde = (sigma*expected_Ncomm+kappa*tau^sigma)
  if abs(sigma) > 1e-8
    change_var_exp = kappa_tilde*upper_inc_gamma(-sigma,tau*s_min)/gamma(1-sigma)
  else
    change_var_exp = kappa_tilde*scp.exp1(tau*s_min)
  end

  # Compute expected number of iterations with rejection strategy
  # (see notes)
  if sigma > 1e-8
    rejection_exp = kappa/(abs(sigma)*gamma(1-sigma)*s_min^sigma)
    #rejection_exp = Inf
  else
    rejection_exp = Inf
  end

  if PRINT_
    println()
    println("Expected numb of iterations change of var = $change_var_exp")
    println("Expected numb of iterations rejection = $rejection_exp")
  end

  if rejection_exp > 5*change_var_exp
    var_activities = rnd_GGP_jumps(kappa_tilde,1.,sigma,tau*s_min) # Test change 08/10/18 (Gamma)
    v_dist = Dirichlet(alpha*ones(n))
    for v_act in var_activities
      log_scaled_target(x) = sigma*log(tau+x^2)
      log_scaled_prop(x) = 2*sigma*log(x)
      q(x) = exp(log_scaled_target(x)-log_scaled_prop(x))
      sum_v,niter = rand_rejection(q,n*alpha+2*sigma,beta)
      if v_act/(sum_v^2+tau) > s_min
        push!(activities,v_act/(sum_v^2+tau))
        affinities[k] = sum_v*rand(v_dist)
        k += 1
      end
    end
  else
    var_activities = rnd_GGP_jumps(kappa,0.,sigma,s_min)
    v_dist = Dirichlet(alpha*ones(n))
    for v_act in var_activities
      p_accept = exp(-n*alpha*log(1+2*v_act*sqrt(tau)/beta))
      if rand(Bernoulli(p_accept)) == 1
        sum_v_dist = Gamma(n*alpha,1/(beta+2*v_act*sqrt(tau)))
        sum_v = rand(sum_v_dist)
        p_accept = exp(-v_act*(sum_v-sqrt(tau))^2)
        if rand(Bernoulli(p_accept)) == 1
          push!(activities,v_act)
          affinities[k] = sum_v*rand(v_dist)
          k += 1
        end
      end
    end
  end
  return activities,affinities,n_observed,slice_matrix,s_min
end

"""
  Step 3 of the Gibbs sampler:
  Update the paramters allowing sigma to be negative


  Warning : Some instability noted we need to address
    If current_alpha is too small (typically < 0.001), then machine precision can lead
    to errors when sampling affinities, therefore we only accept larger alphas
    If current_sigma is too close to zero (|sigma| < 1e-8), then the integration methods
    are instables, leading to abberant behaviors, parameters get stuck to these particular values,
    therefore we set sigma = 0 if |sigma| < 1e-8

  Parameters
    current_kappa,..,current_beta: (float) Current parameters
    prior_params: (Dict) Hyperparameters for the prior distribution
    proposal_params: (Dict) Variances of the poposals
"""
function update_parameters_neg2(current_kappa::Float64,
                          current_sigma::Float64,
                          current_tau::Float64,
                          current_alpha::Float64,
                          current_beta::Float64,
                          prior_params::Dict,
                          proposal_params::Dict,
                          activities::Array{Float64,1},
                          affinities::Affinity,
                          sentAndReceived::Count,
                          s_min::Float64)
  # Number of points and number of jumps
  n_jumps = length(activities)
  n = length(first(affinities)[2])

  # Getting the hyperparameters of the priors
  a_kappa, b_kappa = prior_params["kappa"]
  a_sigma, b_sigma = prior_params["sigma"]
  a_tau, b_tau = prior_params["tau"]
  a_alpha, b_alpha = prior_params["alpha"]
  a_beta, b_beta = prior_params["beta"]

  # Getting hyperparameters of the proposals
  sigma_kappa = proposal_params["kappa"]
  sigma_sigma = proposal_params["sigma"]
  sigma_tau = proposal_params["tau"]
  sigma_alpha = proposal_params["alpha"]
  sigma_beta = proposal_params["beta"]

  # Sample the proposals
  if FIXED_TAU
    prop_tau = current_tau
  else
    tau_dist = LogNormal(log(current_tau),sigma_tau)
    prop_tau = rand(tau_dist)
  end
  if FIXED_SIGMA
    prop_sigma = current_sigma
  else
    transf_sigma = LogNormal(log(1-3*current_sigma),sigma_sigma) # Change scale sigma
    prop_sigma = (1-rand(transf_sigma))/3. # Change scale sigma
  end
  if FIXED_ALPHA
    prop_alpha = current_alpha
  else
    #alpha_dist = truncated(LogNormal(log(current_alpha),sigma_alpha), 0, 1.)
    alpha_dist = LogNormal(log(current_alpha),sigma_alpha)
    prop_alpha = rand(alpha_dist)
  end
  if FIXED_BETA
    prop_beta = current_beta
  else
    beta_dist = LogNormal(log(current_beta),sigma_beta)
    prop_beta = rand(beta_dist)
  end

  # Compute the acceptance probability
  sum_log_aff = sum( [ sum(log.(v)) for (k,v) in affinities ] )
  sum_aff = sum( [ sum(v) for (k,v) in affinities ] )

  log_accept_sigma = a_sigma*(log(1-3*prop_sigma)-log(1-3*current_sigma)) +
                    3*b_sigma*(prop_sigma-current_sigma) -
                    n_jumps*log( gamma(1-prop_sigma)/gamma(1-current_sigma) ) -
                    (prop_sigma-current_sigma)*sum(log.(activities)) # Change scale sigma
  log_accept_tau = a_tau*log(prop_tau/current_tau) -
                  (b_tau+sum(activities))*(prop_tau-current_tau)
  if prop_alpha == current_alpha
    log_accept_alpha = 0.
  # Warning ! Approximation, if alpha is too small (typically around 0.001),
  # then some affiliations are approximated to 0. due to machine precision.
  # If it happens, then only accept a larger alpha
  elseif sum_log_aff == -Inf
    if prop_alpha > current_alpha
      log_accept_alpha = 0
    else
      return current_kappa,current_sigma,current_tau,current_alpha,current_beta
    end
  else
    log_accept_alpha = a_alpha*log(prop_alpha/current_alpha) -
                      (b_alpha-sum_log_aff)*(prop_alpha-current_alpha) -
                      n_jumps*n*log( gamma(prop_alpha)/gamma(current_alpha) )
  end
  log_accept_beta = a_beta*log(prop_beta/current_beta) -
                    (b_beta+sum_aff)*(prop_beta-current_beta)

  # Compute the cross variables part
  log_accept_cross = n_jumps*n*( prop_alpha*log(prop_beta) - current_alpha*log(current_beta) )

  # If n*alpha/beta is too small, then integrating from 0 to Inf can lead to computation errors
  if n*current_alpha/current_beta < 1e7#1e-4
    gamma_dist_c = Gamma(n*current_alpha,1/current_beta)
    x_min_c = quantile(gamma_dist_c,0.00000001)
    x_max_c = cquantile(gamma_dist_c,0.00000001)
  else
    # x_min_c = 0 # Warning
    # x_max_c = Inf # Warning
    gamma_dist_c = Gamma(n*current_alpha,1/current_beta)
    x_min_c = quantile(gamma_dist_c,0.00000001)
    x_max_c = cquantile(gamma_dist_c,0.00000001)
  end
  # If current_sigma is to close to zero, use the approximation current_sigma = 0
  # to avoid computational errors
  if abs(current_sigma) < 1e-4
    #int_current = quadgk(x-> ( log(current_tau+x^2) + scp.exp1((current_tau+x^2)*s_min) )*gamma_pdf(x,n*current_alpha,current_beta),
    #                    x_min_c,x_max_c)[1] - log(current_tau)
    int_current = quadgk(x-> exp( log(log(current_tau+x^2) + scp.exp1((current_tau+x^2)*s_min) ) +
                               log(gamma_pdf(x,n*current_alpha,current_beta)) ),
                      x_min_c,x_max_c)[1] - log(current_tau)
  else
    int_current = quadgk(x-> (current_tau+x^2)^current_sigma*( upper_inc_gamma(-current_sigma,(current_tau+x^2)*s_min)+gamma(1-current_sigma)/current_sigma  )*gamma_pdf(x,n*current_alpha,current_beta),
                        x_min_c,x_max_c)[1] - current_tau^current_sigma*gamma(1-current_sigma)/current_sigma
    #int_current = quadgk(x-> exp(  current_sigma*log(current_tau+x^2) +
    #                            log(upper_inc_gamma(-current_sigma,(current_tau+x^2)*s_min)+gamma(1-current_sigma)/current_sigma  ) +
    #                            log(gamma_pdf(x,n*current_alpha,current_beta)) ),
    #                  x_min_c,x_max_c)[1] - current_tau^current_sigma*gamma(1-current_sigma)/current_sigma
  end


  if n*prop_alpha/prop_beta < 1e7#1e-4
    gamma_dist_p = Gamma(n*prop_alpha,1/prop_beta)
    x_min_p = quantile(gamma_dist_p,0.00000001)
    x_max_p = cquantile(gamma_dist_p,0.00000001)
  else
    # x_min_p = 0 # Warning
    # x_max_p = Inf # Warning
    gamma_dist_p = Gamma(n*prop_alpha, 1/prop_beta)
    x_min_p = quantile(gamma_dist_p, 0.00000001)
    x_max_p = cquantile(gamma_dist_p, 0.00000001)
  end
  # If prop_sigma is to close to zero, use the approximation prop_sigma = 0
  # to avoid computational errors
  if abs(prop_sigma) < 1e-4
    #int_prop = quadgk(x-> ( log(prop_tau+x^2) + scp.exp1((prop_tau+x^2)*s_min) )*gamma_pdf(x,n*prop_alpha,prop_beta),
    #                    x_min_p,x_max_p)[1] - log(prop_tau)
    int_prop = quadgk(x-> exp( log(log(prop_tau+x^2) + scp.exp1((prop_tau+x^2)*s_min) ) +
                               log(gamma_pdf(x,n*prop_alpha,prop_beta)) ),
                      x_min_p,x_max_p)[1] - log(prop_tau)
  else
    int_prop = quadgk(x-> (prop_tau+x^2)^prop_sigma*( upper_inc_gamma(-prop_sigma,(prop_tau+x^2)*s_min)+gamma(1-prop_sigma)/prop_sigma  )*gamma_pdf(x,n*prop_alpha,prop_beta),
                      x_min_p,x_max_p)[1] - prop_tau^prop_sigma*gamma(1-prop_sigma)/prop_sigma
    #int_prop = quadgk(x-> exp(  prop_sigma*log(prop_tau+x^2) +
    #                            log(upper_inc_gamma(-prop_sigma,(prop_tau+x^2)*s_min)+gamma(1-prop_sigma)/prop_sigma ) +
    #                            log(gamma_pdf(x,n*prop_alpha,prop_beta)) ),
    #                  x_min_p,x_max_p)[1] - prop_tau^prop_sigma*gamma(1-prop_sigma)/prop_sigma
  end


  log_accept_cross += (n_jumps+a_kappa)*(log(int_current/gamma(1-current_sigma)+b_kappa) -
                      log(int_prop/gamma(1-prop_sigma)+b_kappa))
  # Probability of accpetance
  log_accept = log_accept_sigma + log_accept_tau +
              log_accept_alpha + log_accept_beta + log_accept_cross
  p_accept = exp(log_accept)

  # If p_accept == NaN, track error
  if p_accept != p_accept
    println(sum_log_aff)
    println(int_current)
    println(int_prop)
    println(p_accept)
    println(prop_sigma)
    error()
  end

  # Draw a Bernoulli to decide if we accept the proposal
  p = min(p_accept,1.)
  b_dist = Bernoulli(p)
  b_rand = rand(b_dist)
  if b_rand == 1
    int_current = int_prop
    current_sigma = prop_sigma
    current_tau = prop_tau
    current_alpha = prop_alpha
    current_beta = prop_beta
  end
  current_kappa = rand(Gamma(n_jumps+a_kappa,1/(int_current/gamma(1-current_sigma)+b_kappa)))

  if current_kappa > 5000
    println(current_kappa)
    println(current_sigma)
    println(current_tau)
    println(current_alpha)
    println(current_beta)
    println(int_current)
    println(n_jumps)
  end

  return current_kappa,current_sigma,current_tau,current_alpha,current_beta
end



"""
  Partition the edges between a pair of nodes for the weighted graph

  Prameters:
    i_, j_, z_ij : (int) Multi edge to partition
    s_ij: (float) slice variable
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    sorted_activities_idx: (array) sorted order of activities
    mass_min, p_: (array) used to sample the partition
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
    partition_I_, partition_J_: (count) Changed in place, used to construct partition
"""
function partition_weighted_edge(i_::Int64,
                                j_::Int64,
                                z_ij::Int64,
                                s_ij::Float64,
                                activities::Array{Float64,1},
                                affinities::Affinity,
                                sorted_activities_idx::Array{Int64,1},
                                mass_min::Array{Float64,1},
                                p_::Array{Float64,1},
                                sentAndReceived::Count,
                                partition_I_::Count,
                                partition_J_::Count,
                                directed::Bool=true)

  # Sample the index of the active community (for edge (i,j)) with
  # the smallest activity
  prev_prob = 0.
  current_state = 0.
  sum_mass_min = 0.
  idx_L = 0
  # Normalize the weights since for large
  max_p_k = 0.
  k_p = length(mass_min)
  for l in 1:k_p
    k = sorted_activities_idx[l]
    r = activities[k]
    if r > s_ij
      affinities_k = affinities[k]
      p_k = r*affinities_k[i_]*affinities_k[j_]
      if directed == false && i_ != j_
        p_k = 2*p_k
      end
      max_p_k = max(max_p_k,p_k)
      p_[k] = p_k
      idx_L = l
    else
      p_[k] = 0.
      mass_min[k] = 0.
    end
  end
  # Compute the probability mass function for the smallest community of (i_,j_)
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    r = activities[k]
    if r > s_ij
      p_k = p_[k]/max_p_k
      current_state += p_k
      current_prob = current_state^z_ij
      mass_min[k] = ( current_prob - prev_prob )/r
      sum_mass_min += mass_min[k]
      prev_prob = current_prob
    end
  end
  if idx_L == 0
    error("Slice variable too large in ($i_,$j_)")
  end
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    mass_min[k] = mass_min[k]/sum_mass_min
  end

  c_min = try
      rand(Categorical(mass_min))
  catch
    -1
  end
  if c_min == -1
    println(k_p)
    println(mass_min)
    println(p_)
    println(z_ij)
    println(string(i_,", ",j_))
    error("Computational error while sampling smallest community of ($i_,$j_)")
  end
  if haskey(sentAndReceived,c_min) == false
    sentAndReceived[c_min] = zeros(n)
    partition_I_[c_min] = []
    partition_J_[c_min] = []
  end
  # Add a nz entry (i_,j_) in partition[c_min] efficiently using the
  # fact that the entries come sorted
  push!(partition_I_[c_min],i_)
  push!(partition_J_[c_min],j_)
  # Sample the number of emails from i to j corresponding to community c_min
  r_c_min = activities[c_min]
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    r_k = activities[k]
    if r_k < r_c_min
      p_[k] = 0.
    end
  end
  #z_min = rand(Truncated(Binomial(z_ij,p_[c_min]/sum(p_)),0.5,z_ij+1))
  z_min = rand_ztbinomial(z_ij,p_[c_min]/sum(p_))
  if z_min > z_ij
    error(string("Error when sampling the zero truncated binomial, ",z_min, "/",z_ij," ",(p_[c_min]/sum(p_))))
  end
  sentAndReceived[c_min][i_] += z_min
  sentAndReceived[c_min][j_] += z_min
  p_[c_min] = 0.
  # Sample the partition of the remaining emails
  if sum(p_) > 0. && z_min < z_ij
    partition_dist = Multinomial(z_ij-z_min,p_/sum(p_))
    part = rand(partition_dist)
    for l in 1:idx_L
      k = sorted_activities_idx[l]
      if part[k] > 0
        if haskey(sentAndReceived,k) == false
          sentAndReceived[k] = zeros(n)
          partition_I_[k] = []
          partition_J_[k] = []
        end
        push!(partition_I_[k],i_)
        push!(partition_J_[k],j_)
        sentAndReceived[k][i_] += part[k]
        sentAndReceived[k][j_] += part[k]
      end
    end
  end

end



"""
  Partition the edges between a pair of nodes for the weighted graph

  Prameters:
    i_, j_: (int) Multi edge to partition
    s_ij: (float) slice variable
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    sorted_activities_idx: (array) sorted order of activities
    mass_min, p_: (array) used to sample the partition
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
    partition_I_, partition_J_: (count) Changed in place, used to construct partition
"""
function partition_unweighted_edge(i_::Int64,
                                j_::Int64,
                                s_ij::Float64,
                                activities::Array{Float64,1},
                                affinities::Affinity,
                                sorted_activities_idx::Array{Int64,1},
                                mass_min::Array{Float64,1},
                                p_::Array{Float64,1},
                                sentAndReceived::Count,
                                partition_I_::Count,
                                partition_J_::Count,
                                directed::Bool=true)
  # Sample the index of the active community (for edge (i,j)) with
  # the smallest activity
  sum_mass_min = 0.
  sum_p_ = 0.
  idx_L = 0
  max_log_mass_min = -Inf
  k_p = length(mass_min)
  for l in 1:k_p
    k = sorted_activities_idx[l]
    r = activities[k]
    if r > s_ij
      affinities_k = affinities[k]
      p_k = r*affinities_k[i_]*affinities_k[j_]
      if directed == false && i_ != j_
        p_k = 2*p_k
      end
      p_[k] = p_k
      if p_k > 700
        # If p_k is too large for machine precision (exp(p_k) = Inf),
        # we use the approximation log(exp(p_k)-1) = p_k
        mass_min[k] = sum_p_ + p_k - log(r)
      else
        mass_min[k] = sum_p_ + log((exp(p_k)-1)/r)
      end
      max_log_mass_min = max(mass_min[k],max_log_mass_min)
      if mass_min[k] != mass_min[k] || mass_min[k] == Inf
        println(affinities_k[i_])
        println(affinities_k[j_])
        println(sum_p_)
        println(p_k)
        error(r)
      end
      sum_p_ += p_k
      idx_L = l
    else
      p_[k] = 0.
      mass_min[k] = 0.
    end
  end
  if idx_L == 0
    error("Slice variable too large in ($i_,$j_)")
  end
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    mass_min[k] = exp(mass_min[k]-max_log_mass_min)
    sum_mass_min += mass_min[k]
  end
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    mass_min[k] = mass_min[k]/sum_mass_min
  end

  c_min = try
      rand(Categorical(mass_min))
  catch
    -1
  end
  if c_min == -1
    println(k_p)
    println(sum_p_)
    println(sum_mass_min)
    println(max_log_mass_min)
    println(mass_min)
    error()
  end
  if haskey(sentAndReceived,c_min) == false
    sentAndReceived[c_min] = zeros(n)
    partition_I_[c_min] = []
    partition_J_[c_min] = []
  end
  # Add a nz entry (i_,j_) in partition[c_min] efficiently using the
  # fact that the entries come sorted
  push!(partition_I_[c_min],i_)
  push!(partition_J_[c_min],j_)
  #partition[c_min][i_,j_] |= true
  # Sample the number of emails from i to j corresponding to community c_min
  r_c_min = activities[c_min]
  z_min = rand_ztpoisson(p_[c_min])
  if z_min == 0
    error(string("Error when sampling the zero truncated Poisson, ",z_min, "/",(p_[c_min])))
  end

  sentAndReceived[c_min][i_] += z_min
  sentAndReceived[c_min][j_] += z_min

  p_[c_min] = 0.
  # Sample the partition of the remaining emails
  if idx_L > 1
    for l in 1:(idx_L-1)
      k = sorted_activities_idx[l]
      part_k = rand(Poisson(p_[k]))
      if part_k > 0
        if haskey(sentAndReceived,k) == false
          sentAndReceived[k] = zeros(n)
          partition_I_[k] = []
          partition_J_[k] = []
        end
        push!(partition_I_[k],i_)
        push!(partition_J_[k],j_)

        sentAndReceived[k][i_] += part_k
        sentAndReceived[k][j_] += part_k

      end
    end
  end
end





"""
  Predict and partition an unobserved edge between

  Prameters:
    i_, j_ : (int) Multi edge to partition
    s_ij: (float) slice variable
    activities: (array) Jumps of G+ (see notes, the observed communities come forst)
    affinities: (affinity) v[k] is the affinity vector of each node to community k
    sorted_activities_idx: (array) sorted order of activities
    mass_min, p_: (array) used to sample the partition
    sentAndReceived: (count) sentAndReceived[k][i] is the number of emails, related to
                            community k, sent and received by i
    partition_I_, partition_J_: (count) Changed in place, used to construct partition
"""
function partition_to_predict_edge(i_::Int64,
                                  j_::Int64,
                                  s_ij::Float64,
                                  activities::Array{Float64,1},
                                  affinities::Affinity,
                                  sorted_activities_idx::Array{Int64,1},
                                  mass_min::Array{Float64,1},
                                  p_::Array{Float64,1},
                                  sentAndReceived::Count,
                                  partition_I_::Count,
                                  partition_J_::Count,
                                  directed::Bool=true)
  # Sample the index of the active community (for edge (i,j)) with
  # the smallest activity
  sum_mass_min = 0.
  sum_p_ = 0.
  idx_L = 0
  max_log_mass_min = -Inf
  k_p = length(mass_min)
  for l in 1:k_p
    k = sorted_activities_idx[l]
    r = activities[k]
    if r > s_ij
      affinities_k = affinities[k]
      p_k = r*affinities_k[i_]*affinities_k[j_]
      if directed == false && i_ != j_
        p_k = 2*p_k
      end
      p_[k] = p_k
      if p_k > 700
        # If p_k is too large for machine precision (exp(p_k) = Inf),
        # we use the approximation log(exp(p_k)-1) = p_k
        mass_min[k] = sum_p_ + p_k - log(r)
      elseif p_k < 1e-13
        # If p_k is too small for machine precision log((exp(p_k)-1)) = -Inf
        # we use the approximation exp(p_k)-1 = p_k
        mass_min[k] = sum_p_ + log(affinities_k[i_]) + log(affinities_k[j_]) #log(p_k) - log(r)
      else
        mass_min[k] = sum_p_ + log((exp(p_k)-1)/r)
      end
      max_log_mass_min = max(mass_min[k],max_log_mass_min)
      if mass_min[k] != mass_min[k] || mass_min[k] == Inf
        println(affinities_k[i_])
        println(affinities_k[j_])
        println(sum_p_)
        println(p_k)
        error(r)
      end
      sum_p_ += p_k
      idx_L = l
    else
      p_[k] = 0.
      mass_min[k] = 0.
    end
  end
  if idx_L == 0
    return 0.
  end
  if max_log_mass_min == -Inf
    return 0.
  end
  for l in 1:idx_L
    k = sorted_activities_idx[l]
    mass_min[k] = exp(mass_min[k]-max_log_mass_min)
    sum_mass_min += mass_min[k]
  end

  # Draw Bernoulli to decide wether there is an edge or not
  if max_log_mass_min >= 0
    p_edge_nzero = exp(-sum_p_-max_log_mass_min)/( exp(-sum_p_-max_log_mass_min) + sum_mass_min )
  else
    p_edge_nzero = 1/( 1. + sum_mass_min*exp(sum_p_ + max_log_mass_min) )
  end
  is_edge_nzero = try rand(Bernoulli(p_edge_nzero))
  catch
    #println(p_)
    #println(mass_min)
    println(exp(-sum_p_-max_log_mass_min))
    println(s_ij)
    println(max_log_mass_min)
    println(sum_mass_min)
    println(p_edge_nzero)
    println("to predict")
    return 0. # TEST DELETE
    error()
  end
  if is_edge_nzero == 1
    return 0.
    #return 1-p_edge_nzero # TEST
  end

  for l in 1:idx_L
    k = sorted_activities_idx[l]
    mass_min[k] = mass_min[k]/sum_mass_min
  end
  c_min = try
      rand(Categorical(mass_min))
  catch
    -1
  end
  if c_min == -1
    println(k_p)
    println(sum_p_)
    println(sum_mass_min)
    println(max_log_mass_min)
    println(mass_min)
    return 0. # TEST DELETE
    error()
  end
  if haskey(sentAndReceived,c_min) == false
    sentAndReceived[c_min] = zeros(n)
    partition_I_[c_min] = []
    partition_J_[c_min] = []
  end
  # Add a nz entry (i_,j_) in partition[c_min] efficiently using the
  # fact that the entries come sorted
  push!(partition_I_[c_min],i_)
  push!(partition_J_[c_min],j_)
  # Sample the number of emails from i to j corresponding to community c_min
  r_c_min = activities[c_min]
  z_min = rand_ztpoisson(p_[c_min])
  if z_min == 0
    error(string("Error when sampling the zero truncated Poisson, ",z_min, "/",(p_[c_min])))
  end
  predicted_edge = z_min

  sentAndReceived[c_min][i_] += z_min
  sentAndReceived[c_min][j_] += z_min

  p_[c_min] = 0.
  # Sample the partition of the remaining emails
  if idx_L > 1
    for l in 1:(idx_L-1)
      k = sorted_activities_idx[l]
      part_k = rand(Poisson(p_[k]))
      if part_k > 0
        if haskey(sentAndReceived,k) == false
          sentAndReceived[k] = zeros(n)
          partition_I_[k] = []
          partition_J_[k] = []
        end
        push!(partition_I_[k],i_)
        push!(partition_J_[k],j_)
        predicted_edge += part_k

        sentAndReceived[k][i_] += part_k
        sentAndReceived[k][j_] += part_k

      end
    end
  end
  return predicted_edge+0.
end





"""
  Test Sample unobserved communities:



  Parameters
    s_min: (float)
    n: (int)
    kappa,..,beta: (float) Current Parameters
"""
function sample_unobs_measure(n::Int64 = 10,
                              s_min::Float64 = 1e-8,
                              kappa::Float64 = 1.,
                              tau::Float64 = .1,
                              sigma::Float64 = 0.25,
                              alpha::Float64 = .5,
                              beta::Float64 = 1.)


  expected_Ncomm = active_feature_mean(n,kappa,tau,sigma,alpha,beta)
  kappa_tilde = (sigma*expected_Ncomm+kappa*tau^sigma)
  if abs(sigma) > 1e-8
    change_var_exp = kappa_tilde*upper_inc_gamma(-sigma,tau*s_min)/gamma(1-sigma)
  else
    change_var_exp = kappa_tilde*scp.exp1(tau*s_min)
  end

  # Compute expected number of iterations with rejection strategy
  # (see notes)
  if sigma > 1e-8
    rejection_exp = kappa/(abs(sigma)*gamma(1-sigma)*s_min^sigma)
    #rejection_exp = Inf
  else
    rejection_exp = Inf
  end


  println()
  println("Expected numb of iterations change of var = $change_var_exp")
  println("Expected numb of iterations rejection = $rejection_exp")


  # Change of variables
  k_change = 0
  sum_change = 0.
  var_activities = rnd_GGP_jumps(kappa_tilde,1.,sigma,tau*s_min) # Test change 08/10/18 (Gamma)
  v_dist = Dirichlet(alpha*ones(n))
  for v_act in var_activities
    log_scaled_target(x) = sigma*log(tau+x^2)
    log_scaled_prop(x) = 2*sigma*log(x)
    q(x) = exp(log_scaled_target(x)-log_scaled_prop(x))
    sum_v,niter = rand_rejection(q,n*alpha+2*sigma,beta)
    if v_act/(sum_v^2+tau) > s_min
      k_change += 1
      sum_change += v_act/(sum_v^2+tau)
    end
  end

  # Rejection
  k_reject = 0
  sum_reject = 0.
  var_activities = rnd_GGP_jumps(kappa,0.,sigma,s_min)
  v_dist = Dirichlet(alpha*ones(n))
  for v_act in var_activities
    p_accept = exp(-n*alpha*log(1+2*v_act*sqrt(tau)/beta))
    if rand(Bernoulli(p_accept)) == 1
      sum_v_dist = Gamma(n*alpha,1/(beta+2*v_act*sqrt(tau)))
      sum_v = rand(sum_v_dist)
      p_accept = exp(-v_act*(sum_v-sqrt(tau))^2)
      if rand(Bernoulli(p_accept)) == 1
        k_reject += 1
        sum_reject += v_act
      end
    end
  end

  k_reject_2 = 0
  sum_reject_2 = 0.
  sum_v_dist = Gamma(n*alpha,1/beta)
  for v_act in var_activities
    sum_v = rand(sum_v_dist)
    p_accept = exp(-v_act*(tau+sum_v^2))
    if rand(Bernoulli(p_accept)) == 1
      k_reject_2 += 1
      sum_reject_2 += v_act
    end
  end

  println()
  println("k change of var = $k_change")
  println("k rejection = $k_reject")
  println("k rejection 2 = $k_reject_2")

  println()
  println("sum change of var = $sum_change")
  println("sum rejection = $sum_reject")
  println("sum rejection 2 = $sum_reject_2")

  k_pareto = length(var_activities)
  sum_pareto = sum(var_activities)
  exp_sum_pareto = kappa*s_min^(1-sigma)/(gamma(1-sigma)*(1-sigma))
  println()
  println("k pareto = $k_pareto")
  println("sum pareto = $sum_pareto")
  println("exp sum pareto = $exp_sum_pareto")
end

"""
  Test rand community

  Parameters
  sentAndReceived: (n vector) sentAndReceived[i] is the number of emails, related to
                              community k, sent and received by i
  tau,sigma,alpha,beta: (float) Parameters od the model (see notes)


"""
function sample_obs_community(n::Int = 1000,
                            total_weight::Int = 10000,
                            tau::Float64 = .1,
                            sigma::Float64 = 0.25,
                            alpha::Float64 = .5,
                            beta::Float64 = 1.)


  # Condition on sum_v strategy
  log_scaled_target(x) = (2*total_weight-2*sigma)*log(x)
  log_scaled_prop(x) = (total_weight-sigma)*log(tau+x^2)
  q(x) = exp(log_scaled_target(x)-log_scaled_prop(x))
  sum_v,niter_cond = try rand_rejection(q,n*alpha+2*sigma,beta)
  catch
    println("Degree of the community $total_weight")
    error("Maximal number of iterations reached")
  end

  # Sample r from posterior (activity of the community)
  r_dist = Gamma(total_weight-sigma,1/(sum_v^2+tau))
  r_cond = rand(r_dist)
  sum_v_cond = sum_v

  # 2D Rejection strategy
  niter_rejec = 0
  r_bar_dist = Gamma(total_weight-sigma,1)
  sum_v_dist = Gamma(n*alpha+2*sigma,1/beta)
  r_rejec = 0.
  sum_v_rejec = 0.
  while true
    niter_rejec += 1
    sum_v = rand(sum_v_dist)
    r_bar = rand(r_bar_dist)
    if rand(Bernoulli(exp(-tau*r_bar/sum_v^2))) == 1
      r_rejec = r_bar/sum_v^2
      sum_v_rejec = sum_v
      break
    end
  end


  println()
  println("Conditioning n iter = $niter_cond")
  println("Rejection n iter = $niter_rejec")

  println()
  println("Conditioning r = $r_cond")
  println("Rejection r = $r_rejec")

  println()
  println("Conditioning sum v = $sum_v_cond")
  println("Rejection sum v = $sum_v_rejec")
end
