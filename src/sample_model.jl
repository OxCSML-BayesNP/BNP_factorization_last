include("sparse_format.jl")
include("sampling_tools.jl")
include("plot_tools.jl")

"""
  Expected number of active communities

  Parameters
    n: (int) Number of nodes
    tau,sigma,alpha,beta: (float) Parameters od the model (see notes)
"""
function active_feature_mean(n::Int = 1000,
                            kappa::Float64 = 1.,
                            tau::Float64 = 3.2,
                            sigma::Float64 = 0.5,
                            alpha::Float64 = .1,
                            beta::Float64 = 2.0)
  integral = expectation_gamma(tau,sigma,n*alpha,beta)
  if sigma == 0
    return kappa*( integral - log(tau) )
  else
    return kappa/sigma*( integral - tau^sigma )
  end
end

"""
  Generate non zero adjacency matrix Z^(k) of active community k
  (see notes for more details)

  Parameters:
    r: (float) Activity of the community (r_k in notes)
    v: (n vector) Vector containing the affinity of each node to community k
    truncLevel: (flaot) If different from zero, truncate the affinities smaller than truncLevel

  Returns:
    adjacency_matrix: (nxn array) Adjacency matrix of the k-th community
"""
function active_adjacency(r::Float64,
                          v::Array{Float64,1},
                          truncLevel::Float64 = 0.,
                          maxIter::Int64 = 10^8)
  n = length(v)
  adjacency_matrix = Array{Int}(n,n)
  if truncLevel > 0
    v = [x > truncLevel ? x : 0 for x in v]
  end
  empty = true
  s = sum(v)
  # Sample the total weight of the community
  weight_dist = Poisson(r*s^2)
  weight = 0
  for t in 1:maxIter
    weight = rand(weight_dist)
    if weight > 0
      empty = false
      break
    end
  end
  if empty == false
    p_vect = reshape((1./s^2)*v*transpose(v),n^2)
    mult_dist = Multinomial(weight,p_vect)
    mult = rand(mult_dist)
    adjacency_matrix = reshape(mult,(n,n))
    return adjacency_matrix
  end
  error("Maximal number of iterations reached")
end



"""
  Generate factorized model

  Parameters
    n: Number of points
    tau, sigma, alpha, beta: Parameters of the model (see notes)

  Returns
    K: (int) Number of active communities
    V: (nxK matrix) Matrix of affinities (V[:,k] affinities to the k-th community)
    Z: (nxnxK array) Factorized adjacency matrix (Z[:,:,k] adjacency matrix associated
                    to the k-th community)
"""
function generate_model_(n::Int = 10,
                        kappa::Float64 = 1.,
                        tau::Float64 = .1,
                        sigma::Float64 = 0.25,
                        alpha::Float64 = .5,
                        beta::Float64 = 1.)
  # Number of active features
  active_mean = active_feature_mean(n,kappa,tau,sigma,alpha,beta)
  K = rand(Poisson(active_mean))
  # Affinity matrix
  #V = Array(Float64,(n,K))
  V = Array{Float64}((n,K))
  # Activity of the communities
  #R = Array(Float64,K)
  R = Array{Float64}(K)
  # Factorized adjacency matrix
  Z = init_factorized(n,K)
  # Distribution of the affinities
  affinity_param = alpha*ones(n)
  v_dist = Dirichlet(affinity_param)
  for k in 1:K
    # Sample affinities to community k
    v = rand(v_dist)
    if sigma == 0
      q = x->tau*log( (1+x^2/tau) )/x^2
    else
      q = x->tau/sigma*( (1+x^2/tau)^sigma - 1.)/x^2
    end
    s,niter = rand_rejection(q,n*alpha+2,beta)
    V[:,k] = s*v
    # Sample activity of community k
    scaled_target(x) = (1-exp(-x*s^2))
    scaled_prop(x) = s^(1+sigma)*x^((1+sigma)/2.)
    q(x) = scaled_target(x)/scaled_prop(x)
    r,niter = rand_rejection(q,(1-sigma)/2.,tau)
    R[k] = r
    # Construct adjacency matrix of community k
    Z[k] = sparse(active_adjacency(r,V[:,k]))
  end
  return R,V,Z
end
