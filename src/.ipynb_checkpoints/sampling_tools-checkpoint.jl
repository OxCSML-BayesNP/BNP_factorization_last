using Distributions
using PyCall
using QuadGK
using SpecialFunctions
#@pyimport scipy.special as scp
scp = pyimport("scipy.special")

"""
  Logarithm of a multinomial beta function

  Paramers:
    alpha: (array) Point where to compute the multinomial beta
"""
function lbeta_mult(alpha::Array{Float64,1})
  log_beta = -lgamma(sum(alpha))
  for a in alpha
    if a >= 10.
      log_beta += lgamma(a)
    else
      log_beta += log(gamma(a))
    end
  end
  return log_beta
end


"""
  Compute the expectation
    E[log(tau+V^2)] if sigma == 0
    E[(tau+V^2)^sigma] otherwise
  where V is a Gamma(alpha,beta) random variable

  Parameters
    tau,sigma: (float) parameters of the function to integrate
    alpha,beta: (float) Parameters of the Gamma

"""
function expectation_gamma(tau::Float64,
                          sigma::Float64,
                          alpha::Float64,
                          beta::Float64)
  # If the distributuin of V is very concentrated around 0, taking 0 and Inf as
  # boundaries for the integral can lead to errors
  if alpha/beta > 1e10#1e-2
    x_min = 0
    x_max = Inf
  else
    gamma_dist_1 = Gamma(alpha + 2*sigma, 1/beta)
    gamma_dist_2 = Gamma(alpha, 1/beta)
    x_min = min(quantile(gamma_dist_1, 0.00001), quantile(gamma_dist_2, 0.00001))
    x_max = max(cquantile(gamma_dist_1, 0.00001), cquantile(gamma_dist_2, 0.00001))
  end
  # if sigma is too close to zero, ue the approximation sigma = 0 to avoid computational
  # errors
  if abs(sigma) < 1e-4
    integral = quadgk(x->log(tau+x^2)*gamma_pdf(x,alpha,beta),x_min,x_max)[1]
  else
    integral = quadgk(x->(tau+x^2)^sigma*gamma_pdf(x,alpha,beta),x_min,x_max)[1]
  end
  return integral
end

"""
  Compute the pdf at x of a Gamma(alpha,beta)

  Prameters
    x: (float) Point where to compute the pdf
    alpha: (float) Shape parameter of the Gamma
    beta: (float) Scale parameter
"""
function gamma_pdf(x::Float64,
                  alpha::Float64 = 0.1,
                  beta::Float64 = 1.)
  return pdf(Gamma(alpha, 1/beta), x)
end


"""
  Upper incomplete gamma function for Gamma(s,x)
  for s > -1

  Parameters:
    s,x: (float) Prameters of the Gamma,
                see https://en.wikipedia.org/wiki/Incomplete_gamma_function
"""
function upper_inc_gamma(s::Float64,
                        x::Float64)
  # We use the cdf of the Gamma distribution available in
  # Distributions.jl
  if s <= -1
    error("The parameter s = $s must be larger than -1,")
  end
  if s < 1e-6
    gamma_dist = Gamma(1+s,1.)
    return -1/s*(x^s*exp(-x)+gamma(1+s)*(cdf(gamma_dist,x)-1))
  else
    gamma_dist = Gamma(s,1.)
    return gamma(s)*(1-cdf(gamma_dist,x))
  end
end

"""
  Rejection sampling using Gamma distribution as proposal

  Parameters:
    q: (function) Scaled target distribution divided by proposal (such that q = target/(M*proposal) <= 1)
    alpha,beta: (floats) Parameters of the proposal (Gamma distributed)
    maxIter: (int) Maximal number of iteration before raising error
"""
function rand_rejection(q::Function,
                        alpha::Float64 = .1,
                        beta::Float64 = 2.,
                        maxIter::Int64 = 10^9)
  prop_dist = Gamma(alpha, 1/beta)
  for t in 1:maxIter
    x_prop = rand(prop_dist)
    u = rand(Uniform())
    if u < q(x_prop)
      return x_prop,t
    end
  end
  error("Maximal number of iterations reached")
end

"""
  Sample the jumps of a truncated Generalized Gamma process using
  Pareto proposals for small jumps and adaptive thinning for large ones

  Reference:
    S. Favaro and Y.W. Teh. MCMC for normalized random measure mixture models

  Parameters
    kappa,tau,sigma: (float) Prameters of the GGP (see notes)
    s: (float) Truncation level
    maxiter: (int) Maximal number of iterations (maximal number of proposals)

  Returns:
    N: (array) Jumps of the GGP
"""
function rnd_GGP_jumps(kappa::Float64 = 1.,
                      tau::Float64 = 3.2,
                      sigma::Float64 = 0.5,
                      s::Float64 = 1e-6,
                      paretoTrunc::Float64 = 1e-6,
                      maxIter::Int = 10^8 )
  # Machine precision can cause kappa < 0 in some cases
  if kappa < 0
    println()
    if kappa >-1e-14
      println("kappa is approximated to 0")
      kappa = 0
    else
      println("kappa = ",kappa)
      error("kappa has to be positive")
    end
  end
  # Jumps of the GGP
  N = Array{Float64,1}()
  count = 0
  # Case when the CRM is finite activity (sigma < 0)
  if sigma < 0.
    rate = -kappa/(sigma*tau^(-sigma))
    Njumps = rand(Poisson(rate))
    untrunc_N = rand(Gamma(-sigma,tau),Njumps)
    for r in untrunc_N
      if r >= s
        push!(N,r)
      end
    end
    return N
  end
  # Case when sampling from trucated pareto
  if tau == 0
    paretoTrunc = Inf
  end
  T = paretoTrunc
  # Sample with pareto proposal on [s,T]
  if s < T
    if sigma == 0
      Njumps_pareto = kappa*log(T/s)
    else
      Njumps_pareto = exp( log(kappa)-log(sigma)-log(gamma(1-sigma)) +
                            log(s^(-sigma)-T^(-sigma)) )
    end
    Njumps_pareto = rand(Poisson(Njumps_pareto))
    if Njumps_pareto > maxIter
      println(exp( log(kappa)-log(sigma)-log(gamma(1-sigma)) +
                            log(s^(-sigma)-T^(-sigma)) ))
      println(Njumps_pareto)
      error("Truncation level too small")
    end
    for i in 1:Njumps_pareto
      if sigma == 0
        log_r = rand()*log(T/s)+log(s)
      else
        log_r = -1/sigma*log( s^(-sigma) - (s^(-sigma) - T^(-sigma))*rand() )
      end
      if log(rand()) < -tau*exp(log_r)
        push!(N,exp(log_r))
      end
    end
    count = Njumps_pareto
  end
  if paretoTrunc == Inf
    return N
  end
  # Minimal value of the next jump, initialized at
  # truncation level
  t = max(s,paretoTrunc)
  # Used to check if the algorithm stopped because of maxIter
  completed = false
  # Lower bound of the expected number of proposals
  if sigma > 1e-3
    expected_Njumps = exp( log(kappa)-log(sigma)-log(gamma(1-sigma))- sigma*log(s))
  else
    expected_Njumps = -kappa*log(s)
  end
  # Check if the expected number of proposals is larger than
  # maxIter
  if expected_Njumps > maxIter-count
    error("Expected number of proposals larger than maxIter")
  end
  # Propose points from a truncated exponential Poisson process
  log_cst = log(kappa) - log(gamma(1-sigma)) - log(tau)
  for i in (count+1):maxIter
    log_r = log(-log(rand()))
    log_G = log_cst - (1+sigma) * log(t) - tau*t
    if log_r > log_G
      completed = true
      break
    end
    t_new = t - log(1 - exp(log_r-log_G))/tau
    if log(rand()) < (1+sigma) * (log(t)-log(t_new))
      push!(N,t_new)
    end
    t = t_new
  end
  if completed == false
    error("Maximal number of iterations reached")
  end
  return N
end


"""
  Sample from the zero truncated Binomial distribution

  Parameters:
    n_trials: (int) Number of trials
    p: (float) Probability of success
"""
function rand_ztbinomial(n_trials::Int64,
                        p::Float64)
  # If p is smaller than machine precision, rounding will produce
  # errors, so here we chose to make the approximation that the zero
  # truncated binomial will almost surely be equal to 1
  if p < 2e-16 && n < 1000
    return 1
  end
  # Compute the probability of 0
  p_0 = (1-p)^n_trials
  # If the probability of 0 is small enough,
  # use the native truncate.jl
  if p_0 == 0#(1-p_0) > 1e-1
    if p == 1.
      return n_trials
    end
    binomial_dist = Binomial(n_trials,p)
    return rand(Truncated(binomial_dist, 0.5, n_trials))
  # Otherwise, use an inversion algorithm
  else
    u = (1-p_0)*rand()
    zt_cdf = 0.
    b_pdf = p_0
    k = 0.
    while u > zt_cdf && k < n_trials
      k += 1.
      b_pdf *= p*(n_trials+1-k)/((1-p)*k)
      zt_cdf += b_pdf
    end
    return max(Int(k),1)
  end
end


"""
  Sample from the zero truncated Poisson distribution

  Parameters:
    p: (float) Probability of success
"""
function rand_ztpoisson(p::Float64)
  # If p is smaller than machine precision, rounding will produce
  # errors, so here we chose to make the approximation that the zero
  # truncated Poisson will almost surely be equal to 1
  if p < 2e-16
    return 1
  end
  poisson_dist = Poisson(p)
  # If the probability of 0 is small enough,
  # use the native truncate.jl
  if p > 1.
    while true
      res = rand(poisson_dist)
      if res > 0
        return res
      end
    end
  # Otherwise, use a rejection algorithm
  else
    while true
      u = rand()
      res = rand(poisson_dist)
      if u < 1/(res+1)
        return res + 1
      end
    end
  end
end



"""
  Sample from the zero truncated Poisson distribution

  Parameters:
    p: (float) Probability of success
"""
function rand_ztpoisson2(p::Float64)
  # If p is smaller than machine precision, rounding will produce
  # errors, so here we chose to make the approximation that the zero
  # truncated Poisson will almost surely be equal to 1
  if p < 2e-16
    return 1
  end
  # Compute the probability of 0
  p_0 = exp(-p)
  # If the probability of 0 is small enough,
  # use the native truncate.jl
  if p_0 < .1
    poisson_dist = Poisson(p)
    while true
      res = rand(poisson_dist)
      if res > 0
        return res
      end
    end
  # Otherwise, use an inversion algorithm
  else
    u = (1-p_0)*rand()
    p_pdf = p*exp(-p)
    zt_cdf = p_pdf
    k = 1
    while u > zt_cdf
      k += 1
      p_pdf *= p/k
      zt_cdf += p_pdf
    end
    return max(k,1)
  end
end
