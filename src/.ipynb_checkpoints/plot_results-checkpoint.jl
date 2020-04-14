

rc("font",size=16.)
#burn_start = (warm_start ? 1 : Int(3*n_iter/4))

ioff()

#-------------------------------------------------------------------------------
# Plot and save number of communities
#-------------------------------------------------------------------------------
PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(n_active_list,label="Estimated number of communities")
if plot_true
  PyPlot.plot(length(R)*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of the number of active communities")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Active communitites")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of the number of active communities.png",bbox_inches="tight")
PyPlot.close()


#-------------------------------------------------------------------------------
# Plot and save parameters
#-------------------------------------------------------------------------------
PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(kappa_list,label="Estimated kappa")
if plot_true
  PyPlot.plot(kappa*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of Kappa")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Value")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of kappa.png",bbox_inches="tight")
PyPlot.close()


PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(tau_list,label="Estimated tau")
if plot_true
  PyPlot.plot(tau*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of Tau")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Value")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of tau.png",bbox_inches="tight")
PyPlot.close()

PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(sigma_list,label="Estimated sigma")
if plot_true
  PyPlot.plot(sigma*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of Sigma")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Value")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of sigma.png",bbox_inches="tight")
PyPlot.close()


PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(alpha_list,label="Estimated alpha")
if plot_true
  PyPlot.plot(alpha*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of Alpha")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Value")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of alpha.png",bbox_inches="tight")
PyPlot.close()

PyPlot.figure(figsize=(15.,12.))
PyPlot.plot(beta_list,label="Estimated beta")
if plot_true
  PyPlot.plot(beta*ones(n_iter),label="True values", linewidth = 4)
end
PyPlot.title("MCMC estimation of Beta")
PyPlot.xlabel("MCMC iteration (x$skip)")
PyPlot.ylabel("Value")
PyPlot.legend()
PyPlot.savefig("MCMC estimation of beta.png",bbox_inches="tight")
PyPlot.close()



#-------------------------------------------------------------------------------
# Plot and save the K largest activities
#-------------------------------------------------------------------------------
for r_idx in 1:K
  estimated_r_list = activities_list[:,r_idx]
  PyPlot.figure(figsize=(15.,12.))
  PyPlot.plot(estimated_r_list,label="Estimated activity")
  if plot_true
    r_true = sort(R,rev=true)[r_idx]
    PyPlot.plot(r_true*ones(n_iter),label="True values", linewidth = 4)
  end
  PyPlot.title("MCMC estimation of $r_idx largest activity")
  PyPlot.xlabel("MCMC iteration (x$skip)")
  PyPlot.ylabel("Activity")
  PyPlot.legend()
  PyPlot.savefig("MCMC estimation of $r_idx largest activity")
  PyPlot.close()
end


if pred_ratio > 0
  PyPlot.figure(figsize=(15.,12.))
  PyPlot.plot(error_mean_list,label="Without burn in")
  PyPlot.plot(error_mean_list_burn, label=string("With ",burn," burn in"))
  PyPlot.title("AUC PR")
  PyPlot.xlabel("MCMC iteration")
  PyPlot.ylabel("Value")
  PyPlot.legend()
  PyPlot.savefig("AUC PR",bbox_inches="tight")
end
PyPlot.close("all")
ion()
