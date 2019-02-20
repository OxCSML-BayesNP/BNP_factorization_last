cd(main_dir)
results_path = string("results/",data_name,"/",current_dir,"/variables/")
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
