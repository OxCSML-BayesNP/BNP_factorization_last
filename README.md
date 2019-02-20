# BNP-factorization


Julia code for the Nonnegative Bayesian nonparametric factor models with completely random measures.


### Installing

First time you run the code, you need to install dependencies. Run in julia

```
include("install_pkg.jl")
```

## Sample from model
In the file test_sample.jl, one can simulate a dataset from the model. Set the parameters of the model to sample from in the 'Set Parameters' section. The simulated data will be stored in data/$data_name/

## Running the tests

In the file test_both.jl, one can test the code on simulated data : Set the parameters of the model to sample from in the 'Set Parameters' section. 
Then the script will run consecutively the Gibbs sampler on a weighted and unweigted version of the simulated data.


In the file test_real_data.jl, one can test the code on real data :
	In the 'Data' section, chose the name of the dataset of interest in "data_name". The file must then be in data/$data_name.txt. The results will then be stored in results/$data_name/$results_path/time_of_run. 
	Set the parameters of the Gibbs sampler in the 'Set Parameters' section. If the parameter "monitoring_sigma" is set to true, display the current value of sigma every 5% of the progress. It also allows to save the current state of the run every 5%. Then use continue_real.jl if the script is stopped before the end of the sampler to carry from last save. If there is a saved progress, one can plot (and save plots) the current results using the script load_.jl.


