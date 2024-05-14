# ASPIRE: **A**mortized posteriors with **S**ummaries that are **P**hysics-based and **I**teratively **RE**fined

Code supporting "ASPIRE: Iterative Amortized Posterior Inference for Bayesian Inverse Problems"
https://arxiv.org/abs/2405.05398


**WARNING CODE AS IS**: code is not in working condition. Please reach out to me for questions. I will be slowly bring the code base to running status. 

In the meantime, the following benchmarks can be run using pregenerated datasets.

## Benchmarks

Solving the non-linear high-dimensional wave-based inverse problem of Transcranial Ultrasound Computed Tomography (TUCT).

- Benchmark 1: Traditional SBI/likelihood-free-inference/amortized-posterior-inference.

**Dataset**

Download the brain prior samples paired with synthetic observations[here](https://www.dropbox.com/scl/fi/9tbqdlw0qzydwknfk8np0/Xs_Ys_train.jld2?rlkey=bxp6cr0h9dfwz1o4sjcic4j9z&dl=0).

**Goal** 
Train an amortized posterior sampler of brain samples given raw observations.

**Metrics** 
Train on the first 1000 samples and compare with results in paper (Figure 7) on an unseen observation. 


## Code installation 

To reproduce this project, do the following:

0. Download this code base. Notice that raw data needs to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts.

