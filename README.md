# ASPIRE.jl


To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data needs to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts.

To download the brain prior samples: https://www.dropbox.com/scl/fi/t7523css1jk6ylblpq99l/v_train.jld2?rlkey=d0dpx3s2lgdkmzqacurx785x8&dl=0