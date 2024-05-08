#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=20G srun --pty julia 

# Download prior samples
using DrWatson
@quickactivate "ASPIRE.jl"
import Pkg; Pkg.instantiate()

data_path = "."
run(`wget "https://www.dropbox.com/scl/fi/t7523css1jk6ylblpq99l/v_train.jld2?rlkey=d0dpx3s2lgdkmzqacurx785x8&dl=0" -q -O $data_path`)
