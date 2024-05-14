# Make sythetic observations with simulator and prior samples

# module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=20G srun --pty julia 

# These settings will use GPU for wave equation PDE solve

# export DEVITO_LANGUAGE=openacc
# export DEVITO_ARCH=nvc
# export DEVITO_PLATFORM=nvidiaX
# module load cudnn nvhpc Miniconda/3

using DrWatson
@quickactivate "ASPIRE.jl"
import Pkg; Pkg.instantiate()

using JUDI
using LinearAlgebra, Random, JLD2

include(srcdir("toneburst.jl"))

# Prior samples file path 
prior_path = "/slimdata/rafaeldata/brain_cond_data/Xs_Ys_train.jld2" 

# Save observations file path 
save_obs_path = "/slimdata/rafaeldata/brain_cond_data/brain_full_dobs_ind_test"

############################### simulation configurations ################################# 

# grid options
n = (512,512)
d = (0.5,0.5)
o = (0,0)

# Set up rec geometry 
nrec = 256 # no. of recs
nsrc = 16  # no. of srcs
tn = 240   # total simulation time in microseconds but he had different geometry
dt = 0.2f0 # around 5Mhz sampling rate

# Modeling time and sampling interval
sampling_rate = 1f0 / dt
f0 = .4    # Central frequency in MHz

cycles_wavelet = 4  # number of sine cycles
nt = Int(div(tn,dt)) + 2
wavelet = tone_burst(sampling_rate, f0, cycles_wavelet; signal_length=nt);
wavelet = reshape(wavelet, length(wavelet), 1)

# Setup circumference of receivers 
domain_x = (n[1] - 1)*d[1]
domain_z = (n[2] - 1)*d[2]
rad = .95*domain_x / 2
xrec, zrec, theta = circle_geom(domain_x / 2, domain_z / 2, rad, nrec)
yrec = 0f0 #2d so always 0 
# Set up source structure
#get correct number of sources by grabbing subset of the receiver positions
step_num = Int(nrec/nsrc)
xsrc  = xrec[1:step_num:end]
ysrc  = range(0f0, stop=0f0, length=nsrc)
zsrc  = zrec[1:step_num:end]

# Convert to cell in order to separate sources 
src_geometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dt, t=tn)
q = judiVector(src_geometry, wavelet)

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=tn, nsrc=nsrc);

# numerical wave solution options
space_order = 32
dt_comp = 0.025f0
opt = Options(dt_comp=dt_comp, space_order=space_order)

# run simulator for all prior samples 
Xs = JLD2.jldopen(prior_path, "r")["Xs"];
for ind in 1:1169  
	println("Synthetic observations for i=$(ind)")
    F = judiModeling(Model(n, d, o, 1 ./ Xs[:,:,1,ind].^2 ;), q.geometry, rec_geometry; options=opt)
    dobs = F*q

    @save save_obs_path*string(ind)*".jld2" q wavelet m dobs space_order dt_comp d o n
end
