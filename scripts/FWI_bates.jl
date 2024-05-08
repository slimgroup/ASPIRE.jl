#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=30G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=30G srun --pty julia 

# export DEVITO_LANGUAGE=openacc
# export DEVITO_ARCH=nvc
# export DEVITO_PLATFORM=nvidiaX 
# module load cudnn nvhpc Miniconda/3

using DrWatson
@quickactivate :IterPhySum
import Pkg; Pkg.instantiate()

using JUDI
using Random, LinearAlgebra
using PyPlot
using SlimPlotting
using JLD2
using Statistics
using JOLI
using SlimOptim

sim_name = "mle_fwi_gaush"
plot_path = joinpath("/slimdata/rafaeldata/plots/IterPhySum",sim_name)
data_path = "/slimdata/rafaeldata/savedfwi"

intervals = [(1.48,1.5782),(1.5782,2.8)]
vmin_brain = intervals[1][1]
vmax_brain =intervals[2][2]
cmap_types = [ColorMap("cet_CET_L1"),matplotlib.cm.Reds]
cmap = create_multi_color_map(intervals, cmap_types)

fac=1000f0
indx = 24
ind_train = 1000
ind = ind_train+15+indx-1
@load "/slimdata/rafaeldata/brain_cond_data/brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2" n o d q m
x_gt = sqrt.(1f0 ./ m)

@load "/slimdata/rafaeldata/brain_cond_data/brain_full_dobs_ind_"*string(ind)*".jld2" dobs
#@load "/slimdata/rafaeldata/brain_cond_data/gauss_brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2"  dobs_noisy;




# noise_fac = 0.01
# dobs_noisy = deepcopy(dobs) 
# for i in 1:dobs.nsrc
#     dobs_noisy.data[i] += noise_fac.*randn(size(dobs_noisy.data[i]))
# end


snr = 35
noise = deepcopy(dobs)
for l = 1:dobs.nsrc
    noise.data[l] = randn(Float32, size(dobs.data[l]))
end
noise = noise/norm(noise) * norm(dobs) * 10f0^(-snr/20f0)
dobs_noisy = dobs + noise



norm_min = 1f10
ind_good = 1
for ind_i = 1:1500
    data_path_mask = "/slimdata/rafaeldata/fastmri_brains/fastmri_training_data_"*string(ind_i)*".jld2"
    v_i  = sqrt.(1f0 ./ JLD2.jldopen(data_path_mask, "r")["m"])
    norm_curr = norm(v_i - x_gt)
    if norm_curr < norm_min
        global norm_min = norm_curr
        global ind_good = ind_i
    end
end

data_path_mask = "/slimdata/rafaeldata/fastmri_brains/fastmri_training_data_"*string(ind_good)*".jld2"
m_i = JLD2.jldopen(data_path_mask, "r")["m"]
m0_i = JLD2.jldopen(data_path_mask, "r")["m0"]
diff = Float32.(m_i - m0_i)
mask_brain = diff .!== 0f0

# fig = figure()
# imshow(mask_brain)
# tight_layout()
# fig_name = @strdict 
# fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_mask.png"), bbox_inches="tight", dpi=400)


m0 = m[50,50].*ones(Float32,size(m))
space_order_inv = 16
dt_comp_inv = 0.05f0
opt_inv = Options(dt_comp=dt_comp_inv, space_order=space_order_inv)

# Bound projection
mmin, mmax = extrema(m)
function proj(x)
    out = 1 .* x
    out[out .< mmin] .= mmin
    out[out .> mmax] .= mmax
    return out
end



#gradient mute for source artifacts
radius = 230;
mask = [(i - 512 / 2)^2 + (j - 512 / 2)^2 - radius^2 for i = 1:512, j = 1:512];
mask[mask .> 0] .= 0f0;
mask[mask .< 0] .= 1f0;

freqs = 10 .*[3, 6, 7,12, 15]
freq = freqs[3]
dobs_noisy_filter = filter_data(dobs_noisy; fmin=0.0, fmax=Float64.(freq))
q = filter_data(q; fmin=0, fmax=freq)

# for freq in freqs
#     dobs_noisy_filter = filter_data(dobs_noisy; fmin=0.0, fmax=Float64.(freq))

#     y_plot = dobs_noisy_filter[1].data[1] 
#     a_data = quantile(abs.(vec(y_plot)), 95/100)

#     fig = figure(figsize=(7,8))
#     plot_sdata(dobs_noisy_filter[1];new_fig=false)
#     #imshow(y_plot, aspect=0.21, vmin=-a_data,vmax=a_data, interpolation="none", cmap="gray")
#     axis("off"); title("Observation filtered from 0 to $(freq)Khzs"); colorbar(fraction=0.0235, pad=0.04);

#     tight_layout()
#     fig_name = @strdict freq
#     safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_filters.png"), fig); close(fig)
# end


#  fig = figure(figsize=(7,8))
# plot_sdata(dobs_noisy[1];new_fig=false)
# #imshow(y_plot, aspect=0.21, vmin=-a_data,vmax=a_data, interpolation="none", cmap="gray")
# axis("off"); title("Observation unfiltered "); colorbar(fraction=0.0235, pad=0.04);

# tight_layout()
# fig_name = @strdict 
# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_nofilters.png"), fig); close(fig)


# fig = figure(figsize=(7,3))
  
# plot(dobs_noisy[1].data[1][1:200,128] ./ maximum(dobs_noisy[1].data[1][:,128]); label="original data")
# plot(dobs_noisy_filter[1].data[1][1:200,128] ./ maximum(dobs_noisy_filter[1].data[1][:,128]); label="filtered 0,60khz")
# legend()
# xlabel("time step")
# ylabel("amplitude")
# tight_layout()
# fig_name = @strdict 
# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_filter.png"), fig); close(fig)



model0 = Model(n, d, o, deepcopy(m0);)
F0 = judiModeling(deepcopy(model0), q.geometry, dobs_noisy.geometry; options=opt_inv)

# Optimization parameters
niterations = 400
batchsize   = 16

line_search = false
ls = BackTracking(order=3, iterations=10)
#starting_alpha = 1f-2
starting_alpha = 1f8

losses = []
losses_rmse = []
losses_rmse_mask = []
save_every = 50
plot_every = 5

total_pde = 0
grad_scale = 0

grad_norm = false
gaush = false
sigma_diag = 0f0 .* m0
mu = copy(m0)

y_plot = dobs_noisy_filter[1].data[1] 
a_data = quantile(abs.(vec(y_plot)), 95/100)

# Main loop
for j=1:niterations
    eps_i = randn(Float32)
    if gaush
        global model0.m .= proj(mu + eps_i .* sigma_diag)
    else
        global model0.m .= proj(mu) 
    end

    i = randperm(dobs_noisy_filter.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], dobs_noisy_filter[i];options=opt_inv)
    global total_pde += 2
 
    x_hat = sqrt.(1f0 ./ abs.(model0.m))

    append!(losses, fval)
    append!(losses_rmse, sqrt(mean((x_hat - x_gt).^2)))
    append!(losses_rmse_mask, sqrt(mean((fac.*mask_brain.*(x_hat-x_gt)).^2)))
 
    gradient .*= mask
    p = -gradient#/norm(gradient, Inf)

    println("FWI iteration no: ",j,"; function value: ",fval)
    step_t = starting_alpha

    println("lineserach step: ",step_t,"; function value: ")
    nabla_m = step_t .* p.data

    mu .= proj(mu .+ nabla_m)
    sigma_diag .= sigma_diag .+ eps_i .* nabla_m


    pred_dobs = F0(model0)q[1]
    res = pred_dobs - dobs_noisy_filter[1]

    if mod(j-1,plot_every) == 0

        fig = figure(figsize=(10,4))
        a = quantile(vec(abs.(p)),0.98)
        imshow(p,vmin=-a,vmax=a,cmap="gray")
        tight_layout()
        fig_name = @strdict  batchsize j ind #IC
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_grad.png"), fig); close(fig)

        fig = figure(figsize=(15,8))
        subplot(2,3,1); title(L"updated $\hat \mathbf{x}$"*"with max freq=$(freq/1000)Mhz")
        imshow(sqrt.(1f0 ./ model0.m),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);
        axis("off");
        colorbar(fraction=0.0235, pad=0.04);
        subplot(2,3,2); title(L"ground truth $\mathbf{x}^{\ast}$")
        imshow(sqrt.(1f0 ./ m) ,vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);
        axis("off");
        colorbar(fraction=0.0235, pad=0.04);

        subplot(2, 3,4); imshow(pred_dobs[1].data[1], aspect=0.21, vmin=-a_data,vmax=a_data, interpolation="none", cmap="gray")
        axis("off"); title(L"predicted data $\mathbf{F}(\hat \mathbf{x})$ ");colorbar(fraction=0.0235, pad=0.04);

        subplot(2, 3,5); imshow(y_plot, aspect=0.21, vmin=-a_data,vmax=a_data, interpolation="none", cmap="gray")
        axis("off"); title(L"Observation $\mathbf{y} = \mathbf{F}(\mathbf{x^{*}}) + \varepsilon$ "); colorbar(fraction=0.0235, pad=0.04);
        subplot(2, 3,6); imshow(res[1].data[1], aspect=0.21, vmin=-a_data,vmax=a_data, interpolation="none", cmap="gray")
        axis("off"); title(L"Residual $\mathbf{F}(\hat \mathbf{x})-\mathbf{y}$ ");colorbar(fraction=0.0235, pad=0.04);

        tight_layout()
        fig_name = @strdict  grad_norm line_search starting_alpha total_pde batchsize j  ind freq 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_update.png"), fig); close(fig)


        a_uq = quantile(vec(abs.(sigma_diag)),0.98)
        a_error = quantile(vec(abs.(sqrt.(1f0 ./ model0.m.data) .- sqrt.(1f0 ./ m) )),0.98)
        fig = figure(figsize=(14,8))
        subplot(2,3,1); title("v0")
        imshow(sqrt.(1f0 ./ abs.(m0));vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);
        colorbar(fraction=0.0235, pad=0.04);
        axis("off")
        subplot(2,3,2); title("updated v")
        imshow(sqrt.(1f0 ./ model0.m),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);
        colorbar(fraction=0.0235, pad=0.04);
        axis("off")

        subplot(2,3,3); title("ground truth v")
        imshow(sqrt.(1f0 ./ m) ,vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);
        colorbar(fraction=0.0235, pad=0.04);
        axis("off")

        subplot(2,3,5); title("UQ")
        imshow(abs.(sigma_diag) ,vmin=0,vmax=a_uq,cmap="jet");
        colorbar(fraction=0.0235, pad=0.04);axis("off")

        subplot(2,3,6); title("abs error")
        imshow(abs.(sqrt.(1f0 ./ model0.m.data) .- sqrt.(1f0 ./ m) ) ,vmin=0,vmax=a_error,cmap="jet");
        colorbar(fraction=0.0235, pad=0.04);axis("off")
        tight_layout()
        fig_name = @strdict  grad_norm gaush  line_search starting_alpha total_pde batchsize j  ind freq 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_update.png"), fig); close(fig)


        if j >= 2
            fig = figure()
            subplot(3,1,1); plot(losses;label="objective",linewidth=0.8,color="black");title("final $(losses[end])")
            grid();ylabel("Data misfit"); xlabel("FWI update")

            subplot(3,1,2); plot(losses_rmse;label="error",linewidth=0.8,color="black");title("final $(losses_rmse[end])")
            grid();ylabel("Model error"); xlabel("FWI update")

            subplot(3,1,3); plot(losses_rmse_mask;label="error inside",linewidth=0.8,color="black");title("final $(losses_rmse_mask[end])")
            grid();ylabel("Model error inside"); xlabel("FWI update")


 
            tight_layout()
            fig_name = @strdict grad_norm gaush  line_search starting_alpha total_pde   batchsize j ind freq 
            safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
        end
    end
   if mod(j,save_every) == 0
        m_final = model0.m.data
        save_dict = @strdict  grad_norm sigma_diag gaush  line_search starting_alpha total_pde batchsize m_final j losses losses_rmse losses_rmse_mask m0 m ind freq 
        @tagsave(
            joinpath(data_path, savename(save_dict, "jld2"; digits=6)),
            save_dict;
            safe=true
        )
    end
end
