#module load Julia/1.8/5; salloc -A rafael -t01:80:00 --gres=gpu:1 --mem-per-cpu=50G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=30G srun --pty julia 
using DrWatson
@quickactivate :IterPhySum
import Pkg; Pkg.instantiate()

# export DEVITO_LANGUAGE=openacc
# export DEVITO_ARCH=nvc
# export DEVITO_PLATFORM=nvidiaX
# module load cudnn nvhpc Miniconda/3

#https://github.com/mloubout/UNet.jl.git
using JUDI
using InvertibleNetworks, Flux, UNet
using PyPlot,SlimPlotting
using LinearAlgebra, Random, Statistics
using ImageQualityIndexes
using BSON, JLD2

function posterior_sampler(G, y, x; device=gpu, num_samples=1, batch_size=16)
  size_x = size(x)
    # make samples from posterior for train sample 
  X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
      ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
      X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
          ZX_noise_i,
          Zy_fixed_train
        ) |> cpu;
  end
  X_post_train
end

function z_shape_simple(G, ZX_test)
    Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
    for i=G.L:-1:1
        if i < G.L
            ZX = tensor_cat(ZX, Z_save[i])
        end
        ZX = G.squeezer.inverse(ZX) 
    end
    ZX
end

function z_shape_simple_forward(G, X)
	orig_shape = size(X)
  G.split_scales && (Z_save = array_of_array(X, G.L-1))
  for i=1:G.L
      (G.split_scales) && (X = G.squeezer.forward(X))
      if G.split_scales && i < G.L    # don't split after last iteration
          X, Z = tensor_split(X)
          Z_save[i] = Z
          G.Z_dims[i] = collect(size(Z))
      end
  end
  G.split_scales && (X = cat_states(Z_save, X))
  return X
end

function load_net(net_path_i;chan_obs=1)
	unet_lev = 4#BSON.load(net_path_i)["unet_lev"];
	n_hidden = BSON.load(net_path_i)["n_hidden"];
	L = BSON.load(net_path_i)["L"];
	K = BSON.load(net_path_i)["K"];
	Params = BSON.load(net_path_i)["Params"]; 

	unet = FluxBlock(Chain(BSON.load(net_path_i)["unet_model"]|> device))
	Random.seed!(123);
	cond_net = NetworkConditionalGlow(1, 1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0));
	G   = SummarizedNet(cond_net, unet)

	G_dummy = deepcopy(G)
	set_params!(G_dummy,Params)
	set_params!(G.cond_net,get_params(G_dummy.cond_net))
	G = G |> device;
end



# Plotting configs
sim_name = "hint2-weak-iter1"
plot_path = joinpath("/slimdata/rafaeldata/plots/IterPhySum",sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)

intervals = [(1.48,1.5782),(1.5782,2.8)]
vmin_brain = intervals[1][1]
vmax_brain = intervals[2][2]
cmap_types = [ColorMap("cet_CET_L1"),matplotlib.cm.Reds]
cmap = create_multi_color_map(intervals, cmap_types)

# Training hyperparameters 
device = gpu

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.01f0

n_epochs     = 200

save_every   = 5
plot_every   = 2

n_condmean   = 40

fac = 1000f0
ind_train = 1000
indx = 24
ind = ind_train+15+indx-1
#@load "/slimdata/rafaeldata/brain_cond_data/brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2" snr dobs_noisy n o d q  m;
#@load "/slimdata/rafaeldata/brain_cond_data/brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2" snr  n o d q  m;

@load "/slimdata/rafaeldata/brain_cond_data/gauss_freq_brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2" n o d q  m dobs_noisy;

x_gt  =  sqrt.(1f0./(m));
mmin, mmax = extrema(m)

function proj(x)
    out = 1 .* x
    out[out .< mmin] .= mmin
    out[out .> mmax] .= mmax
    return out
end

n_x, n_y = size(x_gt)
chan_target = 1
N = n_x*n_y*chan_target

space_order_inv = 16
dt_comp_inv = 0.05f0
opt_inv = Options(dt_comp=dt_comp_inv, isic=false, space_order=space_order_inv)

#get observation (m0 and rtm)
# grad_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_4.jld2", "r")["grad_train"];
# grad_train_norm_4 = grad_train ./ quantile(abs.(vec(grad_train)),0.99)
# y_4 = Float32.(grad_train_norm_4[:,:,:,ind:ind]);
# m0s_4 = permutedims(JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_4.jld2", "r")["m0_train"],(2,1,3,4));
# m0s_4 = sqrt.(1f0 ./ m0s_4); #convert to v
# m0_4  = m0s_4[:,:,:,ind:ind];
# y   = tensor_cat(y_4,m0_4;);

grad_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/gauss_grad_water_no_rho_no_isic_iter_1.jld2", "r")["grad_train"];
grad_train ./= quantile(abs.(vec(grad_train[:,:,:,1:300])),0.99)
y = Float32.(grad_train[:,:,:,ind:ind]);

#load in CNF network
#net_path_1 = "/slimdata/rafaeldata/savednets/K=9_L=3_batch_size=8_chan_cond=1_clipnorm_val=3.0_e=175_iter=1_lr=0.0008_n_hidden=64_n_src=16_n_train=992_noise_lev_x=0.01_noise_lev_y=0.0_unet_lev=4_use_dm=false_use_m=false.bson"
net_path_1 = "/slimdata/rafaeldata/savednets/K=9_L=3_batch_size=8_chan_cond=1_clipnorm_val=3.0_decay_every=5_decay_fac=1.1_decay_start=70_e=200_iter=1_lr=0.0008_n_hidden=64_n_src=16_n_train=992_noise_lev_x=0.000923_noise_lev_y=0.0_unet_lev=4_use_dm=false_use_m=false.bson"
G = load_net(net_path_1;chan_obs=2)

n_hidden = BSON.load(net_path_1)["n_hidden"];
L = BSON.load(net_path_1)["L"];
K = BSON.load(net_path_1)["K"];
Random.seed!(123);
G_z = NetworkGlow(1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0))|> device;

resample = false
batch_size = 32
inner_loop = 64

z_inv = true #seems to be better init. Not neccesarily better optimization route
Random.seed!(123);
Z_fix = randn(Float32, n_x,n_y,1,batch_size) |> device
if z_inv
	G_z(randn(Float32, n_x,n_y,1,batch_size)|> device) #initialize
	global Z_fix = G_z.inverse(vec(Z_fix|> device))
end

z_base = reshape(G_z(Z_fix)[1], n_x,n_y,1,batch_size); #might want to make this better because it currently messes up the noramlity of Z_fix. Might want to make z_base actually Z_inv=G_inv(Z) sp tjat G(Z_inv) is exactly noise
Y_train_latent_repeat = repeat(y |> cpu, 1, 1, 1, batch_size) |> device;
_, Zy_fixed_train, _ = G.forward(Z_fix, Y_train_latent_repeat); #needs to set the proper sizes here

X_gen_init = G.inverse(z_base,Zy_fixed_train)[1] |> cpu;

fig = figure()
imshow(X_gen_init[:,:,1,1]|> cpu, interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
axis("off"); colorbar(fraction=0.046, pad=0.04);
tight_layout()
fig_name = @strdict  z_inv indx
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_init.png"), fig); close(fig)

fig = figure()
imshow(mean(X_gen_init;dims=4)[:,:,1,1]|> cpu, interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
axis("off"); colorbar(fraction=0.046, pad=0.04);
tight_layout()
fig_name = @strdict  z_inv indx
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_init_mean.png"), fig); close(fig)

# sqrt(mean(mean(X_gen_init;dims=4)[:,:,1,1] - x_gt))
# 0.002081844874210445

###### Get inside of brain mask
norm_min = 1f10
ind_good = 1
for ind_i = 1:1500
    data_path_mask = "/slimdata/rafaeldata/fastmri_brains/fastmri_training_data_"*string(ind_i)*".jld2"
    v_ind  = sqrt.(1f0 ./ JLD2.jldopen(data_path_mask, "r")["m"])
    norm_curr = norm(v_ind - x_gt)
    if norm_curr < norm_min
        global norm_min = norm_curr
        global ind_good = ind_i
    end
end

data_path_mask = "/slimdata/rafaeldata/fastmri_brains/fastmri_training_data_"*string(ind_good)*".jld2"
m_mask = JLD2.jldopen(data_path_mask, "r")["m"];
m0_mask = JLD2.jldopen(data_path_mask, "r")["m0"];
diff = Float32.(m_mask - m0_mask);
mask_brain = diff .!== 0f0;

fig = figure()
imshow(mask_brain)
tight_layout()
fig_name = @strdict 
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_mask.png"), bbox_inches="tight", dpi=400)


###### Filter data
freqs = 10 .*[3, 6, 12, 15]
freq = freqs[4]
dobs_noisy_filter = filter_data(dobs_noisy; fmin=0.0, fmax=Float64.(freq))
q_filter = filter_data(q; fmin=0, fmax=freq)
#dobs_noisy_filter = dobs_noisy; 
#q_filter =q; 

model = Model(n, d, o, m;)
F = judiModeling(model, q.geometry, dobs_noisy.geometry; options=opt_inv)
y_pred_star = F(model)q_filter[1] 
res_star = y_pred_star - dobs_noisy_filter[1]
snr_star = round(20*log10(norm(y_pred_star)/norm(res_star));digits=2)




#circulat gradient mute for source artifacts
radius = 230;
mask = [(i - 512 / 2)^2 + (j - 512 / 2)^2 - radius^2 for i = 1:512, j = 1:512];
mask[mask .> 0] .= 0f0;
mask[mask .< 0] .= 1f0;

# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = [];
loss_inner = []

lr = 5f-5
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))


weak_lambda = 0 
factor = 1f-13

starting_alpha = 1f-2
batchsize_src = 2

i_src = randperm(dobs_noisy_filter.nsrc)[1:batchsize_src]
f_star, _ = fwi_objective(model, q_filter[i_src],dobs_noisy_filter[i_src]; options=opt_inv)


m_i =  1 ./ X_gen_init.^2 ; 
for e=1:n_epochs # epoch loop
  i_src = randperm(dobs_noisy_filter.nsrc)[1:batchsize_src]

	gs_rtm = zeros(n_x,n_y,1,batch_size)
	f_all = 0
	for i in 1:batch_size
		model0 = Model(n, d, o, m_i[:,:,1,i];)
		f, g = fwi_objective(model0, q_filter[i_src],dobs_noisy_filter[i_src]; options=opt_inv)
		g .*= mask
    g = g/norm(g, Inf)

    gs_rtm[:,:,:,i] = g.data
    global m_i[:,:,1,i] = proj(m_i[:,:,1,i] .- starting_alpha .* gs_rtm[:,:,1,i:i])
		f_all += f
	end
	append!(loss, f_all / batch_size) 

	v_i = sqrt.(1f0./(m_i));
	
	if(mod(e,plot_every)==0) 
		y_plot = -gs_rtm[:,:,1,1]
	  a = quantile(abs.(vec(y_plot)), 98/100)

	  fig = figure()
	  subplot(1,3,1)
	  imshow(y_plot, vmin=-a,vmax=a,interpolation="none", cmap="gray")
	  axis("off"); title(L"-rtm");#colorbar(fraction=0.046, pad=0.04);

	  subplot(1,3,2); imshow(sqrt.(1f0./(m_i))[:,:,1,1], interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	  axis("off"); title(L"x gen"); #colorbar(fraction=0.046, pad=0.04);

	  subplot(1,3,3); imshow(v_i[:,:,1,1], interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	  axis("off"); title(L"x update"); #colorbar(fraction=0.046, pad=0.04);

	  tight_layout()
	  fig_name = @strdict inner_loop weak_lambda freq e 
	  safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_grad.png"), fig); close(fig)
	end
	
	for t in 1:inner_loop
		#Z_fix = G_z.inverse(vec(randn(Float32, n_x,n_y,1,batch_size)|> device))
		resample && (global Z_fix = randn(Float32, n_x,n_y,1,batch_size) |> device)
		z_base, lgdet = G_z(Z_fix)
		X_gen, Y_gen  = G.inverse(reshape(z_base, n_x,n_y,1,batch_size),Zy_fixed_train);
		X_gen_cpu = X_gen |>cpu

		gs_penalty = zeros(Float32,n_x,n_y,1,batch_size);
		f_all_inner = 0
		for i in 1:batch_size
			g = (X_gen_cpu[:,:,1,i] .- v_i[:,:,1,i])  
			gs_penalty[:,:,:,i] =  g
			f_all_inner += norm(g)^2
		end
		println(f_all_inner / batch_size)
		append!(loss_inner, f_all_inner / batch_size)  # normalize by image size and batch size
		append!(logdet_train, -lgdet / N)


		ΔX, X, ΔY = G.backward_inv(((gs_penalty ./ factor)|>device) / batch_size, X_gen, Y_gen; Y_save=Y_train_latent_repeat)
		G_z.backward(vec(ΔX), vec(z_base);)

		for p in get_params(G_z) 
				Flux.update!(opt,p.data,p.grad)
		end; clear_grad!(G_z)
	end

	print("Iter: epoch=", e, "/", n_epochs, 
	    "; f l2 = ",  loss[end], 
	    "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	Base.flush(Base.stdout)

  if(mod(e,plot_every)==0) 
  	#Z_fix = G_z.inverse(vec(randn(Float32, n_x,n_y,1,batch_size)|> device))
  	resample && (global Z_fix = randn(Float32, n_x,n_y,1,batch_size) |> device)
		X_post, Y_gen  = G.inverse(reshape(G_z(Z_fix)[1], n_x,n_y,1,batch_size), Zy_fixed_train);
		X_post = X_post |> cpu

		X_post_mean = mean(X_post,dims=4)
		X_post_std  = fac.*mask_brain .*std(X_post, dims=4)
		x_hat = X_post_mean[:,:,1,1]	

		error_mean = mask_brain .* fac.* abs.(x_hat-x_gt[:,:,1,1])
		ssim_i = round(assess_ssim(x_hat, x_gt[:,:,1,1]),digits=2)
		rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)
    append!(l2_cm, rmse_i)

		y_plot = dobs_noisy_filter[1].data[1] |> cpu
		a = quantile(abs.(vec(y_plot)), 95/100)

		X_post_m = 1 ./ X_post.^2 
		model0 = Model(n, d, o, X_post_m[:,:,1,1];)
		
		
		y_pred_1 = F(model0)q_filter[1] 
		y_pred_1_plot = (y_pred_1).data[1] |> cpu
		res = y_pred_1 - dobs_noisy_filter[1]
		snr_i = round(20*log10(norm(y_pred_1)/norm(res));digits=2)

		# model0 = Model(n, d, o, m;)
		# y_pred_1 = F(model0)q[1] 
		# res = y_pred_1 - dobs_noisy[1]
		# snr_i = 20*log10(norm(y_pred_1)/norm(res))

	

		num_cols = 4
		fig = figure(figsize=(14, 7)); 
		# subplot(2, num_cols,1); imshow(y_plot, aspect=0.21, vmin=-a,vmax=a, interpolation="none", cmap="gray")
		# axis("off"); title(L"Observation $\mathbf{y} = \mathbf{F}(\mathbf{x^{*}}) + \varepsilon$ "); #colorbar(fraction=0.046, pad=0.04);

		subplot(2, num_cols,1); imshow(X_post[:,:,1,1],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		axis("off");  #colorbar(fraction=0.046, pad=0.04);
		title(L"Posterior samp $G(z_1)$")

		# subplot(2, num_cols,2); imshow(X_post[:,:,1,2],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		# axis("off");  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")
		# title(L"Posterior samp $G(z_2)$")

		# subplot(2, num_cols,3); imshow(X_post[:,:,1,3],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		# axis("off");  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")
		# title(L"Posterior samp $G(z_3)$")

		subplot(2, num_cols,4); imshow(X_post[:,:,1,end],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		axis("off");  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")
		title(L"Posterior samp $G(z_4)$")

		subplot(2, num_cols,5); imshow(x_hat, interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		axis("off"); title("Posterior mean SSIM="*string(ssim_i)) ; #colorbar(fraction=0.046, pad=0.04)

		subplot(2, num_cols,6); imshow(x_gt[:,:,1,1], interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
		axis("off"); title(L"Reference $\mathbf{x^{*}}$") ;# colorbar(fraction=0.046, pad=0.04)
	
		subplot(2, num_cols,7); imshow(error_mean, vmin=0,vmax=50, interpolation="none", cmap="magma")
		axis("off");title("RMSE "*string(rmse_i)) ; cb = colorbar(fraction=0.046, pad=0.04,label="[meters/second]")

		subplot(2, num_cols,8); imshow(X_post_std[:,:,1,1], vmin=0,vmax=50, interpolation="none", cmap="magma")
		axis("off"); title("Posterior variance") ;cb =colorbar(fraction=0.046, pad=0.04,label="[meters/second]")

		# subplot(2, num_cols,8); imshow(res[1].data[1], aspect=0.21, vmin=-a,vmax=a, interpolation="none", cmap="gray")
		# axis("off"); title(L"Residual $\mathbf{F}(G(z_1))-\mathbf{y}$ ");#cb =colorbar(fraction=0.046, pad=0.04)
	
		tight_layout()
		fig_name = @strdict indx z_inv resample inner_loop  freq batchsize_src   factor   e lr  batch_size
		safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_samples.png"), fig); close(fig)


		fig = figure(figsize=(16, 7)); 
		subplot(1, 3,1); imshow(y_plot, aspect=0.21, vmin=-a,vmax=a, interpolation="none", cmap="gray")
		axis("off"); title(L"Observation $\mathbf{y} = \mathbf{F}(\mathbf{x^{*}}) + \varepsilon$ "); colorbar(fraction=0.046, pad=0.04);

		subplot(1, 3,2); imshow(y_pred_1_plot, aspect=0.21, vmin=-a,vmax=a, interpolation="none", cmap="gray")
		axis("off"); title(L"Predicted Observation $\mathbf{F}(G(z_1))$"); colorbar(fraction=0.046, pad=0.04);

		subplot(1, 3, 3); imshow(res[1].data[1], aspect=0.21, vmin=-a,vmax=a, interpolation="none", cmap="gray")
		axis("off"); title(L"Residual | $\mathbf{F}(G(z_1))-\mathbf{y}$"*"| SNR=$(snr_i) | SNR GT=$(snr_star)");cb =colorbar(fraction=0.046, pad=0.04)
	
		tight_layout()
		fig_name = @strdict indx z_inv resample inner_loop  freq batchsize_src   factor   e lr  batch_size
		safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_observed_data.png"), fig); close(fig)
		
		if e >= 6
			#sum_train = loss + logdet_train #+ loss_prior

			fig = figure("training logs ", figsize=(10,12))
			subplot(4,1,1); title("L2 Term: train="*string(loss[end]))
			plot(range(0f0, 1f0, length=length(loss)), loss, label="train");
			xlabel("Parameter Update"); 
			axhline(y=f_star;color="black",linestyle="--",label="noise level");legend();

			subplot(4,1,2); title("penalty Term: train="*string(loss_inner[end]))
			plot(range(0f0, 1f0, length=length(loss_inner)), loss_inner, label="train");
			xlabel("Parameter Update"); legend();

			subplot(4,1,3); title("Logdet Term: train="*string(logdet_train[end])*" test=")
			plot(range(0f0, 1f0, length=length(logdet_train)),logdet_train);
			xlabel("Parameter Update") ;

			# subplot(4,1,3); title("Total Objective: train="*string(sum_train[end])*" test=")
			# plot(range(0f0, 1f0, length=length(sum_train)),sum_train); 

      subplot(4,1,4); title("RMSE train=$(l2_cm[end])")
      plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
      xlabel("Parameter Update") 

			tight_layout()
			fig_name = @strdict z_inv resample inner_loop  freq batchsize_src  factor   e lr  batch_size
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end

		if(mod(e,save_every)==0) 
			Params = get_params(G_z) |> cpu;
			save_dict = @strdict X_post indx z_inv resample factor inner_loop freq batchsize_src Z_fix  clipnorm_val e lr Params loss logdet_train l2_cm ssim     batch_size; 
			@tagsave(
				joinpath("/slimdata/rafaeldata/savednets_hint2", savename(save_dict, "bson"; digits=6)),
				save_dict;
				safe=true
			);
		end
	end


end
