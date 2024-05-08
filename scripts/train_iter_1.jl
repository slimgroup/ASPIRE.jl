#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=60G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=30G srun --pty julia 

using DrWatson
@quickactivate :IterPhySum
import Pkg; Pkg.instantiate()

# using JLD2
# using Statistics 

# grad_train = zeros(Float32,512,512,1,1168) 
# m0_train = zeros(Float32,512,512,1,1168) 
# m_train = zeros(Float32,512,512,1,1168) 
# for i in 1:1168
# 	println("$(i)/$(1168)")
	
# 	@load "/slimdata/rafaeldata/brain_cond_data/brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(i)*".jld2" Y_train m m0  

# 	grad_train[:,:,:, i] = mean(Y_train;dims=3)
# 	m0_train[:,:,:, i] = m0
# 	m_train[:,:,:, i] = m
# end

# @save "/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_1.jld2" grad_train m0_train m_train


#https://github.com/mloubout/UNet.jl.git
using InvertibleNetworks, Flux, UNet
using PyPlot,SlimPlotting
using LinearAlgebra, Random, Statistics
using ImageQualityIndexes
using BSON, JLD2
using Augmentor

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

function get_cm_l2_ssim(G, X_batch, Y_batch, X0_batch; device=gpu, num_samples=1)
		# needs to be towards target so that it generalizes accross iteration
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
	    	y   = Y_batch[:,:,:,i:i]
	    	x_gt   = X_batch[:,:,:,i:i]
	    	x0  = X0_batch[:,:,1,i]

	    	X_post = posterior_sampler(G, y, x_gt; device=device, num_samples=num_samples, batch_size=batch_size)
	    	x_gt   = x_gt[:,:,1,1]

	    	if use_dm
	    		x_gt =  (x0 + x_gt[:,:,1,1]) |> cpu
	    		X_post = x0 .+ X_post
	    	end

	    	if use_m
		    	X_post = sqrt.(1f0 ./ abs.(X_post))
				  x_gt   = sqrt.(1f0 ./ x_gt)
				end

			  x_hat =  mean(X_post,dims=4)[:,:,1,1]

	    	ssim_total += assess_ssim(x_hat, x_gt)
				l2_total   += sqrt(mean((x_hat - x_gt).^2))
		end
	return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16)
	l2_total = 0 
	logdet_total = 0 
	num_batches = div(size(Y_batch)[end], batch_size)
	for i in 1:num_batches
		x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 
    	y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 

    	x_i .+= noise_lev_x*randn(Float32, size(x_i)); 
    	y_i .+= noise_lev_y*randn(Float32, size(y_i)); 
    	Zx, Zy, lgdet = G.forward(x_i|> device, y_i|> device) |> cpu;
    	l2_total     += norm(Zx)^2 / (N*batch_size)
		logdet_total += lgdet / N
	end

	return l2_total / (num_batches), logdet_total / (num_batches)
end

intervals  = [(1.48,1.5782),(1.5782,2.8)]
vmin_brain = intervals[1][1]
vmax_brain = intervals[2][2]
cmap_types = [ColorMap("cet_CET_L1"),matplotlib.cm.Reds]
cmap       = create_multi_color_map(intervals, cmap_types)

# Plotting configs
sim_name = "cond-net-norho-iter-1"
plot_path = joinpath("/slimdata/rafaeldata/plots/IterPhySum",sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)

# Training hyperparameters 
device = gpu

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.01f0
noise_lev_y  = 0.0 

batch_size   = 8
n_epochs     = 200
num_post_samples = 32

save_every   = 25
plot_every   = 5
n_condmean   = 40
use_m = false
use_dm = false

iter = 1
n_src = 16

m_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_1.jld2", "r")["m_train"];
m0_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_1.jld2", "r")["m0_train"];
grad_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_1.jld2", "r")["grad_train"];

if !(use_m)
	m_train = sqrt.(1f0 ./ m_train)
	m0_train = sqrt.(1f0 ./ m0_train)
end


ind_train = 1000

target_train =   m_train[:,:,:,1:ind_train];
X0_train     =  m0_train[:,:,:,1:ind_train];
Y_train      = grad_train[:,:,:,1:ind_train];

target_test = m_train[:,:,:,ind_train+15:ind_train+100];
X0_test     = m0_train[:,:,:,ind_train+15:ind_train+100];
Y_test      = grad_train[:,:,:,ind_train+15:ind_train+100];

n_x, n_y, chan_target, n_train = size(target_train)
n_train = size(target_train)[end]
N = n_x*n_y*chan_target
chan_obs   = size(Y_train)[end-1]
chan_cond  = 1

#convert to velocity for now but test with slowness later
max_y = maximum(Y_train)
vmax_d = maximum(Y_train)

X_train  = target_train ;
X_test   = target_test  ;
if use_dm
	X_train  = target_train - X0_train ;
	X_test   = target_test  - X0_test;
end

n_batches    = cld(n_train, batch_size)-1
n_train = batch_size*n_batches

# Architecture parametrs
sum_net = true
unet = nothing
unet_lev = 4
if sum_net
	unet = Chain(Unet(chan_obs, chan_cond, unet_lev))
	trainmode!(unet, true)  
	unet = FluxBlock(unet) |> device
end

# Create conditional network
L = 3
K = 9 
n_hidden = 64
low = 0.5f0

Random.seed!(123);
cond_net = NetworkConditionalGlow(chan_target, chan_cond, n_hidden,  L, K;  split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

# Optimizer
#opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr));

decay_step = 40
decay = 0.75
#opt = Flux.Optimiser(ExpDecay(lr, decay, n_batches*decay_step, 1f-6),ADAM(lr))
opt = Flux.Optimiser(ExpDecay(lr, decay, n_batches*decay_step, 1f-6),ADAM(lr))

# cur_t = []
# cur_eta = []
# t = [10f0]
# for i in 1:n_epochs
# 	for j in 1:n_batches
# 	  Flux.update!(opt,t,[1f0])
# 	  println(opt[2].eta)
# 	  append!(cur_eta,opt[2].eta)
# 	  append!(cur_t,t)
# 	end
# end

# fig = figure()
# #plot(cur_eta)
# plot(cur_t)
# fig_name = @strdict 
# safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)


# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = [];
loss_test = []; logdet_test  = []; ssim_test = []; l2_cm_test = [];

pl = FlipX(0.5) 
#|>  FlipX(0.5)
aug = augment(target_train[:,:,1,1], pl)

fig = figure()
imshow(aug)
fig_name = @strdict 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)


# fig = figure()
# subplot(2,1,1)
# imshow(X[:,:,1,1])
# subplot(2,1,2)

#  y_plot = Y[:,:,1,1]
# a = quantile(abs.(vec(y_plot)), 98/100)

# imshow(y_plot,vmax=a,vmin=-a)
# fig_name = @strdict 
# safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)


crop_aug = false
flip_aug = true

for e=1:n_epochs # epoch loop
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 
	
		@time begin
    for b = 1:n_batches # batch loop
    	#@time begin
	        X_unaug = X_train[:, :, :, idx_e[:,b]];
	        Y_unaug = Y_train[:, :, :, idx_e[:,b]];

	        X = zeros(n_x,n_y,chan_target,batch_size)
	        Y = zeros(n_x,n_y,chan_target,batch_size)
	        @time for i in 1:batch_size
	        	X[:,:,1,i], Y[:,:,1,i] = augment(X_unaug[:,:,1,i] => Y_unaug[:,:,1,i], pl)
	        end
    			# X = X_train[:, :, :, idx_e[:,b]];
	        # Y = Y_train[:, :, :, idx_e[:,b]];

	        X .+= noise_lev_x*randn(Float32, size(X)); #noises not related to inverse problem 
	        Y .+= noise_lev_y*randn(Float32, size(Y));
	        Y = Y |> device; 
	        
	        @time Zx, Zy, lgdet = G.forward(X |> device, Y) #not a lot of gc time

	        # Loss function is l2 norm 
	        append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
	        append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

	        # Set gradients of flow and summary network
	        @time dx, x, dy = G.backward(Zx / batch_size, Zx, Zy; Y_save = Y)


	        for p in get_params(G) 
	          Flux.update!(opt,p.data,p.grad)
	        end; clear_grad!(G)

	        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	              "; f l2 = ",  loss[end], 
	              "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	        Base.flush(Base.stdout)
    	end
    end
  
  
    if(mod(e,plot_every)==0) 

    	# get loss of training objective on test set
	    @time l2_test_val, lgdet_test_val  = get_loss(G, X_test, Y_test; device=device, batch_size=batch_size)
	    append!(logdet_test, -lgdet_test_val); append!(loss_test, l2_test_val)

	    # get conditional mean metrics over training batch  
	    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean],X0_train[:,:,:,1:n_condmean], device=device, num_samples=num_post_samples)
	    append!(ssim, cm_ssim_train); append!(l2_cm, cm_l2_train)

	    # get conditional mean metrics over testing batch  
	    @time cm_l2_test, cm_ssim_test  = get_cm_l2_ssim(G, X_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean],X0_test[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
	    append!(ssim_test, cm_ssim_test); append!(l2_cm_test, cm_l2_test)

	    for (test_x, test_y, test_x0, file_str) in [[X_train,Y_train, X0_train, "train"], [X_test, Y_test, X0_test, "test"]]
		    num_cols = 8
	    	plots_len = 2
	    	all_sampls = size(test_x)[end]
		    fig = figure(figsize=(25, 5)); 
		    for (i,ind) in enumerate((3:div(all_sampls,3):all_sampls)[1:plots_len])
		    	x0 = test_x0[:,:,1,ind] 
			    x_gt = test_x[:,:,:,ind:ind] 
			    y = test_y[:,:,:,ind:ind]
			    y .+= noise_lev_y*randn(Float32, size(y));

			    # make samples from posterior for train sample 
			   	X_post = posterior_sampler(G,  y, x_gt; device=device, batch_size=batch_size,num_samples=num_post_samples)|> cpu
			   	x_gt   = x_gt[:,:,1,1]
			   	if use_dm
				   	X_post = x0 .+ X_post
				   	x_gt   = x0 .+ x_gt[:,:,1,1]
				  end

				  if use_m
				   	X_post = sqrt.(1f0 ./ abs.(X_post))
				   	x_gt   = sqrt.(1f0 ./ x_gt[:,:,1,1])
				   	x0   = sqrt.(1f0 ./ x0)
				  end

			    X_post_mean = mean(X_post,dims=4)
			    X_post_std  = std(X_post, dims=4)

			    x_hat =  X_post_mean[:,:,1,1]
			    error_mean = abs.(x_hat-x_gt)

			    ssim_i = round(assess_ssim(x_hat, x_gt[:,:,1,1]), digits=2)
		    	rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)

			    y_plot = y[:,:,1,1]
			    a = quantile(abs.(vec(y_plot)), 98/100)

			    subplot(plots_len,num_cols,(i-1)*num_cols+1); imshow(y_plot, vmin=-a,vmax=a,interpolation="none", cmap="gray")
					axis("off"); title(L"rtm");#colorbar(fraction=0.046, pad=0.04);

					subplot(plots_len,num_cols,(i-1)*num_cols+2); imshow(x0', vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
					axis("off"); title(L"$\mathbf{x}_0$") ; #colorbar(fraction=0.046, pad=0.04)

			    subplot(plots_len,num_cols,(i-1)*num_cols+3); imshow(X_post[:,:,1,1],vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
					axis("off"); title("Posterior sample") #colorbar(fraction=0.046, pad=0.04);
				
					subplot(plots_len,num_cols,(i-1)*num_cols+4); imshow(X_post[:,:,1,2],vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
					axis("off");title("Posterior sample")  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")

					subplot(plots_len,num_cols,(i-1)*num_cols+5); imshow(x_gt,  vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
					axis("off"); title(L"Reference $\mathbf{x^{*}}$") ; #colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+6); imshow(x_hat ,vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
					axis("off"); title("Posterior mean | SSIM="*string(ssim_i)) ; #colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+7); imshow(error_mean , vmin=0,vmax=nothing, interpolation="none", cmap="magma")
					axis("off");title("Error | RMSE="*string(rmse_i)) ;# cb = colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+8); imshow(X_post_std[:,:,1,1] , vmin=0,vmax=nothing,interpolation="none", cmap="magma")
					axis("off"); title("Standard deviation") ;#cb =colorbar(fraction=0.046, pad=0.04)
			end
			tight_layout()
		  fig_name = @strdict decay_step decay use_dm use_m  flip_aug vmax_d n_src unet_lev clipnorm_val noise_lev_x n_train  e lr n_hidden L K batch_size
		  safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_"*file_str*".png"), fig); close(fig)
		end
		
	    ############# Training metric logs
		if e != plot_every
			sum_train = loss + logdet_train
			sum_test = loss_test + logdet_test

			fig = figure("training logs ", figsize=(10,12))
			subplot(5,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
			plot(range(0f0, 1f0, length=length(loss)), loss, label="train");
			plot(range(0f0, 1f0, length=length(loss_test)),loss_test, label="test"); 
			axhline(y=1,color="red",linestyle="--",label="Normal Noise")
			ylim(bottom=0.,top=1.5)
			xlabel("Parameter Update"); legend();

			subplot(5,1,2); title("Logdet Term: train="*string(logdet_train[end])*" test="*string(logdet_test[end]))
			plot(range(0f0, 1f0, length=length(logdet_train)),logdet_train);
			plot(range(0f0, 1f0, length=length(logdet_test)),logdet_test);
			xlabel("Parameter Update") ;

			subplot(5,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
			plot(range(0f0, 1f0, length=length(sum_train)),sum_train); 
			plot(range(0f0, 1f0, length=length(sum_test)),sum_test); 
			xlabel("Parameter Update") ;

			subplot(5,1,4); title("SSIM train=$(ssim[end]) test=$(ssim_test[end])")
		    plot(range(0f0, 1f0, length=length(ssim)),ssim); 
		    plot(range(0f0, 1f0, length=length(ssim_test)),ssim_test); 
		    xlabel("Parameter Update") 

		    subplot(5,1,5); title("RMSE train=$(l2_cm[end]) test=$(l2_cm_test[end])")
		    plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
		    plot(range(0f0, 1f0, length=length(l2_cm_test)),l2_cm_test); 
		    xlabel("Parameter Update") 

			tight_layout()
			fig_name = @strdict decay_step decay use_dm use_m   flip_aug n_src unet_lev  clipnorm_val noise_lev_x noise_lev_y n_train e lr  n_hidden L K batch_size
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end

	end
	 #save params every 4 epochs
    if(mod(e,save_every)==0) 
       # Saving parameters and logs
			unet_model = G.sum_net.model|> cpu;
			G_save = deepcopy(G);
			reset!(G_save.sum_net); # clear params to not save twice
			Params = get_params(G_save) |> cpu;
			save_dict = @strdict decay_step decay use_dm chan_cond  use_m iter n_src unet_lev unet_model  clipnorm_val n_train e noise_lev_x noise_lev_y lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size; 

			@tagsave(
			joinpath("/slimdata/rafaeldata/savednets", savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
			);
    end
end