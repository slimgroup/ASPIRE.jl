#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=50G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=30G srun --pty julia 

using DrWatson
@quickactivate :IterPhySum
import Pkg; Pkg.instantiate()

# using JLD2
# using Statistics 

# grad_train = zeros(Float32,512,512,1,1168) 
# m0_train = zeros(Float32,512,512,1,1168) 
# for i in 1:1168
# 	println("$(i)/$(1168)")
	
# 	@load "/slimdata/rafaeldata/brain_cond_data/brain_full_water_no_rho_no_isic_grad_iter_4_ind_"*string(i)*".jld2" Y_train m0  

# 	grad_train[:,:,:, i] = mean(Y_train;dims=3)
# 	m0_train[:,:,:, i] = m0'
# end

# @save "/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_4.jld2" grad_train m0_train 


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
        )[1] |> cpu;
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
sim_name = "cond-net-norho-iter-4"
plot_path = joinpath("/slimdata/rafaeldata/plots/IterPhySum",sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)

# Training hyperparameters 
device = gpu

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.01f0
noise_lev_y  = 0.0 

batch_size   = 4
n_epochs     = 200
num_post_samples = 32

save_every   = 25
plot_every   = 1
n_condmean   = 40
use_m = false
use_dm = false

iter = 4
n_src = 16

isic = false 

m_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_iter_1.jld2", "r")["m_train"];
# m0_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/new_grad_water_no_rho_iter_2.jld2", "r")["m0_train"];
# m0_train = permutedims(m0_train,(2,1,3,4))
# grad_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/new_grad_water_no_rho_iter_2.jld2", "r")["grad_train"];
m0_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_4.jld2", "r")["m0_train"];
m0_train = permutedims(m0_train,(2,1,3,4))
grad_train = JLD2.jldopen("/slimdata/rafaeldata/brain_cond_data/grad_water_no_rho_no_isic_iter_4.jld2", "r")["grad_train"];


 if !(use_m)
 	m_train = sqrt.(1f0 ./ m_train)
  m0_train = sqrt.(1f0 ./ m0_train)
 end

grad_train ./= quantile(abs.(vec(grad_train)),0.99)
vmax_d = maximum(grad_train)

ind_train = 1000

target_train =   m_train[:,:,:,1:ind_train];
X0_train     =  m0_train[:,:,:,1:ind_train];
Y_train      = grad_train[:,:,:,1:ind_train];
Y_train      = tensor_cat(Y_train,X0_train)

target_test = m_train[:,:,:,ind_train+15:ind_train+100];
X0_test     = m0_train[:,:,:,ind_train+15:ind_train+100];
Y_test      = grad_train[:,:,:,ind_train+15:ind_train+100];
Y_test      = tensor_cat(Y_test,X0_test)

n_x, n_y, chan_target, n_train = size(target_train)
n_train = size(target_train)[end]
N = n_x*n_y*chan_target
chan_obs   = size(Y_train)[end-1]
chan_cond  = 1

#convert to velocity for now but test with slowness later
max_y = maximum(Y_train)

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
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr));

# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = [];
loss_test = []; logdet_test  = []; ssim_test = []; l2_cm_test = [];

#pl = FlipX(0.5) 
#|>  FlipX(0.5)
#aug = augment(target_train[:,:,1,1], pl)

# fig = figure()
# imshow(aug)
# fig_name = @strdict 
# safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)


# fig = figure()
# imshow(target_train[end:-1:1,:,1,1])
# fig_name = @strdict 
# safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_.png"), fig); close(fig)

decay_every = 5
decay_start = 70
decay_fac = 1.1f0
noise_lev_x_min = 0.001f0

mag_noise = 1f0


crop_aug = true
flip_aug = true

n_x_t = 256
n_y_t = 256

G.forward(X_train[:, :, :, 1:batch_size] |> device, Y_train[:, :, :, 1:batch_size]|> device) 
using CUDA
for e=1:n_epochs # epoch loop
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 

	if (e > decay_start) && (mod(e,decay_every) ==0) && noise_lev_x > noise_lev_x_min
      global noise_lev_x /= decay_fac
  end
	
		@time begin
    for b = 1:n_batches # batch loop
    	#@time begin
	        # X_unaug = X_train[:, :, :, idx_e[:,b]];
	        # Y_unaug = Y_train[:, :, :, idx_e[:,b]];

	        # X = zeros(n_x,n_y,chan_target,batch_size)
	        # Y = zeros(n_x,n_y,chan_target,batch_size)
	        # for i in 1:batch_size
	        # 	for j in 1:
	        # 	X[:,:,1,i], Y[:,:,1,i] = augment(X_unaug[:,:,:,i] => Y_unaug[:,:,:,i], pl)
	        # end

    		  #G_prime = deepcopy(G)
    		  G_noise = deepcopy(G |>cpu)
	        params_noise = get_params(G_noise) 
	        params_prime = get_params(G) 
	        for p_i in 1:length(params_noise)
	        	params_noise[p_i].data = (mag_noise.*randn(Float32,size(params_noise[p_i].data)))
	          params_prime[p_i] = params_prime[p_i] + (params_noise[p_i] |>device)
	        end; 
	        
    			X_pre = X_train[:, :, :, idx_e[:,b]];
	        Y_pre = Y_train[:, :, :, idx_e[:,b]];

	        X = zeros(Float32,n_x_t,n_y_t,1,batch_size);
					Y = zeros(Float32,n_x_t,n_y_t,2,batch_size);

					for i in 1:batch_size
						ran_x = rand(collect(1:(n_x-n_x_t)))
						ran_y = rand(collect(1:(n_y-n_y_t)))
			    	X[:,:,:,i:i] = X_pre[ran_x:ran_x+n_x_t-1,ran_y:ran_y+n_y_t-1,:,i:i]
			    	Y[:,:,:,i:i] = Y_pre[ran_x:ran_x+n_x_t-1,ran_y:ran_y+n_y_t-1,:,i:i]

			    	if rand() > 0.5
			    		X[:,:,:,i:i] = X[end:-1:1,:,:,i:i]
			    		Y[:,:,:,i:i] = Y[end:-1:1,:,:,i:i]
				  	end
				  end


	        # if rand() > 0.5
	        # 	X = X[:,end:-1:1,:,:]
	        # 	Y = Y[:,end:-1:1,:,:]
	        # end

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

	        params = get_params(G) 
	        #params_prime = get_params(G_prime) 
	        for p_i in 1:length(params)
	          params[p_i] = params[p_i] - (params_noise[p_i]|>device)
	        end; 

	        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	              "; f l2 = ",  loss[end], 
	              "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	        Base.flush(Base.stdout)

	        GC.gc(true)
	        CUDA.reclaim()
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

	    num_cols = 8
	    plots_len = 2
	    for (test_x, test_y, test_x0, file_str) in [[X_train,Y_train, X0_train, "train"], [X_test, Y_test, X0_test, "test"]]
		   
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

					subplot(plots_len,num_cols,(i-1)*num_cols+2); imshow(x0, vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
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
		  fig_name = @strdict mag_noise decay_every decay_start decay_fac isic use_dm use_m  flip_aug vmax_d  noise_lev_x n_train e 
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
			fig_name = @strdict mag_noise decay_every decay_start decay_fac isic use_dm use_m  crop_aug flip_aug vmax_d  noise_lev_x n_train e 
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end

	end
	 #save params every 4 epochs
    if(mod(e,save_every)==0) 
        # Saving parameters and logs
    	save_dict = nothing;
     	unet_model = G.sum_net.model|> cpu;
        G_save = deepcopy(G);
        reset!(G_save.sum_net); # clear params to not save twice
		Params = get_params(G_save) |> cpu;
		global save_dict = @strdict crop_aug mag_noise decay_every decay_start decay_fac  use_dm iter unet_model  clipnorm_val n_train e noise_lev_x lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size; 

		@tagsave(
			joinpath("/slimdata/rafaeldata/savednets", savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
		global G = G |> device;
    end
end