#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=20G srun --pty julia 


using DrWatson
@quickactivate :IterPhySum
import Pkg; Pkg.instantiate()

using PyPlot, SlimPlotting
using LinearAlgebra
using Random
using JLD2, BSON
using Statistics
using ImageQualityIndexes

# Plotting configs
sim_name = "ip-paper"
plot_path = joinpath("/slimdata/rafaeldata/plots/IterPhySum",sim_name)

intervals  = [(1.48,1.5782),(1.5782,2.8)]
vmin_brain = intervals[1][1]
vmax_brain = intervals[2][2]
cmap_types = [ColorMap("cet_CET_L1"),matplotlib.cm.Reds]
cmap       = create_multi_color_map(intervals, cmap_types)
cbarticks = [1.48,1.5782,2.8]

font_size=18
PyPlot.rc("font", family="serif", size=font_size); PyPlot.rc("xtick", labelsize=font_size); PyPlot.rc("ytick", labelsize=font_size);
PyPlot.rc("axes", labelsize=font_size)    # fontsize of the x and y labels

fac = 1000f0
@load "/slimdata/rafaeldata/savednets/e=1.jld2" X_post_4 X_post_3 X_post_2 X_post x_gt
indx = 24

#Load in hint2 samples 
using InvertibleNetworks
m_i_path = "/slimdata/rafaeldata/savednets_hint2/batch_size=32_batchsize_src=2_clipnorm_val=3.0_e=200_factor=0.0_freq=150_indx=24_inner_loop=64_lr=5e-5_resample=false_z_inv=true.bson"
X_post_hint2 = BSON.load(m_i_path)["X_post"];

#Load in guash uq and mean
data_path_gaush = "/slimdata/rafaeldata/savedfwi/batchsize=16_freq=70_gaush=true_grad_norm=false_ind=1038_j=400_line_search=false_starting_alpha=1e8_total_pde=800.jld2"
X_mean_gaush = fac.*sqrt.(1f0 ./ JLD2.jldopen(data_path_gaush, "r")["m_final"])
X_std_guash_base =  JLD2.jldopen(data_path_gaush, "r")["sigma_diag"]


#mse_i = mean(error_mean_gaush.^2) 
#var_i = mean(uq_gaush.^2)
#scaling_guash = mse_i / var_i 



X_std_1 = fac .*std(X_post;dims=4)
X_std_2 = fac .*std(X_post_2;dims=4)
X_std_3 = fac .*std(X_post_3;dims=4)
X_std_4 = fac .*std(X_post_4;dims=4)
X_std_hint2 = fac .*std(X_post_hint2;dims=4)


X_mean_1 = fac .*mean(X_post;dims=4)
X_mean_2 = fac .*mean(X_post_2;dims=4)
X_mean_3 = fac .*mean(X_post_3;dims=4)
X_mean_4 = fac .*mean(X_post_4;dims=4)
X_mean_hint2 = fac .*mean(X_post_hint2;dims=4)

error_mean_1 = abs.(X_mean_1 - fac .*x_gt)
error_mean_2 =  abs.(X_mean_2 - fac .*x_gt)
error_mean_3 =  abs.(X_mean_3 - fac .*x_gt)
error_mean_4 =  abs.(X_mean_4 - fac .*x_gt)
error_mean_hint2 =  abs.(X_mean_hint2 - fac .*x_gt)
error_mean_gaush =  abs.(X_mean_gaush - fac .*x_gt)



ind_train = 1000
ind = ind_train+15+indx-1
@load "/slimdata/rafaeldata/brain_cond_data/gauss_freq_brain_full_water_no_rho_grad_no_isic_iter_1_ind_"*string(ind)*".jld2" n o d q  m dobs_noisy;
x_gt  =  sqrt.(1f0./(m));

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


crop_top = 75
crop_side = 55
flip_range = (crop_side+9:(512-crop_side)+9,crop_top+5:(512-crop_top)+5,1:1,:)


cbar_frac = 0.052
cbar_pad = 0.01

vmin_std = 0
vmax_std = 50

fig = figure(figsize=(6, 6))
imshow((-1.4f0*(1 .- mask_brain[flip_range...]) .*  X_std_hint2[flip_range...] +  2f0 .* X_std_hint2[flip_range...])[:,:,1,1] ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off"); tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_std_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_std_hint_2_red.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(6, 6))
imshow((-1.4f0*(1 .- mask_brain[flip_range...]) .*  X_std_4[flip_range...] +  2f0 .* X_std_4[flip_range...])[:,:,1,1] ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off"); tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_std_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_std_4_red.png"), bbox_inches="tight", dpi=600)


scaling_guash = fac*100f0
X_std_guash = scaling_guash*X_std_guash_base
fig = figure(figsize=(6, 6))
imshow((-1.4f0*(1 .- mask_brain[flip_range...]) .*  X_std_guash[flip_range...] +  2f0 .* X_std_guash[flip_range...])[:,:,1,1] ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off"); tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_std_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_std_gaush_red.png"), bbox_inches="tight", dpi=600)



fig = figure(figsize=(6, 6))
imshow((-0.75f0*(1 .- mask_brain[flip_range...]) .*  error_mean_hint2[flip_range...] + error_mean_hint2[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off");tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(6, 6))
imshow((-0.75f0*(1 .- mask_brain[flip_range...]) .*  error_mean_4[flip_range...] + error_mean_4[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off");tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_error_4_red.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(6, 6))
imshow((-0.75f0*(1 .- mask_brain[flip_range...]) .*  error_mean_gaush[flip_range...] + error_mean_gaush[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off");tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_error_gaush_red.png"), bbox_inches="tight", dpi=600)




num_post_plot = 4
for i in 1:num_post_plot 
	fig = figure(figsize=(6, 6))
	imshow(X_post[:,:,1,i],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	axis("off"); tight_layout()
	fig_name = @strdict indx i  
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_iter_1.png"), fig); close(fig)
end

for i in 1:num_post_plot 
	fig = figure(figsize=(6, 6))
	imshow(X_post_2[:,:,1,i],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	axis("off"); tight_layout()
	fig_name = @strdict indx i  
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_iter_2.png"), fig); close(fig)
end

for i in 1:num_post_plot 
	fig = figure(figsize=(6, 6))
	imshow(X_post_3[:,:,1,i],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	axis("off"); tight_layout()
	fig_name = @strdict indx i  
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_iter_3.png"), fig); close(fig)
end


for i in 1:num_post_plot 
	fig = figure(figsize=(6, 6))
	imshow(X_post_4[:,:,1,i],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	axis("off"); tight_layout()
	fig_name = @strdict indx i  
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_iter_4.png"), fig); close(fig)
end

for i in 1:num_post_plot 
	fig = figure(figsize=(6, 6))
	imshow(X_post_hint2[:,:,1,i],interpolation="none",vmin=vmin_brain,vmax=vmax_brain,cmap=cmap)
	axis("off"); tight_layout()
	fig_name = @strdict indx i  
	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_hint2.png"), fig); close(fig)
end




num_cols=4
num_rows=1

fig = figure(figsize=(9, 5.25))
fig.patch.set_facecolor("black")

subplot(num_rows,num_cols,1);imshow(X_post[flip_range...][:,:,1,1], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,2); imshow(X_post[flip_range...][:,:,1,2], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,3); imshow(X_post[flip_range...][:,:,1,3], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,4); imshow(X_post[flip_range...][:,:,1,4], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 

plt.subplots_adjust(wspace=0, hspace=0)
fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_posts_iter_1.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(9, 5.25))
fig.patch.set_facecolor("black")

subplot(num_rows,num_cols,1);imshow(X_post_2[flip_range...][:,:,1,1], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,2); imshow(X_post_2[flip_range...][:,:,1,2], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,3); imshow(X_post_2[flip_range...][:,:,1,3], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,4); imshow(X_post_2[flip_range...][:,:,1,4], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 

plt.subplots_adjust(wspace=0, hspace=0)
fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_posts_iter_2.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(9, 5.25))
fig.patch.set_facecolor("black")

subplot(num_rows,num_cols,1);imshow(X_post_3[flip_range...][:,:,1,1], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,2); imshow(X_post_3[flip_range...][:,:,1,2], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,3); imshow(X_post_3[flip_range...][:,:,1,3], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,4); imshow(X_post_3[flip_range...][:,:,1,4], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 

plt.subplots_adjust(wspace=0, hspace=0)
fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_posts_iter_3.png"), bbox_inches="tight", dpi=600)

fig = figure(figsize=(9, 5.25))
fig.patch.set_facecolor("black")

subplot(num_rows,num_cols,1); imshow(X_post_4[flip_range...][:,:,1,1], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,2); imshow(X_post_4[flip_range...][:,:,1,2], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,3); imshow(X_post_4[flip_range...][:,:,1,3], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 
subplot(num_rows,num_cols,4); imshow(X_post_4[flip_range...][:,:,1,4], vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); axis("off"); 

plt.subplots_adjust(wspace=0, hspace=0)
fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_posts_iter_4.png"), bbox_inches="tight", dpi=600)









num_cols=4
num_rows=1

vmin_std = 0
vmax_std = 50

cbar_frac = 0.0524
cbar_pad = 0.01

fig = figure(figsize=(21, 6))
#fig.patch.set_facecolor("black")

subplot(num_rows,num_cols,1); 
imshow((-1.6f0*(1 .- mask_brain[flip_range...]) .*  X_std_1[flip_range...] +  2f0 .* X_std_1[flip_range...])[:,:,1,1] ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3");axis("off"); 
subplot(num_rows,num_cols,2);
imshow((-1.6f0*(1 .- mask_brain[flip_range...]) .*  X_std_2[flip_range...] +  2f0 .* X_std_2[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3");axis("off"); 
subplot(num_rows,num_cols,3);
imshow((-1.4f0*(1 .- mask_brain[flip_range...]) .*  X_std_3[flip_range...] +  2f0 .* X_std_3[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3");axis("off"); 
subplot(num_rows,num_cols,4);
imshow((-1.4f0*(1 .- mask_brain[flip_range...]) .*  X_std_4[flip_range...] +  2f0 .* X_std_4[flip_range...])[:,:,1,1]  ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3");axis("off"); 
#cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

fig_name = @strdict  indx
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_hint2.png"), fig); close(fig)
#fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_stds.png"), bbox_inches="tight", dpi=600); close(fig);




facs = [-1.7,-1.7,-1.4,-1.4]
fig = figure(figsize=(17, 5))
gs = matplotlib.gridspec.GridSpec(1, num_cols + 2, width_ratios=[1, 1, 1, 1, 0.025,0.05],height_ratios=[1])  # Last column for colorbar

for i in 1:num_cols
    ax = subplot(gs[i])
    # Assume your data arrays (like X_std_1) are already defined and properly shaped
    data = eval(Meta.parse("(facs[$i] * (1 .- mask_brain[flip_range...]) .* X_std_$i[flip_range...] .+ 2 .* X_std_$i[flip_range...])[:, :, 1, 1]"))
    im = imshow(data, interpolation="none", vmin=vmin_std, vmax=vmax_std, cmap="cet_CET_L3")
    axis("off")
end

# Colorbar
#cbar_ax = subplot(gs[6])
cbar_frac = 0.01
cbar_pad = 0.01

# Create an axes divider for the last axes
# divider = axes_grid1.make_axes_locatable(ax)
# cbar_ax = divider.append_axes("right", size="5%", pad=0.05)


#cb = colorbar(cax=cbar_ax); cb.set_label("[meters/second]")
#cb.ax.set_aspect(40) 
#tight_layout()
subplots_adjust(wspace=0, hspace=0)
#subplots_adjust(left=0.0, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)

fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_stds.png"), bbox_inches="tight", dpi=600); close(fig);
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_post_hint2.png"), fig); close(fig)


fig = figure(figsize=(17, 5))
gs = matplotlib.gridspec.GridSpec(1, num_cols + 2, width_ratios=[1, 1, 1, 1, 0.025,0.05],height_ratios=[1])  # Last column for colorbar

for i in 1:num_cols
    ax = subplot(gs[i])
    # Assume your data arrays (like X_std_1) are already defined and properly shaped
    data = eval(Meta.parse("(-0.75f0*(1 .- mask_brain[flip_range...]) .*  error_mean_$i[flip_range...] + error_mean_$i[flip_range...])[:, :, 1, 1]"))
    im = imshow(data, interpolation="none", vmin=vmin_std, vmax=vmax_std, cmap="cet_CET_L3")
    axis("off")
end

subplots_adjust(wspace=0, hspace=0)

fig_name = @strdict  indx
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_errors.png"), bbox_inches="tight", dpi=600); close(fig);



fig = figure(figsize=(6, 6))
imshow(-0.75f0*(1 .- mask_brain[flip_range...]) .*  error_mean[flip_range...] + error_mean[flip_range...] ,interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L3")
#imshow(2f0 .* m_i_std[flip_range...],interpolation="none",vmin=vmin_std,vmax=vmax_std,cmap="cet_CET_L1")
cb = colorbar(fraction=cbar_frac, pad=cbar_pad); cb.set_label("[meters/second]")
axis("off");tight_layout()
fig_name = @strdict 
#safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), fig); close(fig)  
fig.savefig(joinpath(plot_path, savename(fig_name; digits=6)*"_error_hint_2_red.png"), bbox_inches="tight", dpi=600)





