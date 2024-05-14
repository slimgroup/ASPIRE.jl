export tone_burst, circle_geom

function gaussian(x, magnitude, mean, variance)
	magnitude .* exp.(.-(x .- mean).^2f0 ./ (2f0 .* variance));
end

function tone_burst(sample_freq, signal_freq, num_cycles; signal_length=nothing)
	# calculate the temporal spacing
	dt = 1f0 / sample_freq;

	tone_length = num_cycles / (signal_freq);
	tone_t = 0f0:dt:tone_length;
	tone_burst = sin.(2f0 * pi * signal_freq * tone_t);

	#apply the envelope
	x_lim = 3f0;
	window_x = (-x_lim):( 2f0 * x_lim / (length(tone_burst) - 1f0) ):x_lim;
	window = gaussian(window_x, 1f0, 0f0, 1f0);
	tone_burst = tone_burst .* window;

	if ~isnothing(signal_length)
		signal_full = zeros(Float32, signal_length)
	else 
		signal_full = zeros(Float32, length(tone_burst))
	end

	signal_full[1:length(tone_burst)] = tone_burst
	signal_full / maximum(signal_full)
end

function circle_geom(h, k, r, numpoints)
       #h and k are the center coordes
       #r is the radius
       theta = LinRange(0f0, 2*pi, numpoints+1)[1:end-1]
       Float32.(h .- r*cos.(theta)), Float32.(k .+ r*sin.(theta)), theta
end