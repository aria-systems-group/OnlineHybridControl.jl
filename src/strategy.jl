# Different strategies

function online_strategy(osp::OnlineSynthesisProblem, osr::OnlineSynthesisResults, step::Int)
    if osr.online_strategy == "information-gain"
        mode, metric, best_interval = max_information(osp, osr, step)
    else 
        @error "Invalid strategy:", osr.online_strategy
    end
    return mode, metric, best_interval
end

"""
Max information chooses actions based purely on their information value as (currently) defined by the sup. of the current state.
"""
function max_information(osp::OnlineSynthesisProblem, osr::OnlineSynthesisResults, step::Int) 
        # Get the current state and discrete state
        x_c = osr.system_states[:, step][1]             # TODO: Why does this need an explicit index now?
        q_c = osr.discrete_states[step] 
    
        sigma_bounds = osr.other_data["sigma_bounds"]   # Sigma bounds per discrete region 

        if q_c ∉ keys(osp.safe_actions)
            @warn "State $q_c has no safe actions, using the safest control interval."
            _, best_interval_idx = findmax(osr.other_data["P_safe"][q_c,:])
            best_interval = osr.other_data["control_intervals"][best_interval_idx]
            action_set = [best_interval]
        else
            action_set = osp.safe_actions[q_c]                   # Safe control action intervals
        end
        @assert !isempty(action_set)

        #==
        1. Find the best control interval based on uncertainty at current state. 
        - Also add metric for largest uncertainy at potential successor states
        ==#
        best_interval_metric = -Inf
        best_interval = nothing
        for a in action_set  # These are intervals 
            σ2_bound = sigma_bounds[(q_c,a)]
            if σ2_bound > best_interval_metric
                best_interval_metric = σ2_bound
                best_interval = a 
            end
        end

        #==
        2. Do GP Bounding over the control interval to find the point that will provide the maximum information! 
        ==#
        x_best, _, σ2_ub = compute_σ2_ub_bounds_approx(osr.other_data["gp"], [x_c, best_interval[1]], [x_c+1e-6, best_interval[2]])

        return x_best[2], σ2_ub, best_interval 
end

"""
Sampling-based approach to estimate the σ^2 upper bound.
"""
function compute_σ2_ub_bounds_approx(gp, x_L, x_U; 
    N=100, 
    twister = MersenneTwister(11), 
    x_samp_alloc = Matrix{Float64}(undef, length(x_L), N))

    max_σ2 = -Inf
    for i in eachindex(x_L) 
        @views x_samp_alloc[i,:] = rand(twister, Uniform(x_L[i], x_U[i]), 1, N)
    end

    x_best = nothing
    for x_col in eachcol(x_samp_alloc)
        _, σ2 = predict_f(gp, hcat(x_col))
        if σ2[1] > max_σ2
            max_σ2 = σ2[1]
            x_best = x_col
        end
    end
    return x_best, 0.0, max_σ2
end