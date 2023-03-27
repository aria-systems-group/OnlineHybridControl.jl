function get_discrete_state_idx(ocp::OnlineSynthesisProblem, state)
    # Loop over and get the state idx
    for (idx, discrete_state) in ocp.state_extents
        if discrete_state[1] <= state <= discrete_state[2]          # TODO: Generalize to ND
            return idx
        end
    end
    @warn "Unknown state $state, marking as unsafe"
    return -1
end

"""
Execute one step of the online synthesis instance.
"""
function step(osp::OnlineSynthesisProblem, osr::OnlineSynthesisResults, step::Int; verbose=false)
    # 0. Get the Next Best Action and Metric
    control, metric, best_interval = online_strategy(osp, osr, step)
    osr.actions[step] = control
    osr.metrics[step] = metric

    # 1. Get the Next State
    osr.system_states[step+1] = osp.system_fcn(osr.system_states[:,step][1], control)         # Replace with true dynamics fcn

    # 2. Use the Next State to Get the next Discrete State
    osr.discrete_states[step+1] = get_discrete_state_idx(osp, osr.system_states[:,step+1][1])

    # 3. Check if a terminating condition has been reached
    # TODO: Other terminations besides steps?
    if osr.discrete_states[step+1] < 0
        return "left safe set"
    end

    if step == osp.max_steps
        return "max_steps"
    end

    # 4. Update abstractions with new data - how can I add this in an easy way? 
    new_x = [osr.system_states[:,step][1], control]
    new_y =  osr.system_states[:,step+1][1] - osr.system_states[:,step][1]
    osr.other_data["x_data"] = hcat(osr.other_data["x_data"], new_x)
    push!(osr.other_data["y_data"], new_y)
    GaussianProcesses.fit!(osr.other_data["gp"], osr.other_data["x_data"], osr.other_data["y_data"])

    if mod(step, 10) == 0            # Update GP every $N steps
        mZero = MeanZero()                   # Zero mean function
        kern = Poly(0.9, 0.0, 2)             # Squared exponential kernel (note that hyperparameters are on the log scale)
        logObsNoise = log10(0.1^2)
        osr.other_data["gp"] = GP(osr.other_data["x_data"], osr.other_data["y_data"], mZero, kern, logObsNoise)       #Fit the GP, TODO: hyperparameter optimization?  
        optimize!(osr.other_data["gp"], kernbounds=[[-4.0, -0.00001], [4.0, 0.00001]], noise=false)
    end

    # Update all state σ2 bounds (approximately)
    for i=1:length(osp.state_extents)
        for (control_idx, interval) in osr.other_data["control_intervals"]
            state_ex = osp.state_extents[i]
            _, _, σ2_ub = compute_σ2_ub_bounds_approx(osr.other_data["gp"], [state_ex[1], interval[1]], [state_ex[2], interval[2]])
            osr.other_data["sigma_bounds"][(i, interval)] = σ2_ub
            osr.other_data["sigma_bounds_mat"][i, control_idx] = σ2_ub 
        end
    end

    ## Recompute the barrier for non-safe states with the same control input
    if mod(step, osr.other_data["services"][1]["frequency"]) == 0
        # Get the control idx from the interval
        idx = -1
        for (cidx, interval) in osr.other_data["control_intervals"]
            if best_interval == interval
                idx = cidx 
                break
            end
        end
        for state_idx in keys(osp.state_extents)
            ps = osr.other_data["services"][1]["function"](osr.other_data["gp"], state_idx, idx, osr.other_data["invariant_sets"], osr.other_data["sigma_bounds_mat"])
            osr.other_data["P_safe"][state_idx, idx] = max(ps, osr.other_data["P_safe"][state_idx, idx])    # Only keep new barrier if it is better

            if ps > 0.99
                if state_idx ∉ keys(osp.safe_actions) 
                    verbose && @info "Found a new safe state $state_idx with interval $best_interval."
                    push!(osr.other_data["invariant_sets"], state_idx)
                    osp.safe_actions[state_idx] = [best_interval]
                else
                    verbose && @info "Found a new interval $best_interval for safe state $state_idx."
                    push!(osp.safe_actions[state_idx], best_interval)
                end
            end
        end

        # Then, loop over all the controls for a certain state
        state_idx = osr.discrete_states[step+1] 
        for (idx, control_interval) = osr.other_data["control_intervals"]
            if state_idx ∈ keys(osp.safe_actions) && (control_interval ∈ osp.safe_actions[state_idx] || control_interval == best_interval)
                continue
            end
            ps = osr.other_data["services"][1]["function"](osr.other_data["gp"], state_idx, idx, osr.other_data["invariant_sets"], osr.other_data["sigma_bounds_mat"])
            osr.other_data["P_safe"][state_idx, idx] = max(ps, osr.other_data["P_safe"][state_idx, idx])    # Only keep new barrier if it is better

            if ps > 0.99
                if state_idx ∉ keys(osp.safe_actions) 
                    verbose && @info "Found a new safe state $state_idx with interval $control_interval."
                    push!(osr.other_data["invariant_sets"], state_idx)
                    osp.safe_actions[state_idx] = [control_interval]
                else
                    verbose && @info "Found a new interval $control_interval for safe state $state_idx."
                    push!(osp.safe_actions[state_idx], control_interval)
                end
            end
        end 
    end


    return nothing
end

"""
Execute a full online synthesis realization.
"""
function run(osp::OnlineSynthesisProblem, osic::OnlineSynthesisInitialConditions; verbose=false) # What is the most appropriate function name?
    # In this case, run means "realize one possible instance of the osp (which is stochastic)"

    # 0. Set Initial conditions
    osr = initialize(osp, osic)

    # 1. Run the online control problem until termination is reached
    for i=1:osp.max_steps
        # 1a. Run a step of the osp
        termination_value = step(osp, osr, i, verbose=verbose)
        # 2. Break
        if !isnothing(termination_value)
            osr.termination_step = i+1
            osr.termination_value = termination_value
            break
        end
    end
    
    cleanup!(osr)
    return osr
end