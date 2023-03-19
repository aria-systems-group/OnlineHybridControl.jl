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
function step(osp::OnlineSynthesisProblem, osr::OnlineSynthesisResults, step::Int)
    # 0. Get the Next Best Action and Metric
    control, metric = online_strategy(osp, osr, step)
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

    # Every so often update the up in the current state
    prev_discrete_state = osr.discrete_states[step][1]
    state_ex = osp.state_extents[prev_discrete_state]

    for (_, interval) in osr.other_data["control_intervals"]
        _, _, σ2_ub = compute_σ_ub_bounds_approx(osr.other_data["gp"], [state_ex[1], interval[1]], [state_ex[2], interval[2]])
        osr.other_data["sigma_bounds"][(prev_discrete_state, interval)] = σ2_ub
    end

    return nothing
end

"""
Execute a full online synthesis realization.
"""
function run(osp::OnlineSynthesisProblem, osic::OnlineSynthesisInitialConditions) # What is the most appropriate function name?
    # In this case, run means "realize one possible instance of the osp (which is stochastic)"

    # 0. Set Initial conditions
    osr = initialize(osp, osic)

    # 1. Run the online control problem until termination is reached
    for i=1:osp.max_steps
        # 1a. Run a step of the osp
        termination_value = step(osp, osr, i)
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