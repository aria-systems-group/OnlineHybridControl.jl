using Revise

using GaussianProcesses
using Random
using Distributions

include("src/OnlineHybridControl.jl")
using .OnlineControl       

#===
GP STUFF
===#
"""Initialize the GP
    1. Training data
        -  n: number of training points
        -  x: predictors """

Random.seed!(11)
n = 10; 
x = [rand(Uniform(19.,24.), 1, n);   rand(1, n)];
obs_noise = 0.01
logObsNoise = log10(obs_noise)

""" Dynamics function for thermostat
"""
τ = 5.   
αe = 8.0e-3
αH = 3.6e-3
Te = 15.0
Th = 55.0
σ_noise = 0.1  
f(x) = x[1] + τ*(αe*(Te-x[1]) + αH*(Th-x[1])*x[2]) + σ_noise*randn() # standard normal dist. with var σ^2 = 0.01
y = [f(xk) for xk in eachcol(x)]

# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = Poly(0.9, 0.0, 2)             # Squared exponential kernel (note that hyperparameters are on the log scale)

y_train = y - x[1,:]    # Essentially making training data zero-mean
gp = GP(x, y_train, mZero, kern, logObsNoise)       #Fit the GP, TODO: hyperparameter optimization?  
optimize!(gp, kernbounds=[[-2.0, -0.00001], [2.0, 0.00001]], noise=false)

#===
Get these cludgy functions outta here
===#
function parse_states(state_partitions_filename::String)
    discrete_states = Dict()
    open(state_partitions_filename) do f
        i = 1
        while !eof(f)
            line = readline(f)
            state = parse.(Float64, split(line))
            discrete_states[i] = state
            i += 1
        end
    end
    return discrete_states
end

state_filename = "data/state_partitions.txt"
discrete_states = parse_states(state_filename)

control_filename = "data/control_partitions.txt"
control_states = parse_states(control_filename)

invariant_set_filename = "data/invariant_sets.txt"
function parse_invariant_set(invariant_filename, control_states)
    safe_actions_dict = Dict()
    open(invariant_filename) do f
        while !eof(f)
            line = readline(f)
            state, control = parse.(Int, split(line))

            if state ∈ keys(safe_actions_dict)
                push!(safe_actions_dict[state], control_states[control])
            else
                safe_actions_dict[state] = [control_states[control]]
            end
        end
    end
    return safe_actions_dict
end

safe_actions = parse_invariant_set(invariant_set_filename, control_states)
strategy = nothing
max_steps = 500

ocp = OnlineControl.OnlineSynthesisProblem(
    (x,u) ->  x + τ*(αe*(Te-x) + αH*(Th-x)*u) + σ_noise*randn(), 
    discrete_states,
    safe_actions,
    strategy,
    max_steps
)

# Create the sigma_bounds dictionary
function bound_sigma(states_dict, controls_dict, invariant_set_dict, gp)

    sigma_bounds = Dict()

    for (state_idx, control_intervals) in invariant_set_dict
        for control_interval in control_intervals
            state_ex = states_dict[state_idx]
            # control_ex = controls_dict[control_idx]
            _, _, σ2_ub = OnlineControl.compute_σ_ub_bounds_approx(gp, [state_ex[1], control_interval[1]], [state_ex[2], control_interval[2]])
            sigma_bounds[(state_idx, control_interval)] = σ2_ub
        end
    end
    return sigma_bounds
end

sigma_bounds = bound_sigma(discrete_states, control_states, safe_actions, gp)

oci = OnlineControl.OnlineSynthesisInitialConditions(
    [20.6,],
    "information-gain",
    Dict(
        "sigma_bounds" => sigma_bounds,
        "gp" => gp,
        "x_data" => copy(x),
        "y_data" => copy(y_train),
        "control_intervals" => control_states
    )
)

osr = OnlineControl.OnlineControl.run(ocp, oci)
OnlineControl.save("test-run", osr)

#===
Plot Results
===#
using UnicodePlots

plt = UnicodePlots.lineplot(osr.system_states, title="Temperature")
display(plt)
plt = UnicodePlots.lineplot(cumsum(osr.metrics), title="Cumulative Sum σ^2") 
display(plt)
plt = UnicodePlots.scatterplot(osr.actions, title="Actions") 
display(plt)

using Plots; plotlyjs();
plt1 = Plots.plot(osr.system_states, 
                 label="", 
                 title="Temperature",
                 xlabel="",
                 ylabel="°C",
                 size=(400,200),
                 dpi=600)

plt2 = Plots.plot(osr.metrics, 
                 label="", 
                 title="",
                 xlabel="",
                 ylabel="Metric",
                 size=(400,200),
                 dpi=600)
                 

plt3 = Plots.scatter(osr.actions, 
                 label="", 
                 title="",
                 xlabel="Step",
                 ylabel="Control",
                 size=(400,200),
                 dpi=600
                 )
plt = Plots.plot(plt1, plt2, plt3, layout=(3,1), size=(400, 600), dpi=600)

width, height = plt.attr[:size]
Plots.prepare_output(plt)
using Plotly
PlotlyJS.savefig(Plots.plotlyjs_syncplot(plt), "test-run.png", width=width, height=height, scale=2)
