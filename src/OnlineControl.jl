module OnlineControl

using SparseArrays
using StaticArrays

using Random
using Distributions
using GaussianProcesses

using FileIO
using JLD2

struct OnlineSynthesisProblem 
    # Add the stuff here
    system_fcn
    state_extents
    safe_actions::Dict
    strategy 
    max_steps
end

"""
Validate the online synthesis problem structure.
"""
function validate(osp::OnlineSynthesisProblem)
    # TODO: Add validation checks
end


mutable struct OnlineSynthesisResults
    # Get the stuff specific to the instance here
    system_states
    discrete_states
    actions
    metrics
    termination_step
    termination_value
    online_strategy::String
    other_data::Dict
end

struct OnlineSynthesisInitialConditions
    system_state
    online_strategy::String
    other_data::Dict
end

include("online.jl")
include("strategy.jl")
include("output.jl")

export OnlineSynthesisProblem, OnlineSynthesisResults, OnlineSynthesisInitialConditions

"""
Initialize the synthesis instance with the problem and initial condition.
"""
function initialize(ocp::OnlineSynthesisProblem, ocic::OnlineSynthesisInitialConditions)
    validate(ocp)
    dim = length(ocic.system_state)
    max_length = ocp.max_steps + 1
    ocr = OnlineSynthesisResults(
        spzeros(dim, max_length),
        spzeros(Int, max_length),
        spzeros(max_length),
        spzeros(max_length),
        -1,
        "unterminated",
        ocic.online_strategy,
        ocic.other_data
    ) 
    # Set initial conditions here!
    ocr.system_states[:,1] = ocic.system_state
    ocr.discrete_states[1] = get_discrete_state_idx(ocp, ocr.system_states[:,1][1])
    return ocr
end

end