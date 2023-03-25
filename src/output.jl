"""
Cleanup the online synthesis results.
"""
function cleanup!(osr::OnlineSynthesisResults)
    final_index = osr.termination_step 
    osr.system_states = osr.system_states[:,1:final_index].nzval
    osr.discrete_states = osr.discrete_states[1:final_index].nzval
    osr.actions = osr.actions[1:final_index].nzval
    osr.metrics = osr.metrics[1:final_index].nzval
end

"""
Save the online synthesis results.
"""
function save(filename::String, osr::OnlineSynthesisResults)
    FileIO.save("$filename.jld2", Dict(
                "system_states" => osr.system_states,
                "discrete_states" => osr.discrete_states,
                "actions" => osr.actions,
                "metrics" => osr.metrics,
                "termination_step" => osr.termination_step,
                "termination_value" => osr.termination_value,
                "other_data" => osr.other_data))
end