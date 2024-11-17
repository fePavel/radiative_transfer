using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie

@agent struct test_agent(ContinuousAgent{2,Float64})
    # speed::Float64
    clump_id::Int
end

function agent_step!(test_agent, model)
    # H_atom.vel = H_atom.vel
    println(test_agent.pos)
    move_agent!(test_agent, model)
    println(test_agent.pos)
end

model = StandardABM(test_agent, ContinuousSpace((1, 1)); agent_step!)
pos = (1,1)
vel = (1,1)
add_agent!((0.3,1), model, vel, 1)


model[1].pos = [0.3, 0.4]
agent_step!(model[1], model)

spacesize(model)[1]

