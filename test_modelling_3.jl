using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie

@agent struct test_agent(ContinuousAgent{2,Float64})
    # speed::Float64
    clump_id::Int
end

model = StandardABM(test_agent, ContinuousSpace((1, 1)))
pos = (1,1)
vel = (1,1)
add_agent!((0.3,1), model, vel, 1)

model[1]
