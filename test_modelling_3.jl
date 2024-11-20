using Agents
using Random, LinearAlgebra, Statistics
using GLMakie

@agent struct test_agent(ContinuousAgent{2,Float64})
    # speed::Float64
    clump_id::Int
end

@agent struct test_clump(ContinuousAgent{2,Float64})
    cell_size::Float64
    child_agents::Array{test_agent, 1}
end

function dist(agent_1, agent_2, model)
    spacesize(model)[1]
    dist_1 = min(abs(agent_1.pos[1] - agent_2.pos[1]), 
        abs(spacesize(model)[1] + agent_1.pos[1] - agent_2.pos[1]),
        abs(-spacesize(model)[1] + agent_1.pos[1] - agent_2.pos[1]))
    dist_2 = min(abs(agent_1.pos[2] - agent_2.pos[2]),
        abs(spacesize(model)[2] + agent_1.pos[2] - agent_2.pos[2]),
        abs(-spacesize(model)[2] + agent_1.pos[2] - agent_2.pos[2]))
    return dist_1, dist_2
end

function agent_step!(test_agent::test_agent, model)
    clump_center_x, clump_center_y = model[test_agent.clump_id].pos
    cell_size = model[test_agent.clump_id].cell_size

    dist_1, dist_2 = dist(model[test_agent.clump_id], test_agent, model)
    if dist_1 > 0.5 * cell_size || dist_2 > 0.5 * cell_size
       return 0
    end
        

    move_agent!(test_agent, model, 1)
    next_pos_x, next_pos_y = test_agent.pos

    next_dist = dist(model[test_agent.clump_id], test_agent, model)


    if next_dist[1] > 0.5* cell_size && test_agent.vel[1] > 0
        walk!(test_agent, SVector((-cell_size, 0)), model)
    end
    if next_dist[1] > 0.5 * cell_size && test_agent.vel[1] < 0
        walk!(test_agent, SVector((cell_size, 0)), model)
    end
    if next_dist[2] > 0.5 * cell_size && test_agent.vel[2] > 0
        walk!(test_agent, SVector((0, -cell_size)), model)
    end
    if next_dist[2] > 0.5 * cell_size && test_agent.vel[2] < 0
        walk!(test_agent, SVector((0, cell_size)), model)
    end
end

function agent_step!(clump_agent::test_clump, model)
    cell_size = clump_agent.cell_size
    prev_pos_x, prev_pos_y = clump_agent.pos
    prev_pos = clump_agent.pos
    move_agent!(clump_agent, model, 1)
    next_pos_x, next_pos_y = clump_agent.pos
    next_pos = clump_agent.pos

    for child_agent in clump_agent.child_agents
        next_dist = dist(clump_agent, child_agent, model)

        if next_dist[1] > 0.5 * cell_size && clump_agent.vel[1] > 0
            walk!(child_agent, SVector((cell_size, 0)), model)
        end
        if next_dist[1] > 0.5 * cell_size && clump_agent.vel[1] < 0
            walk!(child_agent, SVector((-cell_size, 0)), model)
        end
        if next_dist[2] > 0.5 * cell_size && clump_agent.vel[2] > 0
            walk!(child_agent, SVector((0, cell_size)), model)
        end
        if next_dist[2] > 0.5 * cell_size && clump_agent.vel[2] < 0
            walk!(child_agent, SVector((0, -cell_size)), model)
        end

    end
end


### interactive mode ####
model = StandardABM(Union{test_agent,test_clump}, ContinuousSpace((1, 1)); agent_step!, scheduler=Schedulers.ByID())

base_clump_agent = test_clump(1, (0.4, 0.4), (0.001 * randn(), 0.001 * randn()), 0.3, Int[])
# base_clump_agent = test_clump(1, (0.4, 0.4), (0.001, 0), 0.3, Int[])
replicate!(base_clump_agent, model)

base_agent = test_agent(2, (0.3, 0.5), vel, 1)
# replicate!(base_agent, model)
# replicate!(base_agent, model; pos=[0.42, 0.3])
# replicate!(base_agent, model; pos=[0.33, 0.4])
# replicate!(base_agent, model; pos=[0.51, 0.5])
N = 100
for i in 1:N
    replicate!(base_agent, model; pos=[0.4 + 0.1 * rand(), 0.4 + 0.1 * rand()], vel=(0.01 * randn(), 0.01 * randn()))
end
for i in 2:1+N
    push!(model[1].child_agents, model[i])
end

# add_agent!(clump_agent, (0.4, 0.4), model, (0.2, 0.2), 1)


# marker = GLMakie.Polygon(Point2f[(0, 0.2), (0.2, 0.2), (0.2, 0), (0, 0)])


function agent_color(agent::test_agent)
    return :blue
end
function agent_color(agent::test_clump)
    return :red
end


fig, ax, abmobs = abmplot(model; add_controls=true, agent_size=4, agent_color=agent_color)
fig


# model = StandardABM(test_agent, ContinuousSpace((1, 1)); agent_step!)
# pos = (1,1)
# vel = (1,1)
# add_agent!((0.3,1), model, vel, 1)


# model[1].pos = [0.3, 0.4]
# agent_step!(model[1], model)

# spacesize(model)[1]

