using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie

number_of_dimensions = 2

@agent struct H_atom_3d(ContinuousAgent{3, Float64}) 
    speed::Float64
    clump_id::Int
end

@agent struct H_atom_2d(ContinuousAgent{2,Float64})
    speed::Float64
    clump_id::Int
end

@agent struct Clump(ContinuousAgent{2,Float64})
    cell_size::Float64
    child_agents::Array{Union{H_atom_2d, H_atom_3d},1}
end

function initialize_model(; number_of_atoms=100, speed = 0.03, extent = Tuple(ones(number_of_dimensions)), number_of_clumps=4, seed = 42)
    space = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)
    if number_of_dimensions == 2 
        H_atom = H_atom_2d
    elseif number_of_dimensions == 3 
        H_atom = H_atom_3d
    end

    properties = Dict(:n => round(Int, number_of_clumps^(1 / number_of_dimensions)),
                      :number_of_clumps => number_of_clumps)

    model = StandardABM(Union{H_atom, Clump}, space; rng, agent_step!, scheduler=Schedulers.ByID(), properties = properties)
    n=round(Int, number_of_clumps^(1/number_of_dimensions))
    cell_size = extent ./ (2n+1)
    base_clump = Clump(1, (0.5, 0.5), (0.0, 0.0), cell_size[1], Int[])
    for i in 1:2n+1
        for j in 1:2n+1
            if i % 2 ==0 && j % 2 == 0 
                clump_pos = [(i - 0.5), (j - 0.5)] .* cell_size
                vel = 1 .* randn(rng, Float64, (number_of_dimensions, 1)) .* 0.001
                replicate!(base_clump, model; pos=clump_pos, vel=vel, child_agents=Int[])
            end
        end
    end

    base_H_atom = H_atom(number_of_clumps+1, (0.5, 0.5), (0.0, 0.0), speed, 1)
    for i in 1:number_of_clumps
        for j in 1:number_of_atoms
            pos = (rand(rng, Float64, (number_of_dimensions, 1)) .- (0.5, 0.5)) .* cell_size .+ model[i].pos 
            vel = randn(rng, Float64, (number_of_dimensions, 1)) .* speed .+ model[i].vel
            H_atom = replicate!(base_H_atom, model; pos=pos, vel=vel, clump_id=i)
            push!(model[i].child_agents, H_atom)
        end
    end
    return model
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

function agent_step!(test_agent::Union{H_atom_2d,H_atom_3d}, model)
    cell_size = model[test_agent.clump_id].cell_size
    prev_dist = dist(model[test_agent.clump_id], test_agent, model)
    if prev_dist[1] > 0.5 * cell_size || prev_dist[2] > 0.5 * cell_size
        return 0
    end

    move_agent!(test_agent, model, test_agent.speed)
    next_dist = dist(model[test_agent.clump_id], test_agent, model)

    if next_dist[1] > 0.5 * cell_size && test_agent.vel[1] > 0
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

function agent_step!(clump_agent::Clump, model)
    cell_size = clump_agent.cell_size
    move_agent!(clump_agent, model, 1)

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


# function agent_step!(H_atom, model; clumps_bounds=true)
#     H_atom.clump_center_x += H_atom.clump_vel_x
#     # println(model.space_size)
#     # println(model.properties[:n])
#     n = model.n
#     cell_size = spacesize(model)[1] / (2n+1)
#     prev_pos_x, prev_pos_y = H_atom.pos 
#     move_agent!(H_atom, model, H_atom.speed)
#     next_pos_x, next_pos_y = H_atom.pos 

#     clump_center_x = H_atom.clump_center_x
#     clump_center_y = H_atom.clump_center_y

#     if clumps_bounds

#         # preiodic bounds for clumps
#         if next_pos_x > clump_center_x + 0.5 * cell_size 
#             # && next_pos_x - cell_size > 0 
#             walk!(H_atom, SVector((-cell_size, 0)), model)
#         end
#         if next_pos_x < clump_center_x - 0.5 * cell_size 
#             # && next_pos_x + cell_size < spacesize(model)[1]
#             walk!(H_atom, SVector((cell_size, 0)), model)
#         end
#         if next_pos_y > clump_center_y + 0.5 * cell_size 
#             # && next_pos_y - cell_size > 0
#             walk!(H_atom, SVector((0, -cell_size)), model)
#         end
#         if next_pos_y < clump_center_y - 0.5 * cell_size 
#             # && next_pos_y + cell_size < spacesize(model)[2]
#             walk!(H_atom, SVector((0, cell_size)), model)
#         end
    
#         # move clump_center when particle intersects an edge
#         if H_atom.vel[1] > 0 && next_pos_x - prev_pos_x < 0
#             H_atom.clump_center_x -= spacesize(model)[1]
#         end
#         if H_atom.vel[1] < 0 && next_pos_x - prev_pos_x > 0
#             H_atom.clump_center_x += spacesize(model)[1]
#         end
#         if H_atom.vel[2] > 0 && next_pos_y - prev_pos_y < 0
#             H_atom.clump_center_y -= spacesize(model)[2]
#         end
#         if H_atom.vel[2] < 0 && next_pos_y - prev_pos_y > 0
#             H_atom.clump_center_y += spacesize(model)[2]
#         end
#     end
# end




function agent_color(agent::Union{H_atom_2d, H_atom_3d})
    return :blue
end
function agent_color(agent::Clump)
    return :red
end


### interactive mode ####
model = initialize_model()
fig, ax, abmobs = abmplot(model; add_controls=true, agent_size=4, agent_color=agent_color)
fig

# model[3].child_agents
# nagents(model)
# model = initialize_model()
# for i in 1:1000
#     agent_step!(model[1], model)
# end

# n = 4
# m = zeros(2n + 1, 2n + 1)
# for i in 1:2n+1
#     for j in 1:2n+1
#         if i % 2 == 0 && j % 2 == 0
#             m[i, j] = 1
#         end
#     end
# end

# m

# abmspace(model).

