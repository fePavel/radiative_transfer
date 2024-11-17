using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie

number_of_dimensions = 2

@agent struct H_atom_3d(ContinuousAgent{3, Float64}) 
    speed::Float64
    clump_id::Int
    clump_center_x::Float64
    clump_center_y::Float64
    clump_center_z::Float64
end

@agent struct H_atom_2d(ContinuousAgent{2,Float64})
    speed::Float64
    clump_id::Int
    clump_center_x::Float64
    clump_center_y::Float64
end


function initialize_model(; number_of_atoms=1000, speed = 0.03, extent = Tuple(ones(number_of_dimensions)), number_of_clumps=1, seed = 42)
    space = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)
    if number_of_dimensions == 2 
        H_atom = H_atom_2d
    elseif number_of_dimensions == 3 
        H_atom = H_atom_3d
    end

    properties = Dict(:n => round(Int, number_of_clumps^(1 / number_of_dimensions)),
                      :number_of_clumps => number_of_clumps)

    model = StandardABM(H_atom, space; rng, agent_step!, scheduler=Schedulers.ByID(), properties = properties)
    n=round(Int, number_of_clumps^(1/number_of_dimensions))
    cell_size = extent ./ (2n+1)
    for k in 1:2n+1
        for j in 1:2n+1
            if k % 2 == 0 && j % 2 == 0
                for i in 1:number_of_atoms
                clump_id = n * (k - 1) + j
                vel = randn(rng, Float64, (number_of_dimensions, 1)) .* speed
                pos = (rand(rng, Float64, (number_of_dimensions, 1)) .+ (k-1, j-1)) .* cell_size
                add_agent!(
                    pos,
                    model,
                    vel,
                    speed,
                    clump_id,
                    (k - 0.5) .* cell_size[1],
                    (j - 0.5) .* cell_size[2]
                )
                end

            end
        end
    end
    return model
end


function agent_step!(H_atom, model; clumps_bounds=true)
    # println(model.space_size)
    # println(model.properties[:n])
    n = model.n
    cell_size = spacesize(model)[1] / (2n+1)
    move_agent!(H_atom, model, H_atom.speed)
    next_pos_x, next_pos_y = H_atom.pos 

    clump_center_x = H_atom.clump_center_x
    clump_center_y = H_atom.clump_center_y

    if clumps_bounds
        if next_pos_x > clump_center_x + 0.5 * cell_size
            # H_atom.pos[1] -= cell_size
            walk!(H_atom, SVector((-cell_size, 0)), model)
        end
        if next_pos_x < clump_center_x - 0.5 * cell_size
            # H_atom.pos[1] += cell_size
            walk!(H_atom, SVector((cell_size, 0)), model)
        end
        if next_pos_y > clump_center_y + 0.5 * cell_size
            # H_atom.pos[2] -= cell_size
            walk!(H_atom, SVector((0, -cell_size)), model)
        end
        if next_pos_y < clump_center_y - 0.5 * cell_size
            # H_atom.pos[2] += cell_size
            walk!(H_atom, SVector((0, cell_size)), model)
        end
    end
    # clumps_vels_x = rand(Random.MersenneTwister(239), model.number_of_clumps)
    # clumps_vels_y = rand(Random.MersenneTwister(566), model.number_of_clumps)
    H_atom.clump_center_x += 0.001
    # H_atom.clump_center_y += clumps_vels_y[H_atom.clump_id] * 0.01

end



### interactive mode ####
model = initialize_model()
fig, ax, abmobs = abmplot(model; add_controls=true, agent_size=2)
fig

# agent_step!(model[1], model)

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

