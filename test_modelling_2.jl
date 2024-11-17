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


function initialize_model(; number_of_atoms=100, speed = 0.03, extent = Tuple(ones(number_of_dimensions)), number_of_clumps=25, seed = 42)
    space = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)
    if number_of_dimensions == 2 
        H_atom = H_atom_2d
    elseif number_of_dimensions == 3 
        H_atom = H_atom_3d
    end

    model = StandardABM(H_atom, space; rng, agent_step!, scheduler = Schedulers.Randomly())
    n=round(Int, number_of_clumps^(1/number_of_dimensions))
    cell_size = extent ./ (2n+1)
    for k in 1:2n+1
        for j in 1:2n+1
            id = n * (k - 1) + j
            if k % 2 == 0 && j % 2 == 0
                for i in 1:number_of_atoms
                vel = randn(rng, Float64, (number_of_dimensions, 1)) .* speed
                pos = (rand(rng, Float64, (number_of_dimensions, 1)) .+ (k-1, j-1)) .* cell_size
                add_agent!(
                    pos,
                    model,
                    vel,
                    speed,
                    id                 
                )
                end
            end
        end
    end
    return model
end


function agent_step!(H_atom, model)
    H_atom.vel = H_atom.vel
    move_agent!(H_atom, model, H_atom.speed)
    
end



### interactive mode ####
model = initialize_model()
fig, ax, abmobs = abmplot(model; add_controls=true, agent_size=2)
fig

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

