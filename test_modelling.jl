using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie

number_of_dimensions = 2

@agent struct H_atom_3d(ContinuousAgent{3, Float64}) 
    speed::Float64
end

@agent struct H_atom_2d(ContinuousAgent{2,Float64})
    speed::Float64
end


function initialize_model(; number_of_atoms = 1000, speed = 0.3, extent = Tuple(ones(number_of_dimensions)), seed = 42)
    space = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)
    if number_of_dimensions == 2 
        H_atom = H_atom_2d
    elseif number_of_dimensions == 3 
        H_atom = H_atom_3d
    end

    model = StandardABM(H_atom, space; rng, agent_step!, scheduler = Schedulers.Randomly())
    for i in 1:number_of_atoms
        # vel = rand(abmrng(model), SVector{number_of_dimensions}) * 2 .- 1
        vel = randn(rng, Float64, (number_of_dimensions, 1)) .* speed
        add_agent!(
            model,
            vel,
            speed,
            # turbulence_group=i % 100
        )
    end
    return model
end

function agent_step!(H_atom, model)
    H_atom.vel = H_atom.vel * 0.99
    move_agent!(H_atom, model, H_atom.speed)
end

# N_atoms = 10000
# model = initialize_model(number_of_atoms=N_atoms)

# figure,  = abmplot(model)
# # figure



# kwargs = Dict(:agent_size => 10)

# figure, = abmplot(model; agent_size=10); figure
# model

# adata = [:vel, :pos]
# model = initialize_model()
# adf, = run!(model, 200; adata)
# vel_abs = [[(i[1]^2+i[2]^2)^0.5 for i in adf[j:1000:end, 3]] for j in 1:10]
# plot(1:201, vel_abs[1])
# for i in 2:10
#     plot!(1:201, vel_abs[i])
# end
# current_figure()


function square(x)
    s = 0
    for i in 1:length(x)
        s += x[i]^2
    end
    return s
end

function plot_velocity_distribution(number_of_bins)
    N_atoms = 100000
    model = initialize_model(number_of_atoms=N_atoms)

    # max_time = 200
    # T = zeros(max_time)
    # for i in 1:max_time
    #     T[i] = mean([square(model[j].vel) for j in 1:N_atoms])
    #     step!(model)
    # end

    hist([square(model[j].vel) .^ 0.5 for j in 1:N_atoms], bins=number_of_bins, color=:red)
end


plot_velocity_distribution(500)

model=initialize_model()
abmvideo(
    "gas.mp4", model;
    framerate = 20, frames = 60,
    title="gas", agent_size=6
)



#### interactive mode ####
# model=initialize_model()
# fig, ax, abmobs = abmplot(model; add_controls=true, agent_size=10)
# fig
# 



#### new feature ####
# abmobs
# plot_layout = fig[:, end+1] = GridLayout()
# count_layout = plot_layout[1, 1] = GridLayout()
# Temperature = @lift(Point2f.($(abmobs.adf).time, $(abmobs.adf).count_black))



extent = (1, 1, 1)
sp = ContinuousSpace(extent; periodic=true)
