using Agents, Random, LinearAlgebra, CairoMakie


@agent struct H_atom(ContinuousAgent{2, Float64}) 
    speed::Float64
end

function initialize_model(; number_of_atoms = 1000, speed = 0.1, extent = (10, 10), seed = 42)
    space2d = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)

    model = StandardABM(H_atom, space2d; rng, agent_step!, scheduler = Schedulers.Randomly())
    for _ in 1:number_of_atoms
        vel = rand(abmrng(model), SVector{2}) * 2 .- 1
        add_agent!(
            model,
            vel,
            speed,
        )
    end
    return model
end

function agent_step!(H_atom, model)
    move_agent!(H_atom, model, H_atom.speed)
end

model = initialize_model()

const bird_polygon = Makie.Polygon(Point2f[0.1 .* (-1, -1), 0.1 .* (2, 0), 0.1 .* (-1, 1)])
function bird_marker(b::H_atom)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    rotate_polygon(bird_polygon, φ)
end

figure, = abmplot(model; agent_marker=bird_marker)
figure

abmvideo(
    "gas.mp4", model;
    # agent_marker = bird_marker,
    framerate = 20, frames = 300,
    title = "gas"
)
