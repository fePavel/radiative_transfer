using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie
using PyPlot
pygui(true)

@agent struct Photon_2d(ContinuousAgent{2,Float64})
    # polarization::Float64 # just for fun with no physical sence
    momentum::Float64
end


function collision(γ1::Photon_2d, γ2::Photon_2d, model)
    # simple collision that agrees with energy and momentum conservation laws.
    p1 = γ1.vel ./ norm(γ1.vel) .* γ1.momentum
    p2 = γ2.vel ./ norm(γ2.vel) .* γ2.momentum
    ϵ = (norm(p1 + p2)) / (norm(p1) + norm(p2))
    a = 0.5 * (norm(p1) + norm(p2))

    α = 2π * rand()
    β = acos((p1+p2)[1] / norm(p1 + p2))
    if (p1+p2)[2] < 0
        β *= -1
    end
    # direction = [cos(α - β); sin(α - β)]

    γ1.momentum = a * (ϵ^2 - 1) / (ϵ * cos(α) - 1)
    γ2.momentum = norm(p1) + norm(p2) - γ1.momentum
    γ1.vel = [cos(α + β); sin(α + β)] .* norm(γ1.vel)

    γ2_vec_momentum = p1 + p2 - γ1.vel .* γ1.momentum ./ norm(γ1.vel)
    γ2.vel = γ2_vec_momentum ./ norm(γ2_vec_momentum) .* norm(γ2.vel)

    return γ1, γ2
end

function agent_step!(photon_agent::Photon_2d, model)
    move_agent!(photon_agent, model, 1)
end

space = ContinuousSpace(Tuple(ones(2)); periodic=true)
seed=43
rng = Random.MersenneTwister(seed)
model = StandardABM(Photon_2d, space; rng, agent_step!, scheduler=Schedulers.ByID())

### adding photons ###
first_photon_index = 1
base_photon = Photon_2d(first_photon_index, (0.5, 0.5), (0.1, 0.1), 1)
number_of_photons=10
for _ in 1:number_of_photons
    pos = (rand(rng, Float64, (2, 1)) .- (0.5, 0.5)) .* 0.333 .+ (0.5, 0.5)
    α = 2π * rand(rng)
    direction = [cos(α); sin(α)]
    vel = 0.5 .* direction
    momentum = rand(rng)
    replicate!(base_photon, model; pos=pos, vel=vel, momentum=momentum)
end


x = [model[i].vel[1] / norm(model[i].vel) * model[i].momentum for i in 1:nagents(model)]
y = [model[i].vel[2] / norm(model[i].vel) * model[i].momentum for i in 1:nagents(model)]

fig, ax = PyPlot.subplots(figsize=[6, 6])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

n = zeros(length(x))
# for i in 1:length(x)
for i in 1:2
    ax.arrow(0,0,x[i], y[i])
end

total_momentum(m1, m2) = [m1.vel[1]; m1.vel[2]] .* m1.momentum ./ norm(m1.vel) + [m2.vel[1]; m2.vel[2]] .* m2.momentum ./ norm(m2.vel)
t = total_momentum(model[1], model[2])
ax.arrow(0, 0, t[1], t[2], color="red")

collision(model[1], model[2], model)

x = [model[i].vel[1] / norm(model[i].vel) * model[i].momentum for i in 1:nagents(model)]
y = [model[i].vel[2] / norm(model[i].vel) * model[i].momentum for i in 1:nagents(model)]
for i in 1:2
    ax.arrow(0, 0, x[i], y[i], color="red")
end
t = total_momentum(model[1], model[2])
ax.arrow(0, 0, t[1], t[2], color="green")

for i in 1:10
    println("momentum: ", total_momentum(model[1], model[2]))
    println("norm momentum: ", norm(total_momentum(model[1], model[2])))
    println("1 momentum: ", model[1].momentum)

    collision(model[1], model[2], model)
    t = total_momentum(model[1], model[2])
    ax.arrow(0, 0, t[1], t[2], color="green")

    # println("energy: ", model[1].momentum + model[2].momentum)
end




# γ1, γ2 = model[1], model[2]

# # simple collision that agrees with energy and momentum conservation laws.
# p1 = γ1.vel ./ norm(γ1.vel) .* γ1.momentum
# p2 = γ2.vel ./ norm(γ2.vel) .* γ2.momentum
# ax.arrow(0, 0, p1[1], p1[2], color="green")
# ax.arrow(0, 0, p2[1], p2[2], color="green")
# ϵ = (norm(p1 + p2)) / (norm(p1) + norm(p2))
# a = 0.5 * (norm(p1) + norm(p2))

# α = 2π * rand()
# β = acos((p1+p2)[1] / norm(p1 + p2))
# if (p1+p2)[2] < 0
#     β *= -1
# end
# # direction = [cos(α - β); sin(α - β)]

# γ1.momentum = a * (ϵ^2 - 1) / (ϵ * cos(α) - 1)
# γ2.momentum = norm(p1) + norm(p2) - γ1.momentum

# println("energy check: ", γ1.momentum + γ2.momentum - 2a)
# γ1.vel = [cos(α + β); sin(α + β)] .* norm(γ1.vel)

# p1_new = γ1.vel ./ norm(γ1.vel) .* γ1.momentum
# ax.arrow(0, 0, p1_new[1], p1_new[2], color="blue")


# δ = -asin(γ1.momentum * sin(α) / (2a - γ1.momentum))
# # if α < π
# #     δ *= -1
# # end
# if γ1.momentum >= a * (1 - ϵ^2)
#     δ = -π / 2 - δ
# end
# γ2.vel = [cos(δ + β); sin(δ + β)] .* norm(γ2.vel)

# p2_new = γ2.vel ./ norm(γ2.vel) .* γ2.momentum
# ax.arrow(0, 0, p2_new[1], p2_new[2], color="blue")







# x = [model[i].vel[1] for i in 1:10][[4, 9]]
# y = [model[i].vel[2] for i in 1:10][[4, 9]]

# PyPlot.scatter(x, y)
# PyPlot.xlim([-0.02, 0.02])
# PyPlot.ylim([-0.02, 0.02])
# println("momentum: ", total_momentum(model[4], model[9]))
# println("energy: ", model[4].momentum + model[9].momentum)


