using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie


# data = randn(1000)
# data2 = Observable(data)

# fig = Figure(; size = (600, 400))
# ax = Axis(fig[1, 1]; xlabel="value")
# hist!(ax, data, normalization=:pdf; color=(:green, 0.2))
# density!(ax, data2; color=(:red, 0.2))
# fig

# record(fig, "animScatters.mp4",
#     framerate=6, profile="main") do io
#     for i in 1:100
#         # msize[] = i * initms
#         data2[] =  randn(1000)
#         recordframe!(io)  # record a new frame
#     end
# end

α = 0.0
C = [cos(α) -sin(α); 
     sin(α) cos(α)]


# β = v / c
β = 0.01
γ = (1 - β^2)^-0.5
boost_x = [γ -β*γ;
            -β*γ γ]


c = 1
m = 0.2
v = [0.01; 0.0]
β = norm(v) / c
γ = (1 - β^2)^-0.5
vec_1 = [m * c^2 * γ; m * v[1] * γ; m * v[2] * γ]
p = [0.8; 0.1]
vec_2 = [norm(p); p[1]; p[2]]
# vec_1' * [1 0 0; 0 -1 0; 0 0 -1] * vec_1
# vec_2' * [1 0 0; 0 -1 0; 0 0 -1] * vec_2
Λ = [γ -γ*v[1]/c -γ*v[2]/c;
    -γ*v[1]/c 1+γ^2/(1+γ)*v[1]^2/c^2 γ^2/(1+γ)*v[1]*v[2]/c^2;
    -γ*v[2]/c γ^2/(1+γ)*v[1]*v[2]/c^2 1+γ^2/(1+γ)*v[2]^2/c^2]
Λ * vec_1
Λ * vec_2

function p_change(p, v; m=1, c=1)
    β = norm(v) / c
    γ = (1 - β^2)^-0.5
    vec_1 = m * γ .* [c^2; v[1]; v[2]]
    vec_2 = [norm(p); p[1]; p[2]]
    Λ = [γ -γ*v[1]/c -γ*v[2]/c;
        -γ*v[1]/c 1+γ^2/(1+γ)*v[1]^2/c^2 γ^2/(1+γ)*v[1]*v[2]/c^2;
        -γ*v[2]/c γ^2/(1+γ)*v[1]*v[2]/c^2 1+γ^2/(1+γ)*v[2]^2/c^2]
    # p_new = (Λ * vec_2)[2:3]
    # v_new = Λ * vec_1
    return 
end