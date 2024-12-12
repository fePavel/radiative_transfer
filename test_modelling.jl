using Agents
using Random, LinearAlgebra, Statistics
# using CairoMakie
using GLMakie


data = randn(1000)
data2 = Observable(data)

fig = Figure(; size = (600, 400))
ax = Axis(fig[1, 1]; xlabel="value")
# hist!(ax, data, normalization=:pdf; color=(:green, 0.5))
density!(ax, data2; color=(:red, 0.5))
# fig
record(fig, "animScatters.mp4",
    framerate=6, profile="main") do io
    for i in 1:100
        # msize[] = i * initms
        data2[] =  randn(1000)
        recordframe!(io)  # record a new frame
    end
end
