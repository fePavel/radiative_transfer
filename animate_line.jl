using GLMakie
using LinearAlgebra
fig = Figure(size=(1200, 600))

ax = Axis(fig[1, 1])
x = 0:0.001:1
f(x) = x^3 - 3x + 0.3*sin(30*x)
g(x) = 3x^4+x^2-x -0.5
y1 = f.(x)
y2 = g.(x)

τ = Observable(0.0)
y = @lift(y1 .* $τ .+ y2 .* (1 - $τ))
plot!(ax, x, y)
plot!(ax, x, y1)
plot!(ax, x, y2)

fig


record(fig, "time_animation.mp4", 0:0.01:1) do t
    τ[] = t
end
