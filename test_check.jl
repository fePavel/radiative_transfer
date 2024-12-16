α = 0:0.001:2π

p = 0.1
mc = 1
r1(α) = p * cos(α) - mc  - (mc^2 - p^2*sin(α)^2 + 2mc*p *(1-cos(α))) ^0.5
r2(α) = p * cos(α) - mc + (mc^2 - p^2 * sin(α)^2 + 2mc * p * (1 - cos(α)))^0.5

x(α) = r1(α) * cos(α)
y(α) = r1(α) * sin(α)
x_2(α) = r2(α) * cos(α)
y_2(α) = r2(α) * sin(α)

x_arr = x.(α)
y_arr = y.(α)

x_arr_2 = x_2.(α)
y_arr_2 = y_2.(α)


using GLMakie
using LinearAlgebra
fig = Figure(size=(1200, 600))

ax = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

# sl_x = Slider(fig[2, 1], range=0:0.01:10, startvalue=3)
# sl_y = Slider(fig[1, 2], range=0:0.01:10, horizontal=false, startvalue=6)

slider_α = Slider(fig[2, 1], range=-π:0.01:π, startvalue=0)
# slider_r = Slider(fig[3, 1], range=0:0.01:10, startvalue=1)

point1 = lift(slider_α.value) do α
    r1 = p * cos(α) - mc * (1 - ((1 - p / mc * cos(α))^2 + 2 * p / mc - (p / mc)^2)^0.5)
    Point2f(r1 * cos(α), r1 * sin(α))
    # Point2f(0.1 - r1 * cos(α), r1 * sin(α))
end

point12 = lift(slider_α.value) do α
    r = r1(α)
    Point2f(r * cos(α), r * sin(α))
end


point2 = lift(slider_α.value) do α
    r1 = p * cos(α) - mc * (1 - ((1 - p / mc * cos(α))^2 + 2 * p / mc - (p / mc)^2)^0.5)
    # Point2f(r1 * cos(α), r1 * sin(α))
    Point2f(0.1 - r1 * cos(α), -r1 * sin(α))
end

point3 = lift(slider_α.value) do α
    r1 = p * cos(α) - mc * (1 - ((1 - p / mc * cos(α))^2 + 2 * p / mc - (p / mc)^2)^0.5)
    # Point2f(r1 * cos(α), r1 * sin(α))
    Point2f(α, r1 + 0.5 * ((0.1 - r1 * cos(α))^2 + r1^2*sin(α)^2))
end

# point4 = lift(slider_α.value) do α
#     r1 = p * cos(α) - mc * (1 - ((1 - p / mc * cos(α))^2 + 2 * p / mc - (p / mc)^2)^0.5)
#     # Point2f(r1 * cos(α), r1 * sin(α))
#     Point2f(α, r1)
# end


scatter!(ax, point1, color=:red, markersize=5)
scatter!(ax, point12, color=:red, markersize=5)
scatter!(ax, point2, color=:green, markersize=5)
scatter!(ax2, point3, color=:blue, markersize=5)

limits!(ax, -1, 1, -1, 1)
limits!(ax2, -2π, 2π, 0.05, 0.15)


fig
