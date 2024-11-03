using DynamicalSystems, CairoMakie
using LinearAlgebra: norm, dot

# Dynamical system and initial conditions
ds = Systems.thomas_cyclical(b = 0.2)
u0s = [[3, 1, 1.], [1, 3, 1.], [1, 1, 3.]] # must be a vector of states!

# Observables we get timeseries of:
function distance_from_symmetry(u)
    v = SVector{3}(1/√3, 1/√3, 1/√3)
    t = dot(v, u)
    return norm(u - t*v)
end
fs = [3, distance_from_symmetry]

fig, dsobs = interactive_trajectory_timeseries(ds, fs, u0s;
    idxs = [1, 2], Δt = 0.05, tail = 500,
    lims = ((-2, 4), (-2, 4)),
    timeseries_ylims = [(-2, 4), (0, 5)],
    add_controls = false,
    figure = (size = (800, 400),)
)

fig