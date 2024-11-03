using HardSphereDynamics, StaticArrays

# create box:

table = HardSphereDynamics.RectangularBox(SA[-0.5, -0.5, -1.0],
                                          SA[+0.5, +0.5, +3.0])

# create fluid:
d = 3     # spatial dimension
n = 20   # number of spheres
r = 0.1  # radius

fluid = HardSphereFluid{d,Float64}(table, n, r)
initial_condition!(fluid, lower=table.lower, upper=-table.lower)

# set up simulation:
collision_type = ElasticCollision()
flow_type = ExternalFieldFlow(SA[0.0, 0.0, -10.0])
event_handler = AllToAll(fluid, flow_type)

simulation =  HardSphereSimulation(
    fluid, event_handler, flow_type, collision_type);

# time evolution:
δt = 0.01
final_time = 100
states, times = evolve!(simulation, δt, final_time);

# visualization:
using Makie

HardSphereDynamics.visualize_3d(states, sleep_step=0.005, lower=table.lower, upper=-table.lower)
