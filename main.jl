using PyPlot
pygui(true)

## for future
# struct coordinate_system
#    x1::Float64 
#    x2::Float64 
#    x3::Float64 
# end


#### physical constants ####
k = 1.380649e-16            # erg * K^-1
m_H = 1.67375e-24           # g
###################


#### physical parameters ####
n_H = 1      # cm^-3,   physical number density of H atoms  
T = 10000    # K,       temperature
#############################

#### numerical parameters ####
N = 10                      # number of H atoms along an axis
time_step_constant = 0.01   # constant of numerical precision
##############################

#### derivative parameters ####
physical_length = n_H ^ (-1/3) * N                       # cm,        physical size of the box with particles
v_p = (2 * k  * T / m) ^ 0.5                             # cm / s,    the most probable velocity
unit_time = time_step_constant * physical_length / v_p   # s,         unit of time
###############################


function simulate_coordinates(number_of_points)
    points_arr = rand(number_of_points, 3)
    return points_arr
end

function simulate_velocities(distribution)
    
    return points_arr
end

function Maxwell_distribution()
    
end


points_arr = simulate_coordinates(N ^ 3)

x, y, z = points_arr[:, 1], points_arr[:, 2], points_arr[:, 3]

fig = PyPlot.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x, y, z, s=1)


