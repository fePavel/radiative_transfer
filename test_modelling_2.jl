using Agents
using Random, LinearAlgebra, Statistics
# using Distributions
# using CairoMakie
using GLMakie

number_of_dimensions = 2
const h = 6.62607015e-34

@agent struct H_atom_3d(ContinuousAgent{3, Float64}) 
    speed::Float64
    clump_id::Int
end

@agent struct H_atom_2d(ContinuousAgent{2,Float64})
    speed::Float64
    clump_id::Int
end
	
@agent struct Clump(ContinuousAgent{2,Float64})
    cell_size::Float64
    child_agents::Array{Union{H_atom_2d, H_atom_3d},1}
end

@agent struct Photon_2d(ContinuousAgent{2,Float64})
    # polarization::Float64 # just for fun with no physical sence
    momentum::Float64
end

@agent struct Photon_3d(ContinuousAgent{3,Float64})
    polarization::Float64 # just for fun with no physical sence
end


function initialize_model(; number_of_atoms=100, speed = 0.03, extent = Tuple(ones(number_of_dimensions)), number_of_clumps=16, seed = 42, number_of_photons=0, mass=1, c=1)
    space = ContinuousSpace(extent; periodic=true)  
    rng = Random.MersenneTwister(seed)
    if number_of_dimensions == 2 
        H_atom = H_atom_2d
        Photon = Photon_2d
    elseif number_of_dimensions == 3 
        H_atom = H_atom_3d
        Photon = Photon_3d
    end

    λ_Ly_α = 1215.67

    properties = Dict(:n => round(Int, number_of_clumps^(1 / number_of_dimensions)),
                      :number_of_clumps => number_of_clumps,
                      :full_number_of_atoms => number_of_atoms * number_of_clumps,
                      :number_of_photons => number_of_photons,
                    #   :mass => mass,
                      :mass => 0.1 * h / λ_Ly_α / speed,
                      :speed_of_light => c,
                      :speed => speed,
                      :atom_indexes => number_of_clumps+1:number_of_clumps+number_of_atoms*number_of_clumps,
                      :clump_indexes => 1:number_of_clumps,
                      :photon_indexes => number_of_clumps+number_of_atoms*number_of_clumps+1:number_of_clumps+number_of_atoms*number_of_clumps + number_of_photons
    )

    model = StandardABM(Union{H_atom, Clump, Photon}, space; rng, agent_step!, model_step!, scheduler=Schedulers.ByID(), properties = properties)
    # model = StandardABM(Union{H_atom,Clump,Photon}, space; rng, model_step!, scheduler=Schedulers.ByID(), properties=properties)
    n = round(Int, number_of_clumps^(1 / number_of_dimensions))
    
    clump_space = 2 # 2 --> no empty space
    
    cell_size = extent ./ (2n+1) 
    base_clump = Clump(1, (0.5, 0.5), (0.0, 0.0), clump_space*cell_size[1], Int[])
    for i in 1:2n+1
        for j in 1:2n+1
            if i % 2 ==0 && j % 2 == 0 
                clump_pos = [(i - 0.5), (j - 0.5)] .* cell_size
                vel = 1 .* randn(rng, Float64, (number_of_dimensions, 1)) .* 0.001
                # vel = [0, 0]
                replicate!(base_clump, model; pos=clump_pos, vel=vel, child_agents=Int[])
            end
        end
    end


    ### adding atoms ###
    base_H_atom = H_atom(number_of_clumps+1, (0.5, 0.5), (0.0, 0.0), speed, 1)
    for i in 1:number_of_clumps
        for j in 1:number_of_atoms
            pos = (rand(rng, Float64, (number_of_dimensions, 1)) .- (0.5, 0.5)) .* cell_size .* clump_space .+ model[i].pos 
            # vel = randn(rng, Float64, (number_of_dimensions, 1)) .* speed 
            vel = randexp(rng, Float64, (number_of_dimensions, 1)) .* speed
            H_atom = replicate!(base_H_atom, model; pos=pos, vel=vel, clump_id=i)
            push!(model[i].child_agents, H_atom)
        end
    end

    ### correct velocities for desired average ###
    for i in 1:number_of_clumps
        mean_vel = mean([atom.vel for atom in model[i].child_agents])
        for atom in model[i].child_agents
            atom.vel += -mean_vel + model[i].vel
        end
    end

    ### adding photons ###
    first_photon_index = number_of_clumps * (1 + number_of_atoms) + 1
    base_photon = Photon(first_photon_index, (0.5, 0.5), (0.1, 0.1), 1)
    for _ in 1:number_of_photons
        pos = (rand(rng, Float64, (number_of_dimensions, 1)) .- (0.5, 0.5)) .* cell_size .+ (0.5, 0.5)
        α = 2π * rand()
        direction = [cos(α); sin(α)]
        vel = 2speed .* direction 
        momentum =  (1 + 0.01 * randn()) # * h / λ_Ly_α
        replicate!(base_photon, model; pos=pos, vel=vel, momentum=momentum)
    end

    return model
end

function dist(agent_1, agent_2, model)
    # spacesize(model)[1]
    dist_1 = min(abs(agent_1.pos[1] - agent_2.pos[1]),
        abs(spacesize(model)[1] + agent_1.pos[1] - agent_2.pos[1]),
        abs(-spacesize(model)[1] + agent_1.pos[1] - agent_2.pos[1]))
    dist_2 = min(abs(agent_1.pos[2] - agent_2.pos[2]),
        abs(spacesize(model)[2] + agent_1.pos[2] - agent_2.pos[2]),
        abs(-spacesize(model)[2] + agent_1.pos[2] - agent_2.pos[2]))
    return dist_1, dist_2
end

function agent_step!(test_agent::Union{H_atom_2d,H_atom_3d}, model)
    cell_size = model[test_agent.clump_id].cell_size
    prev_dist = dist(model[test_agent.clump_id], test_agent, model)
    if prev_dist[1] > 0.5 * cell_size || prev_dist[2] > 0.5 * cell_size
        return 0
    end

    move_agent!(test_agent, model, test_agent.speed)
    next_dist = dist(model[test_agent.clump_id], test_agent, model)

    if next_dist[1] > 0.5 * cell_size && test_agent.vel[1] > 0
        walk!(test_agent, SVector((-cell_size, 0)), model)
    end
    if next_dist[1] > 0.5 * cell_size && test_agent.vel[1] < 0
        walk!(test_agent, SVector((cell_size, 0)), model)
    end
    if next_dist[2] > 0.5 * cell_size && test_agent.vel[2] > 0
        walk!(test_agent, SVector((0, -cell_size)), model)
    end
    if next_dist[2] > 0.5 * cell_size && test_agent.vel[2] < 0
        walk!(test_agent, SVector((0, cell_size)), model)
    end
    # test_agent.vel *=0.999
end

function agent_step!(clump_agent::Clump, model)
    clump_agent.vel = SVector(mean([atom.vel[1] for atom in clump_agent.child_agents]), mean([atom.vel[2] for atom in clump_agent.child_agents]))
    

    cell_size = clump_agent.cell_size
    move_agent!(clump_agent, model, 1)

    for child_agent in clump_agent.child_agents
        next_dist = dist(clump_agent, child_agent, model)
        if next_dist[1] > 0.5 * cell_size && clump_agent.vel[1] > 0
            walk!(child_agent, SVector((cell_size, 0)), model)
        end
        if next_dist[1] > 0.5 * cell_size && clump_agent.vel[1] < 0
            walk!(child_agent, SVector((-cell_size, 0)), model)
        end
        if next_dist[2] > 0.5 * cell_size && clump_agent.vel[2] > 0
            walk!(child_agent, SVector((0, cell_size)), model)
        end
        if next_dist[2] > 0.5 * cell_size && clump_agent.vel[2] < 0
            walk!(child_agent, SVector((0, -cell_size)), model)
        end
    end
    # clump_agent.vel *= 0.999
end

function agent_step!(photon_agent::Photon_2d, model)
    move_agent!(photon_agent, model, 3*model.speed)
end

function model_step!(model; number_of_collisions=10, number_of_photon_collisions=10)
    # if abmtime(model) % 2 == 0
    #     number_of_atoms_per_clump = length(model[1].child_agents)
    #     i, j = rand(2:number_of_atoms_per_clump, 2)
    #     new_vel_1, new_vel_2 = collision(model[i].vel, model[j].vel)
    #     model[i].vel = new_vel_1
    #     model[j].vel = new_vel_2
    # end
    if typeof(model[1]) == "Clump"
        number_of_atoms_per_clump = length(model[1].child_agents)
    else
        number_of_atoms_per_clump = 0
    end

    number_of_collisions = max(round(Int, (number_of_atoms_per_clump / 100)), 1)

    # if number_of_collisions < 1
    #     number_of_steps_per_collision = round(Int, 1/number_of_collisions)
    # end
    for k in 1:model.number_of_clumps
        # first_atom_index = model.atom_indexes[1]
        # last_atom_index = model.atom_indexes[end]
        first_atom_index = model.number_of_clumps + (k-1)*number_of_atoms_per_clump + 1 
        last_atom_index = model.number_of_clumps + k * number_of_atoms_per_clump
        for _ in 1:number_of_collisions
            # i, j = rand(first_atom_index:last_atom_index, 2)
            atom_1, atom_2 = rand(model[k].child_agents, 2)

            # collision(model[i], model[j])
            collision(atom_1, atom_2)

        end
    end

    
    # only for photons
    if model.number_of_photons > 0
        for _ in 1:number_of_photon_collisions
#            i, j = rand(model.photon_indexes, 1), rand(model.atom_indexes, 1) 
            i, j = rand(model.photon_indexes, 1)[1], rand(model.photon_indexes, 1)[1] 
            # println(i, j)
            collision(model[i], model[j], model)
        end
    end
end

function collision(H1::H_atom_2d, H2::H_atom_2d)
    # simple collision which agree with energy and momentum conservation laws.
    velocity_1, velocity_2 = H1.vel, H2.vel

    center = 0.5 * (velocity_1 + velocity_2)
    radius = 0.5 * norm(velocity_1 - velocity_2)
    α = 2π * rand()
    direction = [cos(α); sin(α)]
    new_vel_1 = center + radius * direction
    new_vel_2 = center - radius * direction
    H1.vel, H2.vel = new_vel_1, new_vel_2
    return 0
end





function generate_lyman_alpha_photon(v_source; Γ=6.265e8, λ0=121.567e-9)
    c = 299792458.0
    c = 0.03 * 100

    # 1. Генерация длины волны в системе источника (Лоренц)
    ω0 = 2π * c / λ0
    ω = ω0 + Γ / 2 * tan(π * (rand() - 0.5))
    λ_emit = 2π * c / ω

    # 2. Случайное направление излучения в 2D (в системе источника)
    ϕ = 2π * rand()
    n_emit = [cos(ϕ), sin(ϕ)]

    # 3. Переход в лабораторную систему отсчёта (учёт скорости v_source)
    v = norm(v_source)
    # if v ≈ 0.0
    #     return (λ=λ_emit, direction=n_emit)
    # end

    β = v / c
    γ = 1 / sqrt(1 - β^2)
    v_hat = v_source / v

    # Релятивистский эффект Доплера
    doppler_factor = γ * (1 + dot(v_hat, n_emit) * β)
    λ_lab = λ_emit * doppler_factor
    println(doppler_factor, β)

    # Аберрация света (преобразование направления)
    n_lab = (n_emit + (γ - 1) * dot(v_hat, n_emit) * v_hat + γ * β * v_hat) / (γ * (1 + dot(v_hat, n_emit) * β))
    n_lab = normalize(n_lab)  # Нормировка на случай численных погрешностей

    return λ_lab, n_lab
end

# # Пример
# v_source = [0.1c, 0.2c] 
# photon = generate_lyman_alpha_photon(v_source)
# println("λ (lab): ", photon.λ, " м")
# println("Направление (lab): ", photon.direction)


function collision(γ::Photon_2d, H::H_atom_2d, model)
    # simple collision which agree with energy and momentum conservation laws.
    m = model.mass
    # c = model.speed_of_light

    γ_direction = γ.vel ./ norm(γ.vel)
    γ_p = γ.momentum .* γ_direction
    vel = γ_p ./ m .+ H.vel

    new_λ, new_direction = generate_lyman_alpha_photon(vel)
    # h = 6.62607015e-34
    new_momentum = h / new_λ
    γ.momentum = new_momentum
    γ.vel = model.speed_of_light .* new_direction

    H.vel = vel .- new_momentum .* new_direction ./ m    
    return γ, H
end

function collision(γ1::Photon_2d, γ2::Photon_2d, model)
    # simple collision that agrees with energy and momentum conservation laws.
    p1 = γ1.vel ./ norm(γ1.vel) .* γ1.momentum
    p2 = γ2.vel ./ norm(γ2.vel) .* γ2.momentum
    ϵ = (norm(p1 + p2)) / (norm(p1) + norm(p2))
    a = 0.5 * (norm(p1) + norm(p2))
    b = a*(1 - ϵ^2)^0.5 

    α = 2π * rand()
    β = acos((p1+p2)[1] / norm(p1 + p2))
    if (p1+p2)[2] < 0
        β *= -1
    end
    # direction = [cos(α - β); sin(α - β)]

    # γ1.momentum = a * (ϵ^2 - 1) / (ϵ * cos(α) - 1)
    p1_new = b / (1 - ϵ^2 * cos(α)^2)^0.5 .* [cos(α); sin(α)] + 0.5 .* (p1 + p2) 
    γ1.momentum = norm(p1_new)
    γ2.momentum = 2a - γ1.momentum
    γ1.vel = p1_new ./ norm(p1_new) .* norm(γ1.vel)

    γ2_vec_momentum = p1 + p2 - p1_new
    γ2.vel = γ2_vec_momentum ./ norm(γ2_vec_momentum) .* norm(γ2.vel)

    return γ1, γ2
end


function agent_color(agent::Union{H_atom_2d, H_atom_3d})
    return :blue
end
function agent_color(agent::Clump)
    return :red
end
function agent_color(agent::Photon_2d)
    return :green
end


# ### interactive mode ####
# model = initialize_model()
# 
# plotkwargs = (;
#     agent_color=agent_color, agent_size=2, 
# )
# params = Dict(
#     :speed => 0.02:0.001:0.04,
# )
# 
# fig, ax, abmobs = abmplot(model; add_controls=true, params, plotkwargs..., adata)
# fig



# ### explore data ###
# plotkwargs = (;
#     agent_color=agent_color, agent_size=2, 
# )
# params = Dict(
#     :speed => 0.02:0.001:0.04,
# )
# kin_temp(H_atom::Union{H_atom_2d,H_atom_3d}) = (H_atom.vel[1]^2 + H_atom.vel[2]^2)^0.5
# kin_temp(H_atom::Clump) = 0
# model = initialize_model()
# adata = [(kin_temp, mean)]

# number_of_clumps = 25
# std_velocities_norms(model) = std([norm(model[i].vel) for i in number_of_clumps+1:nagents(model)])
# mdata = [std_velocities_norms]
# # adf, mdf = run!(model, 1000; adata)
# fig, abmobs = abmexploration(model; add_controls=true, params, plotkwargs..., adata, mdata)
# fig


# ### advanced data exploration ###
# plotkwargs = (;
#     agent_color=agent_color, agent_size=1,
# )
# params = Dict(
#     :speed => 0.02:0.001:0.04,
# )
# kin_temp(H_atom::Union{H_atom_2d, H_atom_3d}) = (H_atom.vel[1]^2 + H_atom.vel[2]^2)^0.5
# kin_temp(H_atom::Clump) = 0
# kin_temp(H_atom::Union{Photon_2d, Photon_3d}) = 0
# model = initialize_model(number_of_atoms=10000, number_of_photons=10000)
# adata = [(kin_temp, mean)]
# number_of_clumps=1
# atoms_momentum_x(model) = [model[i].vel[1] for i in number_of_clumps+1:number_of_clumps+model.full_number_of_atoms]
# atoms_momentum(model) = [norm(model[i].vel) for i in number_of_clumps+1:number_of_clumps+model.full_number_of_atoms]
# photons_momentum(model) = [model[i].vel[1] for i in number_of_clumps+model.full_number_of_atoms+1:nagents(model)]

# mdata = [atoms_momentum_x, photons_momentum]
# fig, ax, abmobs = abmplot(model; params, plotkwargs..., adata, mdata, figure = (; size = (1200,600)))
# plot_layout = fig[:,end+1] = GridLayout()
# count_layout = plot_layout[1, 1] = GridLayout()

# ax_counts = Axis(count_layout[1, 1]; backgroundcolor=:lightgrey, ylabel="Number of daisies by color")
# temperature = @lift(Point2f.($(abmobs.adf).time, $(abmobs.adf).mean_kin_temp))
# scatterlines!(ax_counts, temperature; color=:black, label="black")
# ax_hist = Axis(plot_layout[2, 1]; ylabel="super")
# # hist!(ax_hist, @lift($(abmobs.mdf)[end, 2]); bins=50, normalization=:pdf, color=(:blue, 0.5))
# density!(ax_hist, @lift($(abmobs.mdf)[end, 2]), color=(:blue, 0.5), strokewidth=1)
# hist!(ax_hist, @lift($(abmobs.mdf)[end, 3]); bins=50, normalization=:pdf, color=(:green, 0.5))
# # xmin = mean(abmobs.mdf[][1, 2]) - 5*mean(abmobs.mdf[][1, 2])
# # xmax = mean(abmobs.mdf[][1, 2]) + 5 * mean(abmobs.mdf[][1, 2])
# # xlims!(ax_hist, (xmin, xmax))
# # ylims!(ax_hist, (0.0, 1.e6))

function plot_density_for_all_clumps(model)
    plotkwargs = (;
        agent_color=agent_color, agent_size=1,
    )
    params = Dict(
        :speed => 0.02:0.001:0.04,
    )
    kin_temp(H_atom::Union{H_atom_2d,H_atom_3d}) = (H_atom.vel[1]^2 + H_atom.vel[2]^2)^0.5
    kin_temp(H_atom::Clump) = 0
    kin_temp(H_atom::Union{Photon_2d,Photon_3d}) = 0
    adata = [(kin_temp, mean)]
    number_of_clumps = model.number_of_clumps
    if number_of_clumps != 0
        number_of_atoms_per_clump = Int(model.full_number_of_atoms / model.number_of_clumps)
    else
        number_of_atoms_per_clump = 0
    end
    atoms_momentum_x(model) = [[model[i].vel[1] for i in model.number_of_clumps + (k-1)*number_of_atoms_per_clump + 1:model.number_of_clumps + k * number_of_atoms_per_clump] for k in 1:number_of_clumps]
    atoms_momentum_x_all(model) = [model[i].vel[1] for i in model.atom_indexes]
    atoms_momentum(model) = [norm(model[i].vel) for i in number_of_clumps+1:number_of_clumps+model.full_number_of_atoms]
    # photons_momentum(model) = [model[i].vel[1] for i in model.photon_indexes]
    photons_momentum(model) = [model[i].momentum for i in model.photon_indexes]

    mdata = [atoms_momentum_x, atoms_momentum_x_all, photons_momentum]

    model_obs = Observable(model)

    fig, ax, abmobs = abmplot(model_obs[]; params, plotkwargs..., adata, mdata, figure=(; size=(1200, 600)))
    plot_layout = fig[:, end+1] = GridLayout()
    count_layout = plot_layout[1, 1] = GridLayout()

    ### temperature evolution ###
    ax_counts = Axis(count_layout[1, 1]; backgroundcolor=:lightgrey, ylabel="temperature")
    temperature = @lift(Point2f.($(abmobs.adf).time, $(abmobs.adf).mean_kin_temp))
    scatterlines!(ax_counts, temperature; color=:black, label="black")
    autolimits!(ax_counts)
    # xlims!(ax_counts, 0, @lift($(abmobs.adf)[end, 2])[])
    # xlims!(ax_counts, 0, 100)
    #############################

    ### velocity kde for all clumps ###
    ax_hist = Axis(plot_layout[2, 1]; ylabel="velocity kde")
    for k in 1:number_of_clumps
        density!(ax_hist, @lift($(abmobs.mdf)[end, 2][k]), color=(:blue, min(0.5, 1 / number_of_clumps)), strokewidth=min(0.5, 5 / number_of_clumps))
    end
    ###################################

    ### velocity kde for all atoms ###
    if number_of_atoms_per_clump > 0
        ax_hist_2 = Axis(plot_layout[2, 2]; ylabel="velocity kde")
        density!(ax_hist_2, @lift($(abmobs.mdf)[end, 3]), color=(:blue, 0.5), strokewidth=2)
        linkyaxes!(ax_hist, ax_hist_2)
        linkxaxes!(ax_hist, ax_hist_2)
    end
    ##################################

    ### momentum kde for photons ###
    if model.number_of_photons > 0
        ax_hist_3 = Axis(plot_layout[1, 2]; ylabel="momentum kde")
        density!(ax_hist_3, @lift($(abmobs.mdf)[end, 4]), color=(:green, 0.5), strokewidth=2)
    end
    ################################

    on(abmobs.model) do m
        autolimits!(ax_counts)
        autolimits!(ax_hist_3)
    end


    return fig, ax, abmobs, model_obs
end

function plot_density_for_photons(model)
    plotkwargs = (;
        agent_color=agent_color, agent_size=10,
    )
    params = Dict(
        :speed => 0.02:0.001:0.04,
    )

    photons_momentum(model) = [model[i].momentum for i in 1:nagents(model)]

    adata = []
    mdata = [photons_momentum]
    fig, ax, abmobs = abmplot(model; params, plotkwargs..., adata, mdata, figure=(; size=(1200, 600)))
    plot_layout = fig[:, end+1] = GridLayout()
    count_layout = plot_layout[1, 1] = GridLayout()

    ax_1 = Axis(plot_layout[1, 1]; ylabel="not super")
    ax_hist = Axis(plot_layout[2, 1]; ylabel="super")
    # hist!(ax_hist, @lift($(abmobs.mdf)[end, 2]); bins=50, normalization=:pdf, color=(:blue, 0.5))
    
    density!(ax_hist, @lift($(abmobs.mdf)[end, 2]), color=(:green, 0.5), strokewidth=3)
    
    return fig, ax, abmobs
end

model = initialize_model(number_of_atoms=100, number_of_photons=10000, number_of_clumps=100)
# fig, ax, abmobs = plot_density_for_photons(model)
fig, ax, abmobs, model_obs = plot_density_for_all_clumps(model)


record(fig, "example.gif"; framerate=5) do io
    for j in 1:50
        recordframe!(io)
        if j < 50 /4
            Agents.step!(abmobs, 4)
        elseif j < 60 / 4
            Agents.step!(abmobs, 4*(j - 49 / 4))
        else
            Agents.step!(abmobs, 4*10)
        end
    
    end
    recordframe!(io)
end

# abmobs
# collision(model[1001], model[1002], model)
# step!(model)
# model[994]

# 1
# abmobs.adf[][end,2]

# 1
# using PyPlot
# pygui(true)
# abmobs.mdf[]
# PyPlot.hist(log10.((abmobs.mdf[][696, 2])), bins=100)


# 1
# agent_step!(model[1], model)

# step!(model)
# fig
# for i in 1:10; step!(abmobs, 1); end
# fig

# abmobs.mdf[][end, 2]

# @time for i in 1:1000
    # step!(model)
# end

# @allocated(abmobs.mdf[][:,:])s

# typeof(model[1])
# mean(c)
# std(c)
# abmobs.mdf[][:,2]


# 0.035 0.045

# # nagents(model)

# # step!(model)
# # length(model[1].child_agents)
# mean([model[j].vel[1] for j in 2:10001])
# mean([model[j].vel[2] for j in 2:10001])
# t = 2:10001
# z = [sum([model[j].vel[1] for j in 2:i]) for i in t]
# z2 = [mean([model[j].vel[1] for j in 2:i]) for i in t]

# x = [model[j].vel[1] for j in 2:10001]
# y = [model[j].vel[2] for j in 2:10001]
# using PyPlot
# pygui(true)
# PyPlot.scatter(x,y, color="red")
# PyPlot.plot(t,z./10000)
# # model[1].vel[1]
# # model[1].vel[2]

# z
# abmtime(model)
# step!(model)
# println(model[2])

# ### video ###
# model = initialize_model()
# abmvideo(
#     "gas_1.mp4", model;
#     framerate=20, frames=300,
#     title="gas", agent_size=3s
# )


# model[3].child_agents
# nagents(model)
# model = initialize_model()
# for i in 1:1000
#     agent_step!(model[1], model)
# end

# n = 4
# m = zeros(2n + 1, 2n + 1)
# for i in 1:2n+1
#     for j in 1:2n+1
#         if i % 2 == 0 && j % 2 == 0
#             m[i, j] = 1
#         end
#     end
# end

# m

# abmspace(model).

# #### check collision ####
# v1, v2 = [0; 1], [1; 0]
# v11, v21 = collision(v1, v2)
# v1 + v2
# v11 + v21
# norm(v1)^2 + norm(v2)^2
# norm(v11)^2 + norm(v21)^2
