using Oceananigans
using Oceananigans.Units
using Oceananigans.Forcings: MultipleForcings
using JLD2
using Printf
using CairoMakie
using Random
using Statistics
using LinearAlgebra
using Glob
import Dates

Random.seed!(123)

const Lz = 128
const Lx = 2048
const Ly = 2048

const Nz = 64
const Nx = 512
const Ny = 512

# const Nz = 32
# const Nx = 32
# const Ny = 32

const Jᵁ = 0
const Jᴮ = 4.24e-8

advection = WENO(order=9)
closure = nothing

const f = 1e-4

const N² = 9e-5
const M² = -4.24e-7

FILE_NAME = "MLI_test"
FILE_DIR = "./LES/$(FILE_NAME)"
mkpath(FILE_DIR)

size_halo = 5

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

grid = RectilinearGrid(CPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

b_initial(x, y, z) = N² * z
v_initial(x, y, z) = M² / f * (Lz + z)

b_initial_noisy(x, y, z) = b_initial(x, y, z) + 1e-11 * noise(x, y, z)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Jᴮ))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Jᵁ))

damping_rate = 5e-3

b_target(x, y, z, t) = b_initial(x, y, z)
v_target(x, y, z, t) = v_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)
v_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=v_target)
uw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)

u_forcing_func(x, y, z, t) = -M² * (Lz + z)
b_forcing_func(x, y, z, t, u) = -u * M²

u_forcing = Forcing(u_forcing_func)
b_forcing = Forcing(b_forcing_func, field_dependencies=:u)

u_forcings = MultipleForcings(u_forcing, uw_sponge)
b_forcings = MultipleForcings(b_forcing, b_sponge)

model = NonhydrostaticModel(; 
                            grid = grid,
                            closure = closure,
                            coriolis = FPlane(f=f),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            timestepper = :RungeKutta3,
                            advection = advection,
                            forcing = (u=u_forcings, v=v_sponge, w=uw_sponge, b=b_forcings),
                            boundary_conditions = (b=b_bcs, u=u_bcs))

set!(model, b=b_initial_noisy, v=v_initial)

b = model.tracers.b
u, v, w = model.velocities

Δt₀ = Lz / Nz / abs(M² / f) / 10
simulation = Simulation(model, Δt=Δt₀, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=1000, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/coriolis_parameter"] = f
    file["metadata/momentum_flux"] = Jᵁ
    file["metadata/buoyancy_flux"] = Jᴮ
    file["metadata/lateral_buoyancy_gradient"] = M²
    file["metadata/vertical_buoyancy_gradient"] = N²
    return nothing
end

ubar_y = Average(u, dims=2)
vbar_y = Average(v, dims=2)
wbar_y = Average(w, dims=2)
bbar_y = Average(b, dims=2)

ubar = Average(u, dims=(1, 2))
vbar = Average(v, dims=(1, 2))
wbar = Average(w, dims=(1, 2))
bbar = Average(b, dims=(1, 2))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (ubar_y, vbar_y, wbar_y, bbar_y, ubar, vbar, wbar, bbar)

simulation.output_writers[:xy] = JLD2OutputWriter(model, field_outputs,
                                                  filename = "$(FILE_DIR)/instantaneous_fields_xy",
                                                  indices = (:, :, Nz),
                                                  schedule = TimeInterval(10minutes),
                                                  with_halos = true,
                                                  init = init_save_some_metadata!,
                                                  overwrite_existing=true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, field_outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz",
                                                    indices = (:, 1, :),
                                                    schedule = TimeInterval(10minutes),
                                                    with_halos = true,
                                                    init = init_save_some_metadata!,
                                                    overwrite_existing=true)

simulation.output_writers[:yz] = JLD2OutputWriter(model, field_outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz",
                                                    indices = (1, :, :),
                                                    schedule = TimeInterval(10minutes),
                                                    with_halos = true,
                                                    init = init_save_some_metadata!,
                                                    overwrite_existing=true)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          overwrite_existing=true)

# simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
#                                                           filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
#                                                           schedule = TimeInterval(args["time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(args["checkpoint_interval"]days), prefix="$(FILE_DIR)/model_checkpoint")

# if pickup
#     files = readdir(FILE_DIR)
#     checkpoint_files = files[occursin.("model_checkpoint_iteration", files)]
#     if !isempty(checkpoint_files)
#         checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
#         pickup_iter = maximum(checkpoint_iters)
#         run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration$(pickup_iter).jld2")
#     else
#         run!(simulation)
#     end
# else
#     run!(simulation)
# end

# checkpointers = glob("$(FILE_DIR)/model_checkpoint_iteration*.jld2")
# if !isempty(checkpointers)
#     rm.(checkpointers)
# end

run!(simulation)
