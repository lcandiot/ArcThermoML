using JLD2
using Lux, Statistics, Random, LuxCUDA
using MAGEMin_C
using CairoMakie, Base, BenchmarkTools, Printf, Distributions, LinearAlgebra, DataFramesMeta
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

# Type definitions
DatType = Float32

# Define main
@views function main_speed_test(

)

    # Avoid BLAS oversubscription - only choose more threads for large number of points
    BLAS.set_num_threads(64)

    # Randomness
    seed = 123
    rng  = Random.MersenneTwister()
    Random.seed!(rng, seed)

    # Intialize data frame for storage
    df = DataFrame(
        "Solver"            => String[],
        "Npts"              => Int64[],
        "median(Time) [ns]" => Float64[],
        "min(Time) [ns]"    => Float64[],
        "max(Time) [ns]"    => Float64[],
        "mean(Time) [ns]"   => Float64[],
        "std(Time) [ns]"    => Float64[],
        "σ"                 => Float64[],
        "Mem.  [b]"         => Int64[],
        "Alloc."            => Int64[],
        "Julia threads"     => Int64[],
        "BLAS threads"      => Int64[],
        "Batch size"        => Int64[]
    )

    # Test numerics
    nPts_arr = [
        1,      # overhead-dominated
        10,     # tiny batch
        100,    # small
        256,    # exactly one NN batch
        1000,   # medium
        10000  # decade anchor
    ]
    nWarm = 256
    perc_arr = [0.0, 0.1, 1.0, 10.0]

    # Set paths
    data_dir   = "/media/largeData/ArcMagma_NeuralNetworks/MLP/NeuralNetworks/CrossValidation_062025"
    pdat_path  = "/home/lcandiot/Developer/MeltMigration_MagmaticSystems/paper_figures_data/data"
    mname      = "TrainedModel_256HD_3HL_16384batchsize_1000epochs_full.jld2"

    # Load the network
    model  = JLD2.load("$data_dir/$mname",     "MLP/model"           )
    x_mean = JLD2.load("$(data_dir)/$(mname)", "/Scaling/mean_input" )
    x_std  = JLD2.load("$(data_dir)/$(mname)", "/Scaling/std_input"  )
    state  = Lux.testmode(model.st)

    # Send data to gpu
    gdev = gpu_device()
    cdev = cpu_device()
    ps_g = model.ps |> gdev
    st_g = state    |> gdev

    # Define test point
    T = 800.0
    P = 5.0
    X_μ      = [ 52.1,   18.3, 10.5, 5.55,  7.24,  0.0,  0.38,   2.6,   0.68,    0.0,   4.0]
    
    # Initialize network
    #            SiO2, Al2O3, CaO,  MgO,  FeO,  K2O+Na2O,   TiO2,  H2O
    X_Net_μ   = [X_μ[1], X_μ[2], X_μ[3], X_μ[4], X_μ[5], X_μ[7] + X_μ[8], X_μ[9], X_μ[11]]
    X_Net_tot = sum(X_Net_μ)
    input     = Matrix{DatType}(undef, 10, 1)
    input[[1, 2], 1] .= DatType.([T, P])
    input[3:end, 1] .= DatType.(X_Net_μ ./ X_Net_tot)
    input[[2, 5, 6, 8, 9], :] .= log.(input[[2, 5, 6, 8, 9], :] .+ DatType(1e-5))

    # Initialize warmup multi-point input the network
    X_Net_multi_warm  = [Vector{Float64}(undef, size(X_Net_μ,  1)) for _ in 1:nWarm]
    input_multi_warm  = Matrix{DatType}(undef, 10, nWarm)
    input_multi_warm[[1,2], :] .= [T, P]
    for idx in 1:nWarm
        create_randomized_composition!(X_Net_μ, X_Net_multi_warm[idx], 0.1, rng)
        X_tot = sum(X_Net_multi_warm[idx])
        input_multi_warm[3:10, idx] .= X_Net_multi_warm[idx] ./ X_tot # Normalize composition
    end
    input_multi_warm[[2, 5, 6, 8, 9], :] .= log.(input_multi_warm[[2, 5, 6, 8, 9], :] .+ DatType(1e-5)) # Log transform
    input_multi_warm_sc = MAGEMin_MLPs.TransformData(input_multi_warm, transform = :zscore, dims = 2, scale = true, mean_data = x_mean, std_data = x_std; DatType = DatType) # scale

    # Define closures for Network calls
    Net_batch_CPU = () -> begin
        (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_warm_sc, model.ps, state)[1]
        return checksum(vcat(p1, vcat(p2, vcat(y1, vcat(y2, vcat(y3, vcat(y4,vcat(y5, y6))))))))
    end
    input_multi_warm_sc_g = input_multi_warm_sc |> gdev
    Net_batch_GPU = () -> begin
        (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_warm_sc_g, ps_g, st_g)[1]
        return checksum(vcat(cdev(p1), vcat(cdev(p2), vcat(cdev(y1), vcat(cdev(y2), vcat(cdev(y3), vcat(cdev(y4),vcat(cdev(y5), cdev(y6)))))))))
    end

    # Run warm-up
    Net_batch_CPU()
    Net_batch_GPU()

    # Sweep through parameter space and benchmark
    count = 0; nSweeps = size(nPts_arr, 1) * size(perc_arr, 1)
    for (idx_perc, perc) in enumerate(perc_arr)
        for (idx_pts, nPts) in enumerate(nPts_arr)

            # Update counter print status
            count += 1
            @printf "Benchmarking sweep %d / %d\n" count nSweeps

            # Initialize multi-point inputs for the network
            X_Net_multi  = [Vector{Float64}(undef, size(X_Net_μ,    1)) for _ in 1:nPts]
            input_multi  = Matrix{DatType}(undef, 10, nPts)
            input_multi[[1,2], :] .= [T, P]
            for idx in 1:nPts
                create_randomized_composition!(X_Net_μ, X_Net_multi[idx], perc, rng)
                X_tot = sum(X_Net_multi[idx])
                input_multi[3:10, idx] .= X_Net_multi[idx] ./ X_tot # Normalize composition
            end

            input_multi[[2, 5, 6, 8, 9], :] .= log.(input_multi[[2, 5, 6, 8, 9], :] .+ DatType(1e-5)) # Log transform
            input_multi_sc = MAGEMin_MLPs.TransformData(input_multi, transform = :zscore, dims = 2, scale = true, mean_data = x_mean, std_data = x_std; DatType = DatType) # scale

            # Re-define closures to include local scope vars -> saves allocations
            Net_batch_CPU = () -> begin
                (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_sc, model.ps, state)[1]
            end

            # Re-define closures to include local scope vars -> saves allocations
            input_multi_sc_g = input_multi_sc |> gdev
            Net_batch_GPU = () -> begin
                (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_sc_g, ps_g, st_g)[1]
            end

            # Run batched benchmark store the results in a data frame
            tr_Net_CPU = @benchmark $Net_batch_CPU()
            push!(df, ("Net CPU",  nPts, median(tr_Net_CPU).time, minimum(tr_Net_CPU).time, maximum(tr_Net_CPU).time, mean(tr_Net_CPU).time, std(tr_Net_CPU).time, perc, tr_Net_CPU.memory, tr_Net_CPU.allocs, Threads.nthreads(), BLAS.get_num_threads(), 0)
            )
            # Run batched benchmark store the results in a data frame
            tr_Net_GPU = @benchmark $Net_batch_GPU()
            push!(df, ("Net GPU",  nPts, median(tr_Net_GPU).time, minimum(tr_Net_GPU).time, maximum(tr_Net_GPU).time, mean(tr_Net_GPU).time, std(tr_Net_GPU).time, perc, tr_Net_GPU.memory, tr_Net_GPU.allocs, Threads.nthreads(), BLAS.get_num_threads(), 0)
            )

            # Monitor
            println("Median time for $(nPts) Pts on CPU: $(median(tr_Net_CPU).time / 1e9) s")
            println("Median time for $(nPts) Pts on GPU: $(median(tr_Net_GPU).time / 1e9) s")
        end
    end

    # Store timer output as dict
    if !isdir(pdat_path)
        println("Creating directory: $pdat_path")
        mkpath(pdat_path)
    else
        println("Directory already exists: $pdat_path")
    end
    JLD2.save("$(pdat_path)/BenchmarkTools_Network_peakperformance_64BLASthreads_withGPU.jld2", "df", df)

    # Return
    return nothing
end

# Create initial guess
function create_randomized_composition!(
    A   :: AbstractArray{T},
    A1  :: AbstractArray{T},
    p   :: T,
    rng :: AbstractRNG
) where T <: Real
    for (idx, el) in enumerate(A)
        σ = abs(p/100.0*el)
        ϵ = clamp(rand(rng, Distributions.Normal(0.0, σ)), -σ, σ)
        A1[idx] = el + ϵ
    end
end

# Minimal, fast checksum: numbers, arrays, tuples, namedtuples, structs, strings
@inline checksum(x) = (acc = Base.RefValue{Float64}(0.0); checksum!(acc, x); acc[])

@inline checksum!(acc::Base.RefValue{Float64}, x::Number) = (acc[] += Float64(x); acc)
@inline checksum!(acc::Base.RefValue{Float64}, ::Nothing) = acc
@inline checksum!(acc::Base.RefValue{Float64}, ::Missing) = acc
@inline checksum!(acc::Base.RefValue{Float64}, b::Bool)   = (acc[] += b ? 1.0 : 0.0; acc)

@inline function checksum!(acc::Base.RefValue{Float64}, s::AbstractString)
    cu = codeunits(s)
    @inbounds @simd for i in eachindex(cu)
        acc[] += cu[i]
    end
    acc
end

@inline function checksum!(acc::Base.RefValue{Float64}, A::AbstractArray)
    @inbounds @simd for i in eachindex(A)
        xi = A[i]
        if xi isa Number
            acc[] += Float64(xi)
        else
            checksum!(acc, xi)
        end
    end
    acc
end

@inline function checksum!(acc::Base.RefValue{Float64}, t::Tuple)
    @inbounds for i in 1:length(t)
        checksum!(acc, t[i])
    end
    acc
end

@inline function checksum!(acc::Base.RefValue{Float64}, nt::NamedTuple)
    @inbounds for v in values(nt)
        checksum!(acc, v)
    end
    acc
end

@inline function checksum!(acc::Base.RefValue{Float64}, d::AbstractDict)
    for (k, v) in d
        checksum!(acc, k); checksum!(acc, v)
    end
    acc
end

@inline function checksum!(acc::Base.RefValue{Float64}, x::T) where {T}
    @inbounds for fn in fieldnames(T)
        checksum!(acc, getfield(x, fn))
    end
    acc
end

# Run speed test
main_speed_test();