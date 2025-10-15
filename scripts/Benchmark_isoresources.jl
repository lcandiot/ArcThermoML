using JLD2
using Lux, Statistics, Random
using MAGEMin_C
using CairoMakie, Base, BenchmarkTools, Printf, Distributions, LinearAlgebra, DataFramesMeta
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

# Type definitions
DatType = Float32

# Define main
@views function main_speed_test(

)

    # Avoid BLAS oversubscription
    BLAS.set_num_threads(1)

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

    # Testing metrics
    nPts_arr = [
        1,      # overhead-dominated
        10,     # tiny batch
        100,    # small
        256,    # exactly one NN batch
        1000,   # medium
        10000  # decade anchor
    ]
    bsize = 256
    nWarm = copy(bsize)
    perc_arr = [0.0, 0.1, 1.0, 10.0]

    # Set paths
    data_dir   = "./data/surrogate_model"
    pdat_path  = "./user"
    mname      = "TrainedModel_256HD_3HL_16384batchsize_1000epochs_full.jld2"

    # Load the network
    model  = JLD2.load("$data_dir/$mname",     "MLP/model"           )
    x_mean = JLD2.load("$(data_dir)/$(mname)", "/Scaling/mean_input" )
    x_std  = JLD2.load("$(data_dir)/$(mname)", "/Scaling/std_input"  )
    state  = Lux.testmode(model.st)

    # Define test point
    T = 800.0
    P = 5.0
    Xoxides  = ["SiO2"; "Al2O3"; "CaO"; "MgO"; "FeO"; "O"; "K2O"; "Na2O"; "TiO2"; "Cr2O3"; "H2O"]
    X_μ      = [ 52.1,   18.3, 10.5, 5.55,  7.24,  0.0,  0.38,   2.6,   0.68,    0.0,   4.0]
    X_tot    = sum(X_μ)
    X_MAGE_μ   = X_μ ./ X_tot
    
    # Initialize network
    #            SiO2, Al2O3, CaO,  MgO,  FeO,  K2O+Na2O,   TiO2,  H2O
    X_Net_μ   = [X_μ[1], X_μ[2], X_μ[3], X_μ[4], X_μ[5], X_μ[7] + X_μ[8], X_μ[9], X_μ[11]]
    X_Net_tot = sum(X_Net_μ)
    input     = Matrix{DatType}(undef, 10, 1)
    input[[1, 2], 1] .= DatType.([T, P])
    input[3:end, 1] .= DatType.(X_Net_μ ./ X_Net_tot)
    input[[2, 5, 6, 8, 9], :] .= log.(input[[2, 5, 6, 8, 9], :] .+ DatType(1e-5))

    # Initialize MAGEMin
    db     = "ig"
    data   = Initialize_MAGEMin(db, verbose=-1, solver=0)  # <- Legacy solver is needed for initial guesses
    sys_in = "wt"

    # Get initial guess for average composition
    out = single_point_minimization(P, T, data, X=X_MAGE_μ, Xoxides=Xoxides, sys_in=sys_in, progressbar=false)
    Gig = out.mSS_vec

    # Initialize warmup multi-point inputs for MAGEMin and the network
    X_MAGE_multi_warm = [Vector{Float64}(undef, size(X_MAGE_μ, 1)) for _ in 1:nWarm]
    X_Net_multi_warm  = [Vector{Float64}(undef, size(X_Net_μ,  1)) for _ in 1:nWarm]
    T_multi_warm      = [T      for _ in 1:nWarm]
    P_multi_warm      = [P      for _ in 1:nWarm]
    Gig_multi_warm    = [Gig    for _ in 1:nWarm]
    input_multi_warm  = Matrix{DatType}(undef, 10, nWarm)
    input_multi_warm[[1,2], :] .= [T, P]
    for idx in 1:nWarm
        create_randomized_composition!(X_MAGE_μ, X_MAGE_multi_warm[idx], 0.1, rng)
        create_randomized_composition!(X_Net_μ, X_Net_multi_warm[idx], 0.1, rng)
        X_tot = sum(X_Net_multi_warm[idx])
        input_multi_warm[3:10, idx] .= X_Net_multi_warm[idx] ./ X_tot # Normalize composition
    end
    input_multi_warm[[2, 5, 6, 8, 9], :] .= log.(input_multi_warm[[2, 5, 6, 8, 9], :] .+ DatType(1e-5)) # Log transform
    input_multi_warm_sc = MAGEMin_MLPs.TransformData(input_multi_warm, transform = :zscore, dims = 2, scale = true, mean_data = x_mean, std_data = x_std; DatType = DatType) # scale

    # Define closures for MAGEMin and Network calls
    Net_batch = (j, j2) -> begin
        (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_warm_sc[:, j:j2], model.ps, state)[1]
        return checksum(vcat(p1, vcat(p2, vcat(y1, vcat(y2, vcat(y3, vcat(y4,vcat(y5, y6))))))))
    end
    MAGE_batch = (j, j2) -> begin
        Out_multi = multi_point_minimization(collect(P_multi_warm[j:j2]), collect(T_multi_warm[j:j2]), data, X = collect(X_MAGE_multi_warm[j:j2]), Xoxides = Xoxides, sys_in = sys_in, progressbar = false)
        return checksum(Out_multi)
    end
    MAGE_iguess_batch = (j, j2) -> begin
        Out_multi = multi_point_minimization(collect(P_multi_warm[j:j2]), collect(T_multi_warm[j:j2]), data, X = collect(X_MAGE_multi_warm[j:j2]), Xoxides = Xoxides, sys_in = sys_in, iguess=true, G=collect(Gig_multi_warm[j:j2]), progressbar = false)
        return checksum(Out_multi)
    end

    # Run warm-up
    run_batched(MAGE_batch,        nWarm; batchsize = bsize)
    run_batched(MAGE_iguess_batch, nWarm; batchsize = bsize)
    run_batched(Net_batch,         nWarm; batchsize = bsize)

    # Sweep through parameter space and benchmark
    count = 0; nSweeps = size(nPts_arr, 1) * size(perc_arr, 1)
    for (_, perc) in enumerate(perc_arr)
        for (_, nPts) in enumerate(nPts_arr)

            # Update counter print status
            count += 1
            @printf "Benchmarking sweep %d / %d\n" count nSweeps

            # Initialize multi-point inputs for MAGEMin and the network
            X_MAGE_multi = [Vector{Float64}(undef, size(X_MAGE_μ, 1)) for _ in 1:nPts]
            T_multi      = [T      for _ in 1:nPts]
            P_multi      = [P      for _ in 1:nPts]
            Gig_multi    = [Gig    for _ in 1:nPts]
            X_Net_multi  = [Vector{Float64}(undef, size(X_Net_μ,    1)) for _ in 1:nPts]
            input_multi  = Matrix{DatType}(undef, 10, nPts)
            input_multi[[1,2], :] .= [T, P]
            for idx in 1:nPts
                create_randomized_composition!(X_MAGE_μ, X_MAGE_multi[idx], perc, rng)
                create_randomized_composition!(X_Net_μ, X_Net_multi[idx], perc, rng)
                X_tot = sum(X_Net_multi[idx])
                input_multi[3:10, idx] .= X_Net_multi[idx] ./ X_tot # Normalize composition
            end
            input_multi[[2, 5, 6, 8, 9], :] .= log.(input_multi[[2, 5, 6, 8, 9], :] .+ DatType(1e-5)) # Log transform
            input_multi_sc = MAGEMin_MLPs.TransformData(input_multi, transform = :zscore, dims = 2, scale = true, mean_data = x_mean, std_data = x_std; DatType = DatType) # scale

            # Re-define closures to include local scope vars -> saves allocations
            Net_batch = (j, j2) -> begin
                (p1, p2, y1, y2, y3, y4, y5, y6) = Lux.apply(model.model.u, input_multi_sc[:, j:j2], model.ps, state)[1]
            end
            MAGE_batch = (j, j2) -> begin
                Out_multi = multi_point_minimization(collect(P_multi[j:j2]), collect(T_multi[j:j2]), data, X = collect(X_MAGE_multi[j:j2]), Xoxides = Xoxides, sys_in = sys_in, progressbar = false)
            end
            MAGE_iguess_batch = (j, j2) -> begin
                Out_multi = multi_point_minimization(collect(P_multi[j:j2]), collect(T_multi[j:j2]), data, X = collect(X_MAGE_multi[j:j2]), Xoxides = Xoxides, sys_in = sys_in, iguess=true, G=collect(Gig_multi[j:j2]), progressbar = false)
            end

            # Run batched benchmark store the results in a data frame
            tr_MAGE        = @benchmark run_batched($MAGE_batch,        $nPts; batchsize = $bsize)
            tr_MAGE_iguess = @benchmark run_batched($MAGE_iguess_batch, $nPts; batchsize = $bsize)
            tr_Net         = @benchmark run_batched($Net_batch,         $nPts; batchsize = $bsize)
            push!(df, ("MAGE", nPts, median(tr_MAGE).time, minimum(tr_MAGE).time, maximum(tr_MAGE).time, mean(tr_MAGE).time, std(tr_MAGE).time, perc, tr_MAGE.memory, tr_MAGE.allocs, Threads.nthreads(), BLAS.get_num_threads(), bsize)
            )
            push!(df, ("MAGE iguess", nPts, median(tr_MAGE_iguess).time, minimum(tr_MAGE_iguess).time, maximum(tr_MAGE_iguess).time, mean(tr_MAGE_iguess).time, std(tr_MAGE_iguess).time, perc, tr_MAGE_iguess.memory, tr_MAGE_iguess.allocs, Threads.nthreads(), BLAS.get_num_threads(), bsize)
            )
            push!(df, ("Net",  nPts, median(tr_Net).time, minimum(tr_Net).time, maximum(tr_Net).time, mean(tr_Net).time, std(tr_Net).time, perc, tr_Net.memory, tr_Net.allocs, Threads.nthreads(), BLAS.get_num_threads(), bsize)
            )

        end
    end

    # Free MAGEMin and show timing results
    Finalize_MAGEMin(data)

    # Store timer output as dict
    if !isdir(pdat_path)
        println("Creating directory: $pdat_path")
        mkpath(pdat_path)
    else
        println("Directory already exists: $pdat_path")
    end
    JLD2.save("$(pdat_path)/BenchmarkTools_isoresources.jld2", "df", df)

    # Return
    return nothing
end

# Runner function for batched evaluation
function run_batched(f, Npts; batchsize :: Int64 = 256)
    for j in 1:batchsize:Npts
        j2 = min(j + batchsize - 1, Npts)
        f(j, j2)
    end
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