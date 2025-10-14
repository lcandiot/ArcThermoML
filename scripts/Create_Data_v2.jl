using MAGEMin_C
using JLD2, Printf, DataFramesMeta, DelimitedFiles
using Base, ProgressMeter

"""

```
    CreateData(; kwargs)
```

Create the training and testing data for the Physics-Informed Neural Network stored as MAGEMin output structure in `file_name`. Currently `file_name` should be a string that contains the path to a JLD2 file.


# Keyword arguments
* `fload_name` : Name of MAGEMin output file
* `fsave_name` : Name of file to save that contains the training data
"""
function CreateData(; 
    fload_name :: String,
    fsave_name :: String
)
    
    # Load MAGEMin structure
    @printf "\t ... Loading %s ...\n" fload_name
    out = JLD2.load(fload_name, "out")
    @printf "\t ... completed"

    # Allocate storage arrays
    N = size(out, 1)
    input_data  = zeros(Float32, 10, N)
    output_data = zeros(Float32, 11, N)
    bulk   = [Vector{Float64}(undef, 8) for _ in 1:N]
    bulk_M = [Vector{Float64}(undef, 8) for _ in 1:N]

    # Identify oxide indices
    idx_Na = findfirst(x -> x == "Na2O",  out[1].oxides)
    idx_K  = findfirst(x -> x == "K2O",   out[1].oxides)
    idx_O  = findfirst(x -> x == "O",     out[1].oxides)
    idx_Cr = findfirst(x -> x == "Cr2O3", out[1].oxides)

    # Initialize progress bar
    progr = Progress(N, desc="Extracting $(N) points...")

    # Extract data
    Threads.@threads for idx in eachindex(out)

        if isnan(out[idx].bulk_M_wt[1])
            for (iC, C) in enumerate(out[idx].bulk_M_wt)
                out[idx].bulk_M_wt[iC] = 0.0
            end
        end
        if isnan(out[idx].bulk_S_wt[1])
            for (iC, C) in enumerate(out[idx].bulk_S_wt)
                out[idx].bulk_S_wt[iC] = 0.0
            end
        end

        # Input
        bulk[idx] = out[idx].bulk_wt
        # println(out[idx].oxides)
        # println(bulk[idx])
        # @printf "Na2O: %.8f\t K2O: %.8f\t Na2O+K2O: %.8f\n" bulk[idx][idx_Na] bulk[idx][idx_K] (bulk[idx][idx_Na]+bulk[idx][idx_K])
        bulk[idx][idx_Na] += bulk[idx][idx_K]
        deleteat!(bulk[idx], [idx_K, idx_O, idx_Cr])
        # println(bulk[idx])

        input_data[1,    idx]  = Float32(out[idx].T_C   )
        input_data[2,    idx]  = Float32(out[idx].P_kbar)
        input_data[3:10, idx] .= Float32.(bulk[idx]     )

        # Output
        bulk_M[idx] = deepcopy(out[idx].bulk_M_wt)

        # println(out[idx].oxides)
        # println(bulk_M[idx])
        # @printf "Na2O_M: %.8f\t K2O_M: %.8f\t Na2O_M+K2O_M: %.8f\n" bulk_M[idx][idx_Na] bulk_M[idx][idx_K] (bulk_M[idx][idx_Na]+bulk_M[idx][idx_K])
        bulk_M[idx][idx_Na] += bulk_M[idx][idx_K]
        deleteat!(bulk_M[idx], [idx_K, idx_O, idx_Cr])
        # println(bulk_M[idx])

        output_data[1:8,  idx ] .= Float32.( bulk_M[idx]        )
        output_data[9,    idx ]  = Float32(  out[idx].frac_M_wt )
        output_data[10,    idx]  = Float32(  out[idx].rho       )
        output_data[11,    idx]  = Float32(  out[idx].rho_M     )
        
        # Update progess bar
        next!(progr)
    end

    # Free progress bar
    finish!(progr)

    # Store the data
    jldopen(fsave_name, "w") do file
        data = JLD2.Group(file, "data")
        data["input"]  = input_data
        data["output"] = output_data
    end

    # Return
    return nothing
end


function CreateDataAMR(
    bulk_fname :: String,
    ini_lev    :: Int64,
    sub_lev    :: Int64,
    Trange     :: Tuple{Float64, Float64},
    Prange     :: Tuple{Float64, Float64},
    database   :: String,
    bulk_unit  :: String
)
    # Load bulk rock
    data, header = readdlm(bulk_fname, ',', header = true)
    df           = DataFrame(data, vec(header))
    idx_neg = []
    for idx in axes(df, 1)
        ID = findfirst(x -> x < 0.0, df[idx, :])
        isnothing(ID) ? continue : push!(idx_neg, idx)
    end
    deleteat!(df, idx_neg)

    df = @chain df begin
        @rsubset :SiO2 <= 55.0
    end
    display(df)
end
# Run
# out = CreateData(file_name = "/home/lcandiot/Desktop/ArcMagma_MAGEMin_database_nAC25_nT15_nP15_nH2O10_nPts56250_outstruct.jld2");