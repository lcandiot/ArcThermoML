# This version uses a different data set for training and testing. Compositions of training and testing data sets have a similar distribution but it is made sure that they do not contain the exact same points
using JLD2, DataFramesMeta, DelimitedFiles
using MAGEMin_C
using Lux, Random, Zygote, Optimisers, MLUtils, LuxCUDA
using KernelDensity, Statistics, Distributions
using Printf, CairoMakie, ColorSchemes
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

CUDA.allowscalar(false)

# Check if CUDA is functional and supports Float64
function send_to_device(;
    DatType :: Type = Float32
    )
    if CUDA.functional()
        return x -> CUDA.adapt(CuArray{DatType}, x)
    else
        error("CUDA is not available or doesn't support $(DatType).")
    end
end

# Set up devices
cdev = cpu_device()
const gdev = gpu_device()
# gdev = send_to_device(; DatType)

# Tolerances
const ϵ_liq = 1e-3
const ϵ_flu = 1e-3

@views function CreateDataAMR_NonGridded(
    bulk_fname :: String,
    ini_lev    :: Int64,
    sub_lev    :: Int64,
    Trange     :: Tuple{Float64, Float64},
    Prange     :: Tuple{Float64, Float64},
    Sirange    :: Tuple{Float64, Float64},
    database   :: String,
    bulk_unit  :: String;
    write2disk :: Bool = false,
    save_fname :: Union{String, Nothing} = nothing
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
        @rsubset :SiO2 >= Sirange[1] && :SiO2 <= Sirange[2]
    end

    # Regular grid settings for interpolation
    ngrid  = 2^(ini_lev + sub_lev)
    Ti     = range(Trange[1], Trange[2], length=ngrid) |> x -> collect(x)
    Pi     = range(Prange[1], Prange[2], length=ngrid) |> x -> collect(x)
    T_gvec = repeat(Ti,  ngrid) |> T_gvec -> reshape(T_gvec, ngrid, ngrid)[:]
    P_gvec = repeat(Pi', ngrid) |> P_gvec -> reshape(P_gvec, ngrid, ngrid)[:]

    # Initialize input and output matrices
    input  = zeros(Float32, 10, 1)
    output = zeros(Float32, 26, 1)

    # Loop through each composition and create phase diagram
    for (idx_X) in axes(df, 1)

        # Extract bulk
        X = [
                df[idx_X, :SiO2],
                df[idx_X, :Al2O3],
                df[idx_X, :FeOT],
                df[idx_X, :MgO],
                df[idx_X, :CaO],
                df[idx_X, :Na2O],
                df[idx_X, :K2O],
                df[idx_X, :TiO2],
                4.0,                # H2O
                0.0,                # Cr2O3
                0.0,                # O
        ]

        # Call MAGEMin AMR
        data    = Initialize_MAGEMin(database, verbose = false, solver = 2)
        oxides  = ["SiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2", "H2O", "Cr2O3", "O"]
        ph_rm   = remove_phases(["fl"], "ig")
        out     = AMR_minimization(ini_lev, sub_lev, Prange, Trange, data, X=X, Xoxides = oxides, sys_in = bulk_unit,rm_list = ph_rm)
        Finalize_MAGEMin(data)

        # Postprocess output
        npts     = length(out)
        out_curr = Matrix{Float64}(undef, 26, npts)
        in_curr  = Matrix{Float64}(undef, 10, npts)
        T_out    = Vector{Float64}(undef, npts)
        P_out    = Vector{Float64}(undef, npts)
        Φ_Liq    = Vector{Float64}(undef, npts)
        Φ_Flu    = Vector{Float64}(undef, npts)
        Φ_Sol    = Vector{Float64}(undef, npts)
        ρ_Liq    = Vector{Float64}(undef, npts)
        ρ_Sys    = Vector{Float64}(undef, npts)
        ρ_Flu    = Vector{Float64}(undef, npts)
        ρ_Sol    = Vector{Float64}(undef, npts)
        Si_Liq   = Vector{Float64}(undef, npts)
        Al_Liq   = Vector{Float64}(undef, npts)
        Ca_Liq   = Vector{Float64}(undef, npts)
        Mg_Liq   = Vector{Float64}(undef, npts)
        Fe_Liq   = Vector{Float64}(undef, npts)
        KN_Liq   = Vector{Float64}(undef, npts)
        Ti_Liq   = Vector{Float64}(undef, npts)
        H_Liq    = Vector{Float64}(undef, npts)
        Si_Sol   = Vector{Float64}(undef, npts)
        Al_Sol   = Vector{Float64}(undef, npts)
        Ca_Sol   = Vector{Float64}(undef, npts)
        Mg_Sol   = Vector{Float64}(undef, npts)
        Fe_Sol   = Vector{Float64}(undef, npts)
        KN_Sol   = Vector{Float64}(undef, npts)
        Ti_Sol   = Vector{Float64}(undef, npts)
        H_Sol    = Vector{Float64}(undef, npts)
        for (idx_out, Out) in enumerate(out)
            T_out[idx_out] = Out.T_C
            P_out[idx_out] = Out.P_kbar
            ρ_Liq[idx_out] = Out.rho_M
            ρ_Sys[idx_out] = Out.rho
            ρ_Flu[idx_out] = Out.rho_F
            ρ_Sol[idx_out] = Out.rho_S
            Φ_Liq[idx_out] = Out.frac_M_wt
            Φ_Flu[idx_out] = Out.frac_F_wt
            Φ_Sol[idx_out] = Out.frac_S_wt
            isnan(Out.bulk_M_wt[1]) ? Si_Liq[idx_out] = 0.0 : Si_Liq[idx_out] = Out.bulk_M_wt[1]
            isnan(Out.bulk_M_wt[1]) ? Al_Liq[idx_out] = 0.0 : Al_Liq[idx_out] = Out.bulk_M_wt[2]
            isnan(Out.bulk_M_wt[1]) ? Ca_Liq[idx_out] = 0.0 : Ca_Liq[idx_out] = Out.bulk_M_wt[3]
            isnan(Out.bulk_M_wt[1]) ? Mg_Liq[idx_out] = 0.0 : Mg_Liq[idx_out] = Out.bulk_M_wt[4]
            isnan(Out.bulk_M_wt[1]) ? Fe_Liq[idx_out] = 0.0 : Fe_Liq[idx_out] = Out.bulk_M_wt[5]
            isnan(Out.bulk_M_wt[1]) ? KN_Liq[idx_out] = 0.0 : KN_Liq[idx_out] = Out.bulk_M_wt[6] + Out.bulk_M_wt[7]
            isnan(Out.bulk_M_wt[1]) ? Ti_Liq[idx_out] = 0.0 : Ti_Liq[idx_out] = Out.bulk_M_wt[8]
            isnan(Out.bulk_M_wt[1]) ? H_Liq[idx_out ] = 0.0 : H_Liq[idx_out ] = Out.bulk_M_wt[11]
            isnan(Out.bulk_S_wt[1]) ? Si_Sol[idx_out] = 0.0 : Si_Sol[idx_out] = Out.bulk_S_wt[1]
            isnan(Out.bulk_S_wt[1]) ? Al_Sol[idx_out] = 0.0 : Al_Sol[idx_out] = Out.bulk_S_wt[2]
            isnan(Out.bulk_S_wt[1]) ? Ca_Sol[idx_out] = 0.0 : Ca_Sol[idx_out] = Out.bulk_S_wt[3]
            isnan(Out.bulk_S_wt[1]) ? Mg_Sol[idx_out] = 0.0 : Mg_Sol[idx_out] = Out.bulk_S_wt[4]
            isnan(Out.bulk_S_wt[1]) ? Fe_Sol[idx_out] = 0.0 : Fe_Sol[idx_out] = Out.bulk_S_wt[5]
            isnan(Out.bulk_S_wt[1]) ? KN_Sol[idx_out] = 0.0 : KN_Sol[idx_out] = Out.bulk_S_wt[6] + Out.bulk_S_wt[7]
            isnan(Out.bulk_S_wt[1]) ? Ti_Sol[idx_out] = 0.0 : Ti_Sol[idx_out] = Out.bulk_S_wt[8]
            isnan(Out.bulk_S_wt[1]) ? H_Sol[idx_out ] = 0.0 : H_Sol[idx_out ] = Out.bulk_S_wt[11]
        end

        # Create melt binary
        Solid = deepcopy(Φ_Sol)
        Melt  = deepcopy(Φ_Liq)
        Fluid = deepcopy(Φ_Flu)
        Melt[Φ_Liq   .> 0.0] .= 1.0
        Fluid[Φ_Flu  .> 0.0] .= 1.0
        Solid[Φ_Sol  .> 0.0] .= 1.0

        # Store data points
        in_curr[1,  :]  .= T_out
        in_curr[2,  :]  .= P_out
        in_curr[3,  :]  .= out[1].bulk_wt[1]
        in_curr[4,  :]  .= out[1].bulk_wt[2]
        in_curr[5,  :]  .= out[1].bulk_wt[3]
        in_curr[6,  :]  .= out[1].bulk_wt[4]
        in_curr[7,  :]  .= out[1].bulk_wt[5]
        in_curr[8,  :]  .= out[1].bulk_wt[6] + out[1].bulk_wt[7]
        in_curr[9,  :]  .= out[1].bulk_wt[8]
        in_curr[10, :]  .= out[1].bulk_wt[11]
        out_curr[1, :]  .= Melt[:] 
        out_curr[2, :]  .= Fluid[:]
        out_curr[3, :]  .= Si_Liq[:]
        out_curr[4, :]  .= Al_Liq[:]
        out_curr[5, :]  .= Ca_Liq[:]
        out_curr[6, :]  .= Mg_Liq[:]
        out_curr[7, :]  .= Fe_Liq[:]
        out_curr[8, :]  .= KN_Liq[:]
        out_curr[9, :]  .= Ti_Liq[:]
        out_curr[10, :] .= H_Liq[:] 
        out_curr[11, :] .= Φ_Liq[:] 
        out_curr[12, :] .= Φ_Flu[:] 
        out_curr[13, :] .= ρ_Sys[:] 
        out_curr[14, :] .= ρ_Liq[:] 
        out_curr[15, :] .= ρ_Flu[:] 
        out_curr[16, :] .= Solid[:] 
        out_curr[17, :] .= Si_Sol[:]
        out_curr[18, :] .= Al_Sol[:]
        out_curr[19, :] .= Ca_Sol[:]
        out_curr[20, :] .= Mg_Sol[:]
        out_curr[21, :] .= Fe_Sol[:]
        out_curr[22, :] .= KN_Sol[:]
        out_curr[23, :] .= Ti_Sol[:]
        out_curr[24, :] .= H_Sol[:] 
        out_curr[25, :] .= Φ_Sol[:] 
        out_curr[26, :] .= ρ_Sol[:] 
        
        # Add to global storage
        output = hcat(output, out_curr)
        input  = hcat(input, in_curr)
    end

    # Write to disk
    if write2disk
        jldopen(save_fname, "w") do file
            data = JLD2.Group(file, "data")
            data["input"]  = input[:, 2:end]
            data["output"] = output[:, 2:end]
        end
    end
end

function TrainIgneous(;
    create_data     :: Bool = false,
    create_data_fig :: Bool = false,
    data_file       :: Union{String, Nothing} = nothing,
    log_transform   :: Bool = false,
    DatType         :: Type = Float32
)

    # Randomness
    seed = 123
    rng  = Random.MersenneTwister()
    Random.seed!(rng, seed)

    # Create the data
    if create_data
        T_range  = (653.0, 1097.0)
        P_range  = (0.02,    9.9)
        Si_range = (45.0,    71.0)
        ini_lev  = 5
        sub_lev  = 2
        database = "ig"
        sys_in   = "wt"
        
        # Sanity check
        if isnothing(data_file)
            MAGEMin_MLPs.BadDefaultError(:data_file)
        end

        bulk_fname = data_file
        CreateDataAMR_NonGridded(bulk_fname, ini_lev, sub_lev, T_range, P_range, Si_range, database, sys_in; write2disk = true, save_fname = "./user/SynGEOROC_testing_data_$(ini_lev)iniLev_$(sub_lev)sub_lev_$(Si_range[1])_$(Si_range[2])wtSi_$(T_range[1])_$(T_range[2])C_$(P_range[1])_$(P_range[2])kbar.jld2")

        # Early return
        return
    end

    # Set hyperparameters for tuning
    hidden_dims = [64, 128, 256]
    hidden_lays = [5]
    epochs      = 5_00
    batchsizes  = [1024] #[16384, 4096, 1024] -- We had to perform the cross validation per batch size in separate sessions for technical reasons
    
    # Load original input and output data
    input, output = MAGEMin_MLPs.LoadData(data_path = "./user/zenodo_data/SynGEOROC_training_data_5iniLev_2sub_lev_45.0_71.0wtSi_650.0_1300.0C_0.1_10.0kbar.jld2")

    # Prepare the data -- shuffling is crucial!
    Ndpts        = size(output, 2)
    perm         = shuffle(rng, 1:Ndpts)
    input_shuff  = input[:,      perm]
    output_shuff = output[1:15, perm]

    # New order: Bin. Class. melt, bin. class. fluid, SiO2l, Al2O3l, CaOl, (Na2O+K2O)l, MgOl, FeOl, TiO2l, H2O, ρSys, ϕl, ϕf, ρLiq, ρFlu
    output_ord = deepcopy(output_shuff)
    output_ord[[5, 6, 7, 8, 11, 12, 13, 14],    :] .= output_shuff[[5, 8, 6, 7, 13, 11, 12, 14],   :]

    @printf "No. points in data set: %d\n" size(output_ord, 2)
    @printf "Input data features:\n"
    for i in axes(input_shuff, 1)
        @printf "Min feat = %.5g \t Max feat = %.5g\n" minimum(input_shuff[i, :]) maximum(input_shuff[i, :])
    end
    @printf "Output data features:\n"
    for i in axes(output_ord, 1)
        @printf "Min feat = %.5g \t Max feat = %.5g\n" minimum(output_ord[i, :]) maximum(output_ord[i, :])
    end

    if create_data_fig
        # Output features
        spl_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
        fg1 = Figure(size = (1200, 800))
        axs1 = [Axis(fg1[row, col], ylabel =L"$$No. pts", aspect = 1.681) for row in 1:4 for col in 1:4]
        for idx in axes(output_ord, 1)
            out = deepcopy(output_ord[idx, :])
            (idx > 2 && idx < 11 || idx == 12 || idx == 13) ? out .*= 100.0 : nothing
            sc = hist!(axs1[idx], out)
        end
        axs1[1].xlabel = L"$$Liquid Binary [ ]"
        axs1[2].xlabel = L"$$Fluid Binary [ ]"
        axs1[3].xlabel = L"SiO$_2^{\text{Liq}}$ [wt.%]"
        axs1[4].xlabel = L"Al$_2$O$_3^{\text{Liq}}$ [wt.%]"
        axs1[5].xlabel = L"CaO$^{\text{Liq}}$ [wt.%]"
        axs1[6].xlabel = L"(Na$_2$O + K$_2$O)$^{\text{Liq}}$ [wt.%]"
        axs1[7].xlabel = L"MgO$^{\text{Liq}}$ [wt.%]"
        axs1[8].xlabel = L"FeO$^{\text{Liq}}$ [wt.%]"
        axs1[9].xlabel = L"TiO$_2^{\text{Liq}}$ [wt.%]"
        axs1[10].xlabel = L"H$_2$O$^{\text{Liq}}$ [wt.%]"
        axs1[11].xlabel = L"$\rho^{\text{Sys}}$ [kg.m$^{-3}$]"
        axs1[12].xlabel = L"$\phi^{\text{Liq}}$ [wt.%]"
        axs1[13].xlabel = L"$\phi^{\text{Flu}}$ [wt.%]"
        axs1[14].xlabel = L"$\rho^{\text{Liq}}$ [kg.m$^{-3}$]"
        axs1[15].xlabel = L"$\rho^{\text{Flu}}$ [kg.m$^{-3}$]"
        axs1[1].yticks =([3e5, 6e5, 9e5], ["300k", "600k", "900k"])
        axs1[1].xticks =([0, 1], ["0", "1"])
        axs1[2].yticks =([3e5, 6e5, 9e5], ["300k", "600k", "900k"])
        axs1[2].xticks =([0, 1], ["0", "1"])
        axs1[3].yticks =([1e5, 2e5], ["100k", "200k"])
        axs1[3].xticks = 0.0:15.0:90.0
        axs1[4].yticks =([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        axs1[4].xticks = 0.0:7:90.0
        axs1[5].yticks =([2e5, 4e5, 6e5], ["200k", "400k", "600k"])
        axs1[5].xticks = 0.0:5.0:90
        axs1[6].yticks =([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        axs1[7].yticks =([2e5, 4e5, 6e5], ["200k", "400k", "600k"])
        axs1[7].xticks = 0.0:5.0:90.0
        axs1[8].yticks =([2e5, 4e5, 6e5], ["200k", "400k", "600k"])
        axs1[9].xticks = 0.0:2.0:90.0
        axs1[9].yticks =([2e5, 4e5, 6e5], ["200k", "400k", "600k"])
        axs1[10].yticks =([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        axs1[11].yticks =([5e4, 1.0e5], ["50k", "100k"])
        axs1[12].yticks =([6e4, 1.2e5], ["60k", "120k"])
        axs1[13].yticks =([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        axs1[14].yticks =([3e5, 6e5, 9e5], ["300k", "600k", "900k"])
        axs1[15].yticks =([3e5, 6e5, 9e5], ["300k", "600k", "900k"])
        for (idx, Ax) in enumerate(axs1[1:15])
            text!(Ax, spl_labels[idx], position = (0.05, 0.9), space = :relative)
        end
        hidedecorations!(axs1[16])
        hidespines!(axs1[16])
        display(fg1)
        CairoMakie.save("./user/figures/svg/Figure_S1b_Feature_distr_out.svg", fg1)
        # save("./user/figures/Figure_S1b_Feature_distr_out.png", fg1, px_per_unit = 4)

        # Input features
        spl_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        fg2 = Figure(size = (1200, 600))
        axs2 = [Axis(fg2[row, col], ylabel =L"$$No. pts", aspect = 1.681) for row in 1:3 for col in 1:4]
        for idx in axes(input_shuff, 1)
            inp = deepcopy(input_shuff[idx, :])
            idx > 2 ? inp .*= 100.0 : nothing
            sc = hist!(axs2[idx], inp)
        end
        axs2[1].xlabel = L"$T$ [°C]"
        axs2[2].xlabel = L"$P$ [kbar]"
        axs2[3].xlabel = L"SiO$_2^{\text{Sys}}$ [wt.%]"
        axs2[4].xlabel = L"Al$_2$O$_3^{\text{Sys}}$ [wt.%]"
        axs2[5].xlabel = L"CaO$^{\text{Sys}}$ [wt.%]"
        axs2[6].xlabel = L"MgO$^{\text{Sys}}$ [wt.%]"
        axs2[7].xlabel = L"FeO$^{\text{Sys}}$ [wt.%]"
        axs2[8].xlabel = L"(Na$_2$O + K$_2$O)$^{\text{Sys}}$ [wt.%]"
        axs2[9].xlabel = L"TiO$_2^{\text{Sys}}$ [wt.%]"
        axs2[10].xlabel = L"H$_2$O$^{\text{Sys}}$ [wt.%]"
        axs2[1].yticks = ([5e4, 1e5], ["50k", "100k"])
        axs2[2].yticks = ([5e4, 1e5], ["50k", "100k"])
        axs2[3].yticks = ([5e4, 1e5, 1.5e5], ["50k", "100k", "150k"])
        axs2[4].yticks = ([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        axs2[5].yticks = ([1e5, 2e5], ["100k", "200k"])
        axs2[6].yticks = ([1e5, 2e5], ["100k", "200k"])
        axs2[7].yticks = ([5e4, 1e5, 1.5e5], ["50k", "100k", "150k"])
        axs2[8].yticks = ([5e4, 1e5, 1.5e5], ["50k", "100k", "150k"])
        axs2[9].yticks = ([1e5, 2e5], ["100k", "200k"])
        axs2[10].yticks = ([1e5, 2e5, 3e5], ["100k", "200k", "300k"])
        for (idx, Ax) in enumerate(axs2[1:10])
            text!(Ax, spl_labels[idx], position = (0.05, 0.9), space = :relative)
        end
        hidedecorations!(axs2[11])
        hidespines!(axs2[11])
        hidedecorations!(axs2[12])
        hidespines!(axs2[12])
        display(fg2)
        CairoMakie.save("./user/figures/svg/Figure_S1a_Feature_distr_in.svg", fg2)
        # save("./user/figures/Figure_S1a_Feature_distr_in.png", fg2, px_per_unit = 4)
    end
    
    # Send to device
    dev     = cdev
    dev_str = "CPU"
    size(input_shuff, 2) > 100_000 ? (dev = gdev; dev_str = "GPU") : nothing
    @printf "Size of current data set: %7d \t Sending data to -> %s\n" size(input_shuff, 2) dev_str
    
    # Create user directory if not existing yet
    if !isdir(pdat_path)
        println("Creating directory: $pdat_path")
        mkpath(pdat_path)
    else
        println("Directory already exists: $pdat_path")
    end

    # Train the model
    for iHD in eachindex(hidden_dims)
        for iHL in eachindex(hidden_lays)
            for iBS in eachindex(batchsizes)
                bsize = batchsizes[iBS]
                hidden_lay = hidden_lays[iHL]
                hidden_dim = hidden_dims[iHD]
                @printf "Running bsize = %.5g \t HD = %5g \t HL = %.5g \t epochs = %5g \n" bsize hidden_dim hidden_lay epochs
                hyperparams = Dict("bsize" => bsize, "θ1"=> 0.0, "θ2" => 1.0) |> dev
                _ = MAGEMin_MLPs.train_model_crossval(input_shuff, output_ord, dev, rng; nEpochs = epochs, hidden_dims = hidden_dim, hidden_lays = hidden_lay, hyperparams = hyperparams, early_stop = 10, nprint = 20, nFolds = 5, save_model = true, fname_root = "./user/HyperParameterStudy_$(hidden_dim)HD_$(hidden_lay)HL_$(bsize)batchsize_$(epochs)epochs", log_transform = log_transform, DatType = DatType)
            end
        end
    end

    # Print success
    @printf "Successfully completed cross validation. Exiting now.\n"

    # Return
    return nothing
end

# Hint: do not set log_fluid = true in combination with Train_Igneous_v3.jl. Network architecture is incompatible.
TrainIgneous(; create_data = false, create_data_fig = true, data_file ="./data/surrogate_model/test_data.csv", log_transform = true, DatType = Float32);