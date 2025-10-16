# Load experimental data set and run numerical experiments and same conditions in MAGEMin using no buffer
using MAGEMin_C, DataFramesMeta, DelimitedFiles, CSV, JLD2

# Include utilities
include("thermodynamic_functionality.jl")

function compare_experiments_MAGEMin()
    # Set paths
    data_path    = "./data/lab_experiments/MineRS_MagmaDB_merged_v2_H2Ocalc.csv"
    data, header = readdlm(data_path, ';', header=true)
    df_exp       = DataFrame(data, vec(header))

    # Switches
    save_MAGE = true

    @show names(df_exp)

    # ---------------------------------------- #
    #| Preprocess the experimental data set   |#
    # ---------------------------------------- #

    # Remove empty lines in data frame
    idEmpty = findall(x -> x == "", df_exp[:, "Experiment"])
    deleteat!(df_exp, idEmpty)

    # Do not run with Cr2O3 = 0.0 - Use our GEOROC fit instead
    # idCr2O3 = findall(x -> x <= 5e-2, df_exp[:, "Cr2O3_start"])
    # df_exp[idCr2O3, "Cr2O3_start"] .= (-1.90613533e-08 .* df_exp[idCr2O3, "SiO2_start"] .^ 4 .+
    #                                   3.76177678e-06 .* df_exp[idCr2O3, "SiO2_start"] .^ 3   .-
    #                                   2.07004920e-04 .* df_exp[idCr2O3, "SiO2_start"] .^ 2   .-
    #                                   4.94661464e-04 .* df_exp[idCr2O3, "SiO2_start"]        .+
    #                                   2.18949178e-01)

    # Make sure oxygen is not specified in the experimental data set
    df_exp[:, "O_start"] .= 0.0

    # ---------------------------------------- #
    #|             MAGEMin call               |#
    # ---------------------------------------- #

    # Initialize variables
    db      = "ig"
    sys_in  = "wt"
    out_all = []

    # Loop through data set and run MAGEMin experiment
    for iExp in 1:size(df_exp)[1]

        # Convert iron oxides to total iron, if Fe2O3 is specified
        Xoxides = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "O", "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
        if df_exp[iExp, "Fe2O3_start"] > 0.0
            X_FeOt                    = convert_ironoxides_to_FeOt(df_exp[iExp, "Fe2O3_start"], df_exp[iExp, "FeO_start"], "wt")
            df_exp[iExp, "FeO_start"] = X_FeOt
        end
        
        # Extract relevant variables
        X      = [df_exp[iExp, "$(iOx)_start"] for iOx in Xoxides]
        buffer = string(df_exp[iExp, :fo2_buffer])
        B      = df_exp[iExp, :delta_buffer]
        T      = df_exp[iExp, :T_C]
        P      = df_exp[iExp, :P_kbar]


        # Initialize, run, and finalize MAGEMin
        if buffer !== "NONE"
            X_oversat = oversaturate_bulk(X, "O", 4.0, Xoxides)                                                         # Oversaturate in O
            data      = Initialize_MAGEMin(db, verbose=-1, buffer=buffer, solver=2)                                     # Initialize
            # rm_list   = remove_phases(["fl"], db)                                                                       # Remove phase that are not observed in experiments
            out       = single_point_minimization(P, T, data, X=X_oversat, B=B, Xoxides=Xoxides, sys_in=sys_in)         # Run minimization
            push!(out_all, out)                                                                                         # Store output
        else
            data = Initialize_MAGEMin(db, verbose=-1, solver=2)
            # rm_list = remove_phases(["fl"], db)
            out  = single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
            push!(out_all, out)
        end

        # Finalize
        Finalize_MAGEMin(data)
    end

    # Save MAGEMin structure to disk
    if save_MAGE
        save("./user/crystallization_experiments_MAGEMin_082025.jld2", "out_all", out_all)
    end

    # Return
    return out_all
end

# Run main
out = compare_experiments_MAGEMin()