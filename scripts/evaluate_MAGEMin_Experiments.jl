# Postprocessing MAGEMin vs Experiments comparison
using DataFramesMeta, DelimitedFiles, CSV, JLD2, CairoMakie, MAGEMin_C, ColorSchemes, MathTeXEngine, Random, Statistics, Lux, Printf
Makie.inline!(true)

set_theme!(theme_latexfonts())

# Include some functionality for Visualisation
include("./viz_functionality.jl")
include("./thermodynamic_functionality.jl")

# Define main
function postprocess_comparison()
    # -------------------------------------- #
    #|          Definitions                 |#
    # -------------------------------------- #

    # Set paths and load data
    data_path    = "./data/lab_experiments/MineRS_MagmaDB_merged_v2_H2Ocalc.csv"
    data, header = readdlm(data_path, ';', header=true)
    df_exp       = DataFrame(data, vec(header))
    out          = JLD2.load("./data/num_experiments/crystallization_experiments_MAGEMin_082025.jld2", "out_all")

    # Remove empty lines in data frame
    idEmpty = findall(x -> x == "", df_exp[:, "Experiment"])
    deleteat!(df_exp, idEmpty)

    # Pre-definitions
    start_oxides = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "O", "K2O", "Na2O", "Cr2O3", "TiO2", "H2O"]
    phase_exp    = ["ol", "opx", "cpx", "plag", "amph", "mag", "sp", "ilm", "bt", "ksp", "gt", "qz"]
    liq_oxides   = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "H2O"]
    remove_fluid = true

    # -------------------------------------- #
    #|          Data collection             |#
    # -------------------------------------- #

    # Postprocess MAGEMin structure
    df_MAGE = postprocess_MAGEMin_struct(out; remove_fluid = remove_fluid)

    # Add data from lab experiment to data frame
    for iOx in start_oxides                                         # ... starting oxides as named in original data set
        df_MAGE[:, "$(iOx)_exp_start"] .= df_exp[:, "$(iOx)_start"]
    end
    for iOx in liq_oxides                                           # ... liquid oxides as named in original data set
        df_MAGE[:, "$(iOx)_exp_liq"]   .= df_exp[:, "$(iOx)_liquid"]
    end
    for iPh in phase_exp                                            # ... mineral phases as named in original data set
        df_MAGE[:, "$(iPh)_exp"    ]   .= df_exp[:, iPh]
    end
    df_MAGE[:, "Exp_Type"           ]  .= df_exp[:, "Exp_Type"              ]   # ... citation
    df_MAGE[:, "Citation"           ]  .= df_exp[:, "Cit"                   ]   # ... citation
    df_MAGE[:, "Lab"                ]  .= df_exp[:, "Lab"                   ]   # ... citation
    df_MAGE[:, "Comments"           ]  .= df_exp[:, "Comments"              ]   # ... citation
    df_MAGE[:, "Duration_hrs"       ]  .= df_exp[:, "Duration_hrs"          ]   # ... citation
    df_MAGE[:, "Run"                ]  .= string.(df_exp[:, "Experiment"   ])   # ... run no. in the article
    df_MAGE[:, "Buffer"             ]  .= df_exp[:, "fo2_buffer"            ]   # ... buffer name
    df_MAGE[:, "dfO2"               ]  .= df_exp[:, "delta_buffer"          ]   # ... ΔfO₂
    df_MAGE[:, "density_exp_liq"    ]  .= df_exp[:, "liq_density_calculated"]   # ... calculated liquid density
    df_MAGE[:, "density_unc_exp_liq"]  .= df_exp[:, "liq_density_unc"       ]   # ... and its uncertainty
    df_MAGE[:, "density_exp_sol"    ]  .= df_exp[:, "sol_density"           ]   # ... solid density
    df_MAGE[:, "liq_exp"            ]  .= df_exp[:, "melt_fraction"         ]   # ... melt fraction

    # -------------------------------------- #
    #|           Visualization              |#
    # -------------------------------------- #

    # Switches
    fig_oxides   = false    # Figure 4
    fig_MFdens   = false    # Figure 7
    fig_BLcomp   = false    # Figure 6
    fig_Blatter6 = false     # Figure 14
    fig_expBias  = true    # Figure 1
    fig_bench    = false    # Figure 11
    fig_phcomp   = false    # Figure 5

    # Set plot options
    plt_opts_Oxides = makie_plot_options(; fig_size = (1100, 600), fig_res = 2, font_size = 15.0, line_width = 4.0, marker_size = 10.0, label_size = 16.0)
    plt_opts_MvsV   = makie_plot_options(; fig_size = (1000, 1000), fig_res = 2, font_size = 18.0, line_width = 4.0, marker_size = 14.0, label_size = 16.0)
    plt_opts_BLcomp = makie_plot_options(; fig_size = (1000, 800), fig_res = 2, font_size = 16.0, line_width = 4.0, marker_size = 14.0, label_size = 16.0, figure_pad = 25.0)
    plt_opts_BLF6   = makie_plot_options(; fig_size = (1241, 1754), fig_res = 2, font_size = 18.0, line_width = 4.0, marker_size = 14.0, label_size = 16.0)
    plt_opts_Bias   = makie_plot_options(; fig_size = (1241, 767),  fig_res = 2, font_size = 18.0, line_width = 4.0, marker_size = 14.0, label_size = 16.0)
    plt_opts_bench  = makie_plot_options(; fig_size = (1000, 552), fig_res = 2, font_size = 18.0, line_width = 2.0, marker_size = 18.0, label_size = 16.0, figure_pad = 25.0)
    plt_opts_phcomp = makie_plot_options(; fig_size = (1100, 552), fig_res = 2, font_size = 15.0, line_width = 4.0, marker_size = 10.0, label_size = 16.0)

    # Oxide composition
    fig_oxides ? create_composition_figure(df_MAGE, liq_oxides, plt_opts_Oxides) : nothing

    # Melt fraction and liquid density
    fig_MFdens ? create_MeltFractionDensity_figure(df_MAGE, plt_opts_MvsV) : nothing

    # Mineral abundance comparison (Blatter 2013 & 2023)
    fig_BLcomp ? create_BlatterComparison_figure(df_MAGE, plt_opts_BLcomp) : nothing

    # Experimental biases
    fig_expBias ? create_experimentBiases_figure(plt_opts_Bias) : nothing

    # Blatter 2013 figure 6
    fig_Blatter6 ? Blatter2013_temperature_vsOxide(df_MAGE, plt_opts_BLF6) : nothing

    # Benchmark
    fig_bench ? create_benchmark_figure(plt_opts_bench) : nothing

    # Mineral abundance comparison all
    fig_phcomp ? create_mineralComparison_figure(df_MAGE, plt_opts_phcomp) : nothing

    # Return
    return df_MAGE, out
end

# Define postprocessing functionality
function postprocess_MAGEMin_struct(Out_MAGEMin  :: AbstractArray;
                                    remove_fluid :: Bool = false
        )

    # ----------------------------- #
    #|         Initialize          |#
    # ----------------------------- #

    # Major oxides predicted by MAGEMin
    MAGEMin_ox = Out_MAGEMin[1].oxides

    # Variables that are present at each experiment
    df_out_db = DataFrame(  "P [kbar]" => Float64[],
                            "T [C]"    => Float64[],
                            "Liq. density [kg/m3]" => Float64[],
                            "Sol. density [kg/m3]" => Float64[],
    )

    # Solid and liquid composition columns
    for (_, iOx) in enumerate(MAGEMin_ox)
        df_out_db[!, "$(iOx)_liq [wt%]"]        = Float64[]
        df_out_db[!, "$(iOx)_sol [wt%]"]        = Float64[]
        df_out_db[!, "$(iOx)_MAGE_start [wt%]"] = Float64[]
    end

    # Create an exhaustive list of all occurring phases
    ph_obs = String[]
    for (_, out) in enumerate(Out_MAGEMin)
        ph_obs = vcat(ph_obs, out.ph)
    end
    ph_obs = unique(ph_obs)

    # Remove phases from list
    if remove_fluid
        idx_fl = findfirst(x -> x == "fl", ph_obs)
        isnothing(idx_fl) ? nothing : deleteat!(ph_obs, idx_fl)
    end

    # Stable phases
    for iPh in ph_obs
        df_out_db[!, "$(iPh) [wt%]"] .= Float64[]
    end

    # ----------------------------- #
    #|       Extract data          |#
    # ----------------------------- #

    for (idx, out) in enumerate(Out_MAGEMin)

        # Stable phases of current experiment
        target_phases    = deepcopy(out.ph)
        target_phases_wt = deepcopy(out.ph_frac_wt)
        bulk_wt          = deepcopy(out.bulk_wt)
        bulk_M_wt        = deepcopy(out.bulk_M_wt)
        bulk_S_wt        = deepcopy(out.bulk_S_wt)

        # Remove fluid phase if necessary
        if remove_fluid                                                                 # In case fluid has to be remove
            idx_fl   = findall(x -> x == "fl", target_phases)                       # ... find their indices in the out struct
            if isnothing(idx_fl)
                nothing
            else
                deleteat!(target_phases,    idx_fl)                                  # ... remove them from the target
                deleteat!(target_phases_wt, idx_fl)                                  # ... remove their weights too
                sum_ph_wt   = sum(target_phases_wt)                                   # ... calculate weight sum
                target_phases_wt ./= sum_ph_wt                                              # ... renormalize
                idx_liq = findfirst(x -> x == "liq", target_phases)
                if isnothing(idx_liq)
                    bulk_S_wt .= deepcopy(bulk_wt)
                elseif isapprox(target_phases_wt[idx_liq], 1.0, atol = 1e-3)
                    bulk_M_wt .= deepcopy(bulk_wt)
                else
                    liq_frac    = target_phases_wt[idx_liq]
                    bulk_S_wt .*= (1.0 .- liq_frac)
                    bulk_M_wt .*= (liq_frac)
                    bulk_wt    .= deepcopy(bulk_S_wt .+ bulk_M_wt)
                end
            end
        end

        for (_, iPh) in enumerate(ph_obs)                                               # Fill the data frame
            idPh = findfirst(x -> x == iPh, target_phases)
            if isnothing(idPh)
                push!(df_out_db[!, "$(iPh) [wt%]"], NaN)
            else
                push!(df_out_db[!, "$(iPh) [wt%]"], target_phases_wt[idPh] .* 100.0)
            end
        end

        # P, T, ρ
        push!(df_out_db[!, "P [kbar]"], out.P_kbar)
        push!(df_out_db[!, "T [C]"], out.T_C)
        push!(df_out_db[!, "Liq. density [kg/m3]"], out.rho_M)
        push!(df_out_db[!, "Sol. density [kg/m3]"], out.rho_S)

        # Composition
        for (idx_Ox, iOx) in enumerate(MAGEMin_ox)
            # Liquid
            if isnan(bulk_M_wt[idx_Ox])
                push!(df_out_db[!, "$(iOx)_liq [wt%]"], NaN)
            else
                push!(df_out_db[!, "$(iOx)_liq [wt%]"], bulk_M_wt[idx_Ox] .* 100.0)
            end

            # Solid
            if isnan(bulk_S_wt[idx_Ox])
                push!(df_out_db[!, "$(iOx)_sol [wt%]"], NaN)
            else
                push!(df_out_db[!, "$(iOx)_sol [wt%]"], bulk_S_wt[idx_Ox] .* 100.0)
            end

            # Start
            push!(df_out_db[!, "$(iOx)_MAGE_start [wt%]"], bulk_wt[idx_Ox] .* 100.0)
        end
    end

    # Return
    return df_out_db
end

# Composition plot
function create_composition_figure(df_MAGE :: DataFrame, liq_oxides :: Vector{String}, plt_opts :: makie_plot_options)

    # Ensure to compare only to experiments where liquid is reported
    df_sub = @chain df_MAGE begin
        @rsubset :SiO2_exp_liq > 1.0
        @rsubset :("SiO2_liq [wt%]") > 1.0
    end

    # Convert starting iron oxides to total iron
    for idx_exp in axes(df_sub, 1)
        X_FeOt = convert_ironoxides_to_FeOt(df_sub[idx_exp, "Fe2O3_exp_start"], df_sub[idx_exp, "FeO_exp_start"], "wt")
        df_sub[idx_exp, "FeO_exp_start"] = X_FeOt
    end

    # Calculate Variance of experiment
    pnames_exp  = ["ol_exp",   "opx_exp",   "cpx_exp",   "gt_exp",  "plag_exp",  "amph_exp", "sp_exp",    "ilm_exp",   "bt_exp",  "qz_exp",  "liq_exp"]
    pnames_MAGE = ["ol [wt%]", "opx [wt%]", "cpx [wt%]", "g [wt%]", "fsp [wt%]", "amp [wt%]", "spl [wt%]", "ilm [wt%]", "bi [wt%]","q [wt%]", "liq [wt%]"]
    
    Var_exp = Vector{Float64}(undef, size(df_sub, 1))
    for idx_Exp in axes(df_sub, 1)
        variance = 0
        for (_, Phase) in enumerate(pnames_exp)
            df_sub[idx_Exp, "$Phase"] > 0.0 ? variance += 1 : nothing
        end
        Var_exp[idx_Exp] = variance
    end

    # Calculate variance of the experiment
    Var_MAGE = Vector{Float64}(undef, size(df_sub, 1))
    for idx_MAGE in axes(df_sub, 1)
        variance = 0
        for (_, Phase) in enumerate(pnames_MAGE)
            df_sub[idx_MAGE, "$Phase"] > 0.0 ? variance += 1 : nothing
        end
        Var_MAGE[idx_MAGE] = variance
    end

    # Normalize bulk to anhydrous
    X_MAGE        = [[df_sub[idx_bulk, "$(iOx)_liq [wt%]"]  for iOx in liq_oxides] for idx_bulk in axes(df_sub, 1)]
    X_MAGE_anh, _ = normalize_to_anhydrous_bulk(X_MAGE, liq_oxides)
    X_exp        = [[df_sub[idx_bulk, "$(iOx)_exp_liq"]  for iOx in liq_oxides] for idx_bulk in axes(df_sub, 1)]
    X_exp_anh, _ = normalize_to_anhydrous_bulk(X_exp, liq_oxides)

    # Normalize starting compositions, too
    X_exp_start                 = [[df_sub[idx_bulk, "$(iOx)_exp_start"]  for iOx in liq_oxides] for idx_bulk in axes(df_sub, 1)]
    X_exp_start_anh, oxides_anh = normalize_to_anhydrous_bulk(X_exp_start, liq_oxides)
    X_MAGE_start                = [[df_sub[idx_bulk, "$(iOx)_MAGE_start [wt%]"]  for iOx in liq_oxides] for idx_bulk in axes(df_sub, 1)]
    X_MAGE_start_anh, _         = normalize_to_anhydrous_bulk(X_MAGE_start, liq_oxides)

    # Set coloring options 
    color_by_H2Ostart = false
    color_by_Varexp   = false
    color_by_VarMAGE  = false
    color_by_meltfrac = true

    # Adjust labels according to coloring options
    coloring = []
    cb_label = String[]
    if color_by_H2Ostart
        coloring = vec(Float64.(df_sub[:, "H2O_exp_start"]))
        cb_label = "H2O start [wt%]"
        cmp = :blues
    elseif color_by_Varexp
        coloring = Var_exp
        cb_label = "Variance Exp. []"
        cmp = :greens
    elseif color_by_VarMAGE
        coloring = Var_MAGE
        cb_label = "Variance MAGEMin []"
        cmp = :greens
    elseif color_by_meltfrac
        coloring = vec(Float64.(df_sub[:, "liq_exp"]))
        cb_label = L"$\Phi$ Exp. [wt-%]"
        cmp = :reds
    end

    # Layout
    nax  = length(oxides_anh)       # Set no. axis
    ncol = 4                        # Set no. columns
    nrow = cld(nax, ncol)           # Calculate no. rows
    ox_labels = [L"SiO$_2$", L"Al$_2$O$_3$", L"$$CaO", L"$$MgO", L"$$FeO", L"K$_2$O", L"Na$_2$O", L"TiO$_2$"]
    # Initialize
    fg1 = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    GL  = [fg1[row, col][1,1] = GridLayout() for row in 1:nrow, col in 1:ncol]
    # Fill figure
    axs = []
    scs = []
    for (idx_Ox, Oxide) in enumerate(oxides_anh)
        X_MAGE_liq   = getindex.(X_MAGE_anh, idx_Ox)
        X_exp_liq    = getindex.(X_exp_anh, idx_Ox)
        X_exp_start  = getindex.(X_exp_start_anh, idx_Ox)
        X_MAGE_start = getindex.(X_MAGE_start_anh, idx_Ox)
        xplot = X_exp_liq  .- X_exp_start
        yplot = X_MAGE_liq .- X_MAGE_start
        ax = Axis(GL[idx_Ox][1,1], aspect = 1.0, ylabel = L"$Δ$ MAGE [wt-%]", xlabel = L"$Δ$ Exp. [wt-%]", ytickformat = "{:.1f}", xtickformat = "{:.1f}")
        push!(axs, ax)
        ln = lines!(axs[idx_Ox], minimum(yplot):1:maximum(yplot), minimum(yplot):1:maximum(yplot), color = :goldenrod1, linewidth = plt_opts.line_width)
        sc = scatter!(axs[idx_Ox], xplot , yplot, color = coloring, colormap = cmp, label = ox_labels[idx_Ox], markersize = plt_opts.marker_size)
        axislegend(axs[idx_Ox], position = :rb, framevisible = false, labelsize = 11, backgroundcolor = (:white, 0.0))
        text!(  axs[idx_Ox], 0, 1, 
                text = string(plt_opts.spl_alpha_small[idx_Ox]), 
                space = :relative, align = (:left, :top), 
                offset = (4, -2), 
                font = :bold, 
                fontsize = plt_opts.font_size
        )
        push!(scs, sc)

        # Adjust ticks
        if Oxide == "SiO2"
            ax.limits = (-5.0, 25.0, -5.0, 25.0)
            ax.xticks = [0.0, 10.0, 20.0]
            ax.yticks = [0.0, 10.0, 20.0]
            ax.xlabelvisible = false
        elseif Oxide == "CaO"
            ax.limits = (-8.0, 4.0, -8.0, 4.0)
            ax.xticks = [-8.0, -4.0, 0.0, 4.0]
            ax.yticks = [-8.0, -4.0, 0.0, 4.0]
            ax.xlabelvisible = false
            ax.ylabelvisible = false
        elseif Oxide == "FeO"
            ax.limits = (-10.0, 10.0, -10.0, 10.0)
            ax.xticks = [-5.0, 0.0, 5.0]
            ax.yticks = [-5.0, 0.0, 5.0]
            ax.xlabelvisible = false
            ax.ylabelvisible = false
        elseif Oxide == "Na2O"
            ax.limits = (-2.0, 4.0, -2.0, 4.0)
            ax.xticks = [-2.0, 0.0, 2.0, 4.0]
            ax.yticks = [-2.0, 0.0, 2.0, 4.0]
            ax.xlabelvisible = false
            ax.ylabelvisible = false
        elseif Oxide == "Al2O3"
            ax.limits = (-10.0, 10.0, -10.0, 10.0)
            ax.xticks = [-5.0, 0.0, 5.0]
            ax.yticks = [-5.0, 0.0, 5.0]
        elseif Oxide == "MgO"
            ax.limits = (-20.0, 2.0, -20.0, 2.0)
            ax.xticks = [-15.0, -10.0, -5.0, 0.0]
            ax.yticks = [-15.0, -10.0, -5.0, 0.0]
            ax.ylabelvisible = false
        elseif Oxide == "K2O"
            ax.limits = (-1.0, 5.0, -1.0, 5.0)
            ax.xticks = [0.0, 2.5, 5.0]
            ax.yticks = [0.0, 2.5, 5.0]
            ax.ylabelvisible = false
        elseif Oxide == "TiO2"
            ax.limits = (-1.5, 3.0, -1.5, 3.0)
            ax.xticks = [-1.5, 0.0, 1.5, 3.0]
            ax.yticks = [-1.5, 0.0, 1.5, 3.0]
            ax.ylabelvisible = false
        end
    end

    # Add colorbar
    Colorbar(fg1[3, 1:2], scs[1], vertical = false, label = cb_label, flip_vertical_label = true, flipaxis = false)

    # Align labels
    yspace = maximum(tight_yticklabel_spacing!, axs)
    xspace = maximum(tight_xticklabel_spacing!, axs)
    for Axs in axs
        Axs.yticklabelspace = yspace + plt_opts.tick_label_pad
        Axs.xticklabelspace = xspace + plt_opts.tick_label_pad
    end

    # Print figure
    display(fg1)
    save("./figures/Figure_4_oxide_comparison.png", fg1, px_per_unit = plt_opts.fig_res)

    # Return
    return nothing
end

# Melt fraction and Density
function create_MeltFractionDensity_figure(df_MAGE :: DataFrame, plt_opts :: makie_plot_options)
    # Sort for experiment
    df_sub = @chain df_MAGE begin
        @rsubset :SiO2_exp_liq > 1.0
    end

    idMU2019 = findall(df_sub[:, "Citation"] .== "MarxerUlmer2019")
    idVi2004 = findall((df_sub[:, "Citation"] .== "Villiger2004" .&& df_sub[:, "Exp_Type"] .== "FC"))
    idVi2004hy = findall(df_sub[:, "Citation"] .== "Villiger2004hy")

    # Identify supra solidus experiments
    idhasExpD = findall(df_sub[:, "density_exp_liq"] .> 0.0)
    idhasLiqM = findall(df_sub[idhasExpD, "Liq. density [kg/m3]"] .> 0.0)

    # Initialize figure
    fg1    = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    GL_MF1 = fg1[1, 1] = GridLayout()
    GL_MF2 = fg1[1, 2] = GridLayout()
    GL_DE  = fg1[2, 1] = GridLayout()
    GL_LE  = fg1[2, 2] = GridLayout()
    ax1 = Axis( GL_MF1[1,1], 
                xlabel      = L"$T$ [°C]",
                ylabel      = L"$\Phi$ [%]",
                limits      = (680.0, 1010.0, 0.0, 100.0),
                xticks      = [i for i in 700:100:1000],
                title       = "MU 2019 (hydrous)",
                ytickformat = "{:.1f}",
                xtickformat = "{:.1f}",
                aspect      = plt_opts.golden_ratio
                )
    ax2 = Axis( GL_MF2[1,1], 
                xlabel      = L"$T$ [°C]",
                ylabel      = L"$\Phi$ [%]",
                limits      = (1020.0, 1320.0, 0.0, 100.0),
                xticks      = [i for i in 1000:100:1300],
                title       = "Vi 2004 (anhydrous)",
                ytickformat = "{:.1f}",
                xtickformat = "{:.1f}",
                aspect      = plt_opts.golden_ratio
                )
    ax3 = Axis( GL_DE[1,1], 
                xlabel = L"$\rho$ Exp. liq. [kg.m$^{-3}$]", 
                ylabel = L"$\rho$ MAGE liq. [kg.m$^{-3}$]", 
                aspect = 1.0,
                limits = (2000.0, 3000.0, 2000.0, 3000.0),
                ytickformat = "{:.1f}",
                xtickformat = "{:.1f}"
                )

    # Replace NaN in subsolidus conditions
    idNan = findall(x -> isnan(x), df_sub[:, "liq [wt%]"])
    df_sub[idNan, "liq [wt%]"] .= 0.0

    # Fill figure
    sc1 = scatter!(ax1, df_sub[idMU2019, "T [C]"], df_sub[idMU2019, "liq_exp"],   marker = :circle, markersize = plt_opts.marker_size, label = "Exp.", color = :skyblue)
    ln1 = lines!(ax1, df_sub[idMU2019, "T [C]"], df_sub[idMU2019, "liq_exp"], color = :skyblue)
    sc2 = scatter!(ax1, df_sub[idMU2019, "T [C]"], df_sub[idMU2019, "liq [wt%]"], marker = :utriangle, markersize = plt_opts.marker_size, label = "MAGE", color = :skyblue4)
    ln2 = lines!(ax1, df_sub[idMU2019, "T [C]"], df_sub[idMU2019, "liq [wt%]"], color = :skyblue4)
    sc3 = scatter!(ax2, df_sub[idVi2004, "T [C]"], df_sub[idVi2004, "liq_exp"],   marker = :circle, markersize = plt_opts.marker_size, color = :skyblue)
    ln3 = lines!(ax2, df_sub[idVi2004, "T [C]"], df_sub[idVi2004, "liq_exp"], color = :skyblue)
    sc4 = scatter!(ax2, df_sub[idVi2004, "T [C]"], df_sub[idVi2004, "liq [wt%]"], marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln4 = lines!(ax2, df_sub[idVi2004, "T [C]"], df_sub[idVi2004, "liq [wt%]"], color = :skyblue4, label = "0.0 wt-% H2O")
    sc4 = scatter!(ax2, df_sub[idVi2004hy, "T [C]"], df_sub[idVi2004hy, "liq [wt%]"], marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln4 = lines!(ax2, df_sub[idVi2004hy, "T [C]"], df_sub[idVi2004hy, "liq [wt%]"], color = :skyblue4, linestyle = :dash, label = "0.5 wt-% H2O")
    eb1 = errorbars!(ax3, df_sub[idhasLiqM, "density_exp_liq"], df_sub[idhasLiqM, "Liq. density [kg/m3]"], df_MAGE[idhasLiqM, "density_unc_exp_liq"], direction = :x, color = vec(Float64.(df_sub[idhasLiqM, "H2O_exp_start"])), colormap = :blues) 
    sc5 = scatter!( ax3, df_sub[idhasLiqM, "density_exp_liq"], df_sub[idhasLiqM, "Liq. density [kg/m3]"], 
                    color      = vec(Float64.(df_sub[idhasLiqM, "H2O_exp_start"])), 
                    colormap   = :blues, 
                    markersize = plt_opts.marker_size
                    )
    ln1 = lines!(ax3, minimum(df_sub[idhasLiqM, "density_exp_liq"]):maximum(df_sub[idhasLiqM, "density_exp_liq"]),minimum(df_sub[idhasLiqM, "density_exp_liq"]):maximum(df_sub[idhasLiqM, "density_exp_liq"]), color = :goldenrod1, linewidth = plt_opts.line_width)
    axislegend(ax1, framecolor = (:gray, 0.5), position = :rb, labelsize = 11)
    axislegend(ax2, framecolor = (:gray, 0.5), position = :rb, backgroundcolor = (:white, 0.1), labelsize = 11)

    # Add colorbar
    ax4 = Axis(GL_LE[1,1][1,2], aspect = 1.0)
    hidedecorations!(ax4)
    hidespines!(ax4)
    Colorbar(GL_LE[1,1][1,1], sc5, vertical = true, label = L"H$_{2}$O start [wt-%]", labelsize = plt_opts.label_size)

    # Align labels
    yspace = maximum(tight_yticklabel_spacing!, [ax1, ax2, ax3])
    xspace = maximum(tight_xticklabel_spacing!, [ax1, ax2, ax3])
    ax1.yticklabelspace = yspace + plt_opts.tick_label_pad
    ax2.yticklabelspace = yspace + plt_opts.tick_label_pad
    ax3.yticklabelspace = yspace + plt_opts.tick_label_pad
    ax1.xticklabelspace = xspace + plt_opts.tick_label_pad
    ax2.xticklabelspace = xspace + plt_opts.tick_label_pad
    ax3.xticklabelspace = xspace + plt_opts.tick_label_pad

    # Add subplot label
    axs = [ax1, ax2, ax3]
    for (idx_ax, Ax) in enumerate(axs)
        text!(  Ax, 0, 1, 
        text = string(plt_opts.spl_alpha_small[idx_ax]), 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    end

    # Print figure
    display(fg1)
    save("./figures/Figure_7_MUVi_meltfrac_dens.png", fg1, px_per_unit = plt_opts.fig_res)

    # Return
    return nothing
end

# Phase comparison
function create_BlatterComparison_figure(df :: DataFrame, plt_opts :: makie_plot_options)
    
    # Get the blatter experiments and extract information
    df_B13 = @chain df begin
        @rsubset :Citation == "Blatter2013" || :Citation == "Blatter2023"
    end

    # Load phase diagram data
    # data, header = readdlm("./refined/Blatter2013_new_ig.csv", ',', header = true)
    # df_PD        = DataFrame(data, vec(header))

    # # Get the stable phase for each minimization
    # idSystem = findall(x -> x .== "system", df_PD[:, "phase"])
    # deleteat!(df_PD, idSystem)
    # npts       = length(unique(df_PD[:, "point[#]"]))                   # No. of minimizations

    # ph_vec     = Vector{Vector{String}}(undef, npts)                    # Initialize vector that contains stable phase at each minimization point
    # T_vec      = Vector{Float64}(undef, npts)                           # Initialize vector that contains T at each minimization point
    # P_vec      = Vector{Float64}(undef, npts)                           # Initialize vector that contains P at each minimization point
    # field_vec  = Vector{Float64}(undef, npts)                           # Initialize vector that contains the field index to which the minimization point belongs
    # var_vec    = Vector{Float64}(undef, npts)                           # Initialize vector that contains the field index to which the minimization point belongs
    
    # Threads.@threads for iPt in eachindex(ph_vec)                       # Loop through each minimization point ...
    #     idPts = findall(x -> x .== iPt, df_PD[:, "point[#]"])           # ... get all data entries for current point
    #     # deleteat!(idPts, 1)                                             # ... remove system info from detected indices
    #     ph_vec[iPt] = String.(df_PD[idPts, "phase"])                    # ... collect stable phases at each point
    #     T_vec[iPt]  = reshape(unique(df_PD[idPts, "T[°C]"  ]), 1)[1]                    # ... collect stable phases at each point
    #     P_vec[iPt]  = reshape(unique(df_PD[idPts, "P[kbar]"]), 1)[1]                    # ... collect stable phases at each point
    # end

    # # Get fields
    # sort!.(ph_vec)                                                      # Sort the phase strings at each minimization point for comparison
    # ph_fields = unique(ph_vec)                                        # Initialize container fields
    # nflds     = length(ph_fields)                                     # No. of fields in diagram

    # for (idx, field) in enumerate(ph_fields)                            # Loop through phase fields ...
    #     idPts  = findall(x -> x  == field, ph_vec)                      # ... find all minimization points that belong to the current field
    #     field_vec[idPts] .= idx
    #     var_vec[idPts] .= 11 - length(field) + 2
    #     # if idx == 16
    #     #     println(idPts)
    #     # end
    # end

    # println(unique(field_vec))
    # return

    # Interpolate data
    # Trange                 = (900.0, 1400.0)
    # Prange                 = (3.0, 18.0)
    # num_ini_lev            = 3
    # num_ref_lev            = 4
    # npts                   = 2^(num_ini_lev + num_ref_lev)
    # var_2d, T_plot, P_plot = interp_scattered_2D(T_vec, P_vec, var_vec, Trange, Prange, npts, "MAGEMin")
    # field_2d, _, _         = interp_scattered_2D(T_vec, P_vec, field_vec, Trange, Prange, npts, "MAGEMin")

    # T2D = repeat(T_plot , npts) |> x -> reshape(x, size(var_2d))
    # P2D = repeat(P_plot', npts) |> x -> reshape(x, size(var_2d))

    # Initialize figure
    fg1    = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    # GL_PD  = fg1[1,1] = GridLayout()                                                               # Layout for phase diagram
    # ax_PD  = Axis(GL_PD[1,1][1,1], xlabel = L"$T$ [°C]", ylabel = L"$P$ [kbar]", xtickformat = "{:.1f}", ytickformat = "{:.1f}")            # Axis for phase diagram
    GL_ph  = fg1[1, 1:2][1,1] = GridLayout()                                                            # Layout for mineral abundance comparison
    axs_ph = [Axis(GL_ph[i, j], xtickformat = "{:.1f}", ytickformat = "{:.1f}", aspect = 1.0) for i in 1:2, j in 1:3]                                             # Axis for mineral abundance comparison
    
    # Visualize phase diagram
    # hm1 = heatmap!(ax_PD, T_plot, P_plot, var_2d, colormap = Reverse(:blues), colorrange = (minimum(var_2d), maximum(var_2d)), nan_color = :magenta)
    # cl1 = contour!(ax_PD, T_plot, P_plot, field_2d, color = :black, linewidth = 1.0, levels = (1:nflds), linestyle = :dash)
    
    # fg2 = Figure()
    # ax1 = Axis(fg2[1,1][1,1])
    # hm1 = heatmap!(ax1, field_2d)
    # Colorbar(fg2[1,1][1,2], hm1)
    # display(fg2)

    # # Generate diagram labels
    # mask        = zeros(Int64, size(field_2d))                  # This mask is used to identify only the currently searched phase field
    # leg_str     = Vector{String}(undef, size(ph_fields)[1])     # Initialize legend string
    # lost_fields = []
    # counter = 0
    # for (idx, field) in enumerate(ph_fields)                # Loop through fields ...
    #     mask .= 0                                           # ... maks all inactive to 0
    #     mask[field_2d .== idx] .= 1                         # ... except for the current
    #     idx_field = findall(mask .== 1)                     # ... identify all indices that belong to current field

    #     # Small fields might have been lost during interpolation ... flag and remove them from the list
    #     if isempty(idx_field)
    #         push!(lost_fields, idx)
    #         continue
    #     else
    #         counter += 1                                        # Recovered field no.
    #         idx_field_min = minimum(getindex.(idx_field, 1))    # ... their minimum and maximum values
    #         idx_field_max = maximum(getindex.(idx_field, 1))
    #         idy_field_min = minimum(getindex.(idx_field, 2))
    #         idy_field_max = maximum(getindex.(idx_field, 2))
    #         idx_text_annot = cld((idx_field_max - idx_field_min), 2) + idx_field_min        # ... calculate a useful position in the field to place the annotation
    #         idy_text_annot = cld((idy_field_max - idy_field_min), 2) + idy_field_min
    #         # A_field = sum(mask .* (T_plot[2] - T_plot[1]) .* (P_plot[2] - P_plot[1])) / ( (Trange[2] - Trange[1]) * (Prange[2] - Prange[1]) ) * 100.0

    #         text_str = "$(counter)"                                 # ... create annotation string
    #         txt = text!(ax_PD, T2D[idx_text_annot, idy_text_annot], P2D[idx_text_annot, idy_text_annot], text = text_str, color = :white, align = (:center, :center), justification = :center, fontsize = 13)                                                   # ... set annotation

    #         ph_str = replace("$(ph_fields[idx])", "[" => "") |> x -> replace(x, "]" => "") |> x -> replace(x, "\"" => "")       # ... strip stable phase string for the legend
    #         leg_str[idx] = "$(counter) = $ph_str\n"                                                                                 # ... store it for later use
    #     end
    # end

    # # Remove small fields that could be interpolated
    # deleteat!(leg_str, lost_fields)

    # # Add stable phases list
    # nbanks    = 2
    # bank_marg = 0.00
    # wcol      = 0.42 / nbanks + (nbanks - 1) * bank_marg
    # idx_partition = cld(length(leg_str), nbanks)
    # for idx in 1:nbanks
    #     idx_start = 1 + (idx - 1) * idx_partition
    #     idx_end   = idx * idx_partition
    #     idx == nbanks ? idx_end = length(leg_str) : nothing
    #     text!(fg1.scene, (0.57 + (idx - 1)*wcol, 0.96), space = :relative, text = join(leg_str[idx_start:idx_end]), word_wrap_width = 250, align = (:left, :top), fontsize = plt_opts.label_size)        # Create phase legend
    # end
    # text!(fg1.scene, (0.57, 0.99), space = :relative, text = "Stable phases", align = (:left, :top), font = :bold)
    # cb = Colorbar(GL_PD[1,1][1,2], hm1, label = L"$Va$ [ ]")                                                                                           # Add colorbar to phase diagram

    # Visualize phase comparison
    P_Bl13  = [4.0, 9.0, 16.7]                                                                       # Set pressure and phase names
    pnames_exp  = ["ol_exp",   "opx_exp",   "cpx_exp",   "gt_exp",  "plag_exp",  "amph_exp", "sp_exp",    "ilm_exp",   "bt_exp",  "qz_exp",  "liq_exp"]
    pnames_MAGE = ["ol [wt%]", "opx [wt%]", "cpx [wt%]", "g [wt%]", "fsp [wt%]", "amp [wt%]", "spl [wt%]", "ilm [wt%]", "bi [wt%]","q [wt%]", "liq [wt%]"]
    ph_coloring = cgrad(:bam, length(pnames_exp), categorical = true)                               # Generate color palette for different phases
    count = 0                                                                                       # Counter to plot experiment and MAGEMin simultaneously into their corresponding axis
    normalize_exp = true                                                                            # Normalize the exp. abundance when phase not predicted by MAGEMin have to be excluded
    for (idx, pressure) in enumerate(P_Bl13)                                                        # Loop through three different pressures
        df_curr = @rsubset df_B13 :"P [kbar]" == pressure                                           # Get data at current pressure
        baseline = []                                                                               # Initialize baseline for the area plots
        plot_first = true                                                                           # Switch between first and all subsequent plots
        if normalize_exp
            df_curr[:, "Total"] .= 0.0
            for (_, phase) in enumerate(pnames_exp)
                for (row, abundance) in enumerate(df_curr[:, "$phase"])
                    isnan(abundance) ? df_curr[row, "Total"] += 0.0 : df_curr[row, "Total"] += abundance
                end
            end
        end
        for (idx_ph, phase) in enumerate(pnames_exp)                                                # Loop through phases in the experiment ...
            if ~isempty(df_curr[:, "$phase"])                                                       # ... if current phase is present
                if plot_first
                    phase_abund = df_curr[:, "$phase"]                                              # ... get current abundance
                    phase_abund[phase_abund .== NaN] .= 0.0                                         # ... remove NaNs
                    normalize_exp ? phase_abund ./= df_curr[:, "Total"] ./ 100.0 : nothing          # ... normalize if required (attention you have to divide by 100.0 to get %)
                    band!(axs_ph[idx + count], df_curr[:, "T [C]"], 0.0 .* df_curr[:, "T [C]"], phase_abund , color = ph_coloring[idx_ph])      # Plot area
                    plot_first = false                                                              # ... switch of baseline = 0.0
                    baseline = phase_abund                                                          # ... store new baseline
                else
                    phase_abund = df_curr[:, "$phase"]                                              # See last comments above
                    phase_abund[phase_abund .== NaN] .= 0.0
                    normalize_exp ? phase_abund ./= df_curr[:, "Total"] ./ 100.0 : nothing
                    band!(axs_ph[idx + count], df_curr[:, "T [C]"], baseline, baseline .+ phase_abund, color = ph_coloring[idx_ph])
                    baseline .+= phase_abund                                                        # ... update baseline
                end
            end
        end
        xlims!(axs_ph[idx + count], low = minimum(df_curr[:, "T [C]"]), high = maximum(df_curr[:, "T [C]"]))
        ylims!(axs_ph[idx + count], low = 0.0, high = 100.0)
        txt_str = "Exp. @ $(pressure) kbar"                                                            # Add label to phase plots
        text!(axs_ph[idx + count], maximum(df_curr[:, "T [C]"]) - 10.0, 90.0, text = txt_str, color = :white, align = (:right, :baseline), justification = :right)
        baseline   = []                                                                             # Reset baseline 
        plot_first = true                                                                           # Reset switch
        count += 1                                                                                  # Update counter to now plot in the MAGEMin axis
        for (idx_ph, phase) in enumerate(pnames_MAGE)                                                   # AS ABOVE just now we loop through MAGEMin data
            if ~isempty(df_curr[:, "$phase"])
                if plot_first
                    phase_abund = df_curr[:, "$phase"]
                    phase_abund[isnan.(phase_abund)] .= 0.0
                    band!(axs_ph[idx + count], df_curr[:, "T [C]"], 0.0 .* df_curr[:, "T [C]"], phase_abund, color = ph_coloring[idx_ph])
                    plot_first = false
                    baseline = phase_abund
                else
                    phase_abund = df_curr[:, "$phase"]
                    phase_abund[isnan.(phase_abund)] .= 0.0
                    band!(axs_ph[idx + count], df_curr[:, "T [C]"], baseline, baseline .+ phase_abund, color = ph_coloring[idx_ph])
                    baseline .+= phase_abund
                end
            end
        end
        xlims!(axs_ph[idx + count], low = minimum(df_curr[:, "T [C]"]), high = maximum(df_curr[:, "T [C]"]))
        ylims!(axs_ph[idx + count], low = 0.0, high = 100.0)
        txt_str = "MAGE @ $(pressure) kbar"                                                                        # Add label to phase plots
        text!(axs_ph[idx + count], maximum(df_curr[:, "T [C]"]) - 10.0, 90.0, text = txt_str, color = :white, align = (:right, :baseline), justification = :right)
    end

    # Add legend
    leg_str_ph = [replace("$(phase)", "_exp" => "") for phase in pnames_exp]
    leg_rects  = [PolyElement(color = color, strokecolor = :black, strokewidth = 1.0) for color in ph_coloring]
    Legend(fg1[1,1:2][2,1], leg_rects, leg_str_ph, orientation = :horizontal, framevisible = false, nbanks = 2, labelsize = plt_opts.label_size, halign = :left, valign = :top)
    
    # Add axis labels
    axs_ph[1].ylabel = L"$$Abundance [wt-%]"
    axs_ph[2].ylabel = L"$$Abundance [wt-%]"
    axs_ph[2].xlabel = L"$T$ [°C]"
    axs_ph[4].xlabel = L"$T$ [°C]"
    axs_ph[6].xlabel = L"$T$ [°C]"

    # Align labels
    # yspace = maximum(tight_yticklabel_spacing!, [ax_PD, axs_ph[1]])
    # xspace = maximum(tight_xticklabel_spacing!, [ax_PD, axs_ph[1]])
    # for Axs in axs_ph
    #     Axs.xticklabelspace = xspace + plt_opts.tick_label_pad
    #     Axs.yticklabelspace = yspace + plt_opts.tick_label_pad
    # end
    # ax_PD.xticklabelspace = xspace + plt_opts.tick_label_pad
    # ax_PD.yticklabelspace = yspace + plt_opts.tick_label_pad

    # Add subplot labels
    # text!(  ax_PD, 0, 1, 
    #         text = string(plt_opts.spl_alpha_small[1]), 
    #         space = :relative, align = (:left, :top), 
    #         offset = (4, -2), 
    #         font = :bold, 
    #         fontsize = plt_opts.font_size,
    #         color = :white
    # )
    for (idx_ax, Ax) in enumerate(axs_ph)
        text!(  Ax, 0, 1, 
        text = string(plt_opts.spl_alpha_small[idx_ax]), 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size,
        color = :white
        )
    end
    # Print figure
    display(fg1)
    save("./figures/Figure_6_Blatter_abundance_comparison.png", fg1, px_per_unit = plt_opts.fig_res)

    # Return
    return nothing
end

# Blatter2013's fig 6
function Blatter2013_temperature_vsOxide(df :: DataFrame, plt_opts :: makie_plot_options)
    
    # Get data
    df_BL = @chain df begin
        @rsubset :Citation == "Blatter2013" || :Citation == "Blatter2023"
        @rsubset :"P [kbar]" == 4.0 || :"P [kbar]" == 9.0 || :"P [kbar]" == 16.7
    end
    marker_symbols = [:circle, :rect, :utriangle, :dtriangle, :ltriangle, :rtriangle, :diamond, :hexagon, :cross, :xcross]
    markers        = Vector{Symbol}(undef, size(df_BL, 1))
    pressures      = sort(unique(df_BL[:, "P [kbar]"]))
    for idx_ExP in axes(df_BL, 1)
        idP = findfirst(x -> x == df_BL[idx_ExP, "P [kbar]"], pressures)
        markers[idx_ExP] = marker_symbols[idP]
    end
    # Set oxides
    oxides = ["SiO2", "CaO", "Al2O3", "MgO", "K2O", "FeO", "Na2O", "TiO2"]

    # Initialize figure
    fg1   = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    nrows = 4
    ncols = 2
    GL    = [fg1[row, col] = GridLayout() for row in 1:nrows, col in 1:ncols]
    axs = []

    # Loop over pressures
    Ps = [4.0, 9.0, 16.7]
    markers = [:rect, :circle, :utriangle]
    for (idx_P, P) in enumerate(Ps)
        df_BLP = @chain df_BL begin
            @rsubset :"P [kbar]" == P
        end
        # Get experimental and MAGEMin bulk of oxides and normalize
        X_exp    = [[df_BLP[idx_Exp, "$(Oxide)_exp_liq"] for Oxide in oxides] for idx_Exp in axes(df_BLP, 1)]
        X_MAGE   = [[df_BLP[idx_Exp, "$(Oxide)_liq [wt%]"] for Oxide in oxides] for idx_Exp in axes(df_BLP, 1)]
        X_exp_n  = normalize_bulk(X_exp)
        X_MAGE_n = normalize_bulk(X_MAGE)
        # Fill figure
        for (idx_Ox, Oxide) in enumerate(oxides)
            x_plt_exp = getindex.(X_exp_n, idx_Ox)
            x_plt_MAGE = getindex.(X_MAGE_n, idx_Ox)
            if idx_P == 1
                ax = Axis(GL[idx_Ox][1,1], aspect = plt_opts.golden_ratio, title = "$Oxide")
                push!(axs, ax)
            end
            scatter!(axs[idx_Ox], df_BLP[:, "T [C]"], x_plt_exp, color = (:steelblue, 0.5), marker = markers[idx_P], markersize = plt_opts.marker_size)
            lines!(axs[idx_Ox], df_BLP[:, "T [C]"], x_plt_exp, color = (:steelblue, 0.5), linestyle = :solid, label = "Exp.", linewidth = plt_opts.line_width)
            scatter!(axs[idx_Ox], df_BLP[:, "T [C]"], x_plt_MAGE, color = (:goldenrod1, 1.0), marker = markers[idx_P], markersize = plt_opts.marker_size)
            lines!(axs[idx_Ox], df_BLP[:, "T [C]"], x_plt_MAGE, color = (:goldenrod1, 1.0), linestyle = :dash, linewidth = plt_opts.line_width, label = "MAGE")

            # Hide decorations
            if idx_Ox == 4 || idx_Ox == 8
                axs[idx_Ox].xlabel = L"$T$ [°C]"
            else
                axs[idx_Ox].xticklabelsvisible = false
            end
            axs[idx_Ox].ytickformat = "{:.1f}"
            axs[idx_Ox].ylabel = L"$$Abundance [wt-%]"
        end
    end

    # Add legends
    axislegend( axs[1], "Color", 
                labelsize    = plt_opts.label_size, 
                titlesize    = plt_opts.label_size + 1.0, 
                titlehalign  = :left, 
                framevisible = false,
                unique = true
                )
    Legend( fg1[end+1,1], 
            [MarkerElement(color = :gray, marker = marker_symbols[iP]) for iP in eachindex(pressures)], ["$(iP) [kbar]" for iP in pressures], 
            "Symbol",
            orientation  = :horizontal,
            labelsize    = plt_opts.label_size, 
            halign       = :left, 
            titlesize    = plt_opts.label_size + 1.0, 
            titlehalign  = :left, 
            framevisible = false, 
            nbanks       = 1
            )

    # Align labels
    yspace = maximum(tight_yticklabel_spacing!, axs)
    xspace = maximum(tight_xticklabel_spacing!, axs)
    for Axs in axs
        Axs.yticklabelspace = yspace + plt_opts.tick_label_pad
        Axs.xticklabelspace = xspace + plt_opts.tick_label_pad
    end

    # Add subplot labels
    for (idx_ax, Ax) in enumerate(axs)
        text!(  Ax, 0, 1, 
        text = string(plt_opts.spl_alpha_small[idx_ax]), 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    end
    # Print figure
    display(fg1)
    save("./figures/Figure_14_Blatter_liquidline.png", fg1, px_per_unit = plt_opts.fig_res)

    # Return
    return nothing
end

# Experimental biases plot
function create_experimentBiases_figure(plt_opts :: makie_plot_options)
    
    # Load data
    data_path    = "./data/lab_experiments/MineRS_MagmaDB_merged_v2_H2Ocalc.csv"
    data, header = readdlm(data_path, ';', header=true)
    df           = DataFrame(data, vec(header))

    # Create 2D KDE plot
    npts = 100
    σT   = 30.0
    σP   = 0.5
    σSi  = 2.0
    σϕ   = 4.0

    # Initialize figure
    fg1 = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    GL1 = fg1[1,1] = GridLayout()
    GL2 = fg1[1,2] = GridLayout()

    # Fill figure
    density_map!(GL1, df[:, "T_C"], df[:, "P_kbar"]; colormap = Reverse(:bilbao), σx = σT, σy = σP, npts = npts, marg = :hist, bar_color = :salmon, xlabel = L"$T$ [°C]", ylabel = L"$P$ [kbar]")
    density_map!(GL2, df[:, "SiO2_start"], df[:, "melt_fraction"]; colormap = Reverse(:bilbao), σx = σSi, σy = σϕ, npts = npts, marg = :hist, xlabel = L"SiO$_2$ [wt-%]", ylabel = L"$\phi$ [%]")

    axs1 = [contents(contents(GL1)[i])[1] for i in 1:3]
    axs2 = [contents(contents(GL2)[i])[1] for i in 1:3]
    axs = vcat(axs1, axs2)
    spl_labels = ["b", "a", "c", "e", "d", "f"]
    # Add subplot labels
    text!(  axs[1], 0, 1, 
        text = "a", 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    text!(  axs[4], 0, 1, 
        text = "b", 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    # for (idx_ax, Ax) in enumerate(axs)
    #     text!(  Ax, 0, 1, 
    #     text = string(spl_labels[idx_ax]), 
    #     space = :relative, align = (:left, :top), 
    #     offset = (4, -2), 
    #     font = :bold, 
    #     fontsize = plt_opts.font_size
    #     )
    # end

    # Print figure
    display(fg1)
    save("./figures/Figure_1_experimental_bias.png", fg1, px_per_unit = plt_opts.fig_res)

    # Return
    return nothing

end

# Benchmark
function create_benchmark_figure(plt_opts :: makie_plot_options)
    
    # Load results
    df_isor  = JLD2.load("./data/benchmark/BenchmarkTools_isoresources.jld2", "df")
    df_Mpeak = JLD2.load("./data/benchmark/BenchmarkTools_MAGEMin_peakperformance.jld2", "df")
    df_Npeak = JLD2.load("./data/benchmark/BenchmarkTools_Network_peakperformance_64BLASthreads_withGPU.jld2", "df")

    # Subset data frames
    df_Npeak_CPU  = @rsubset df_Npeak :Solver == "Net CPU"
    df_Npeak_GPU  = @rsubset df_Npeak :Solver == "Net GPU"

    # Different σ for ini composition - Net isoresources
    df_isor_Net0  = @chain df_isor begin
        @rsubset :Solver == "Net"
        @rsubset :σ      == 0.0
    end
    df_isor_Net01  = @chain df_isor begin
        @rsubset :Solver == "Net"
        @rsubset :σ      == 0.1
    end
    df_isor_Net1  = @chain df_isor begin
        @rsubset :Solver == "Net"
        @rsubset :σ      == 1.0
    end
    df_isor_Net10  = @chain df_isor begin
        @rsubset :Solver == "Net"
        @rsubset :σ      == 10.0
    end
    
    # Different σ for ini composition - MAGE isoresources
    df_isor_Mnoig0 = @chain df_isor begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 0.0
    end
    df_isor_Mnoig01 = @chain df_isor begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 0.1
    end
    df_isor_Mnoig1 = @chain df_isor begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 1.0
    end
    df_isor_Mnoig10 = @chain df_isor begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 10.0
    end

    # Different σ for ini composition - MAGE iguess isoresources
    df_isor_Mig0 = @chain df_isor begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 0.0
    end
    df_isor_Mig01 = @chain df_isor begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 0.1
    end
    df_isor_Mig1 = @chain df_isor begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 1.0
    end
    df_isor_Mig10 = @chain df_isor begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 10.0
    end

    # Different σ for ini composition - MAGE peak performance
    df_peak_Mnoig0 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 0.0
    end
    df_peak_Mnoig01 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 0.1
    end
    df_peak_Mnoig1 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 1.0
    end
    df_peak_Mnoig10 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE"
        @rsubset :σ      == 10.0
    end

    # Different σ for ini composition - MAGE iguess peak performance
    df_peak_Mig0 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 0.0
    end
    df_peak_Mig01 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 0.1
    end
    df_peak_Mig1 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 1.0
    end
    df_peak_Mig10 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :σ      == 10.0
    end

    # Different σ for ini composition - Net peak performance
    df_peak_Net0_CPU  = @rsubset df_Npeak_CPU :σ == 0.0
    df_peak_Net01_CPU = @rsubset df_Npeak_CPU :σ == 0.1
    df_peak_Net1_CPU  = @rsubset df_Npeak_CPU :σ == 1.0
    df_peak_Net10_CPU = @rsubset df_Npeak_CPU :σ == 10.0
    df_peak_Net0_GPU  = @rsubset df_Npeak_GPU :σ == 0.0
    df_peak_Net01_GPU = @rsubset df_Npeak_GPU :σ == 0.1
    df_peak_Net1_GPU  = @rsubset df_Npeak_GPU :σ == 1.0
    df_peak_Net10_GPU = @rsubset df_Npeak_GPU :σ == 10.0

    # Initialize figure
    fg1 = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    gl1 = fg1[1,1:2] = GridLayout()
    gl2 = fg1[1,3:4] = GridLayout()
    gl3 = fg1[2,1:2] = GridLayout()
    gl4 = fg1[2,3:4] = GridLayout()
    gl5 = fg1[1:2, 5] = GridLayout()
    ax1 = Axis(
        gl1[1,1],
        xlabel = L"$$Npts [ ]", ylabel = L"$$Time [s]",
        xscale = log10, yscale = log10,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9),
        limits = (0.7, 20_000, 1e-5, 2000),
        xticks = ([1, 10, 100, 1000, 10000], ["1", "10", "100", "1000", "10000"]),
        yticks = ([1e-4, 1e-2, 1, 100], ["0.0001", "0.01", "1", "100"]),
        title = L"$$Iso resources"
    )
    ax2 = Axis(
        gl3[1,1],
        xlabel = L"$$Npts [ ]", ylabel = L"$$Allocations [MiB]",
        yscale = log10,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(9), xminorticks = IntervalsBetween(9),
        xticks = ([0, 1, 2, 3, 4], ["1", "10", "100", "1000", "10000"]),
        yticks = ([0.0, 0.01, 1, 100, 10000], ["0", "0.01", "1", "100", "10000"])
    )
    ax3 = Axis(
        gl2[1,1],
        xlabel = L"$$Npts [ ]", ylabel = L"$$Time [s]",
        xscale = log10, yscale = log10,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9),
        limits = (0.7, 20_000, 1e-5, 2000),
        xticks = ([1, 10, 100, 1000, 10000], ["1", "10", "100", "1000", "10000"]),
        yticks = ([1e-4, 1e-2, 1, 100], ["0.0001", "0.01", "1", "100"]),
        title = L"$$Peak performance"
    )
    ax4 = Axis(
        gl4[1,1],
        xlabel = L"$$Npts [ ]", ylabel = L"$$Memory [MiB]",
        yscale = log10,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(9), xminorticks = IntervalsBetween(9),
        xticks = ([0, 1, 2, 3, 4], ["1", "10", "100", "1000", "10000"]),
        yticks = ([0.0, 0.01, 1, 100, 10000], ["0", "0.01", "1", "100", "10000"])
    )

    # Iso resources - Time
    sc1  = scatter!(ax1, df_isor_Mnoig0[:,  :Npts], df_isor_Mnoig0[:,  :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln1  =   lines!(ax1, df_isor_Mnoig0[:,  :Npts], df_isor_Mnoig0[:,  :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc2  = scatter!(ax1, df_isor_Mnoig01[:, :Npts], df_isor_Mnoig01[:, :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln2  =   lines!(ax1, df_isor_Mnoig01[:, :Npts], df_isor_Mnoig01[:, :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc3  = scatter!(ax1, df_isor_Mnoig1[:,  :Npts], df_isor_Mnoig1[:,  :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln3  =   lines!(ax1, df_isor_Mnoig1[:,  :Npts], df_isor_Mnoig1[:,  :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc4  = scatter!(ax1, df_isor_Mnoig10[:, :Npts], df_isor_Mnoig10[:, :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :skyblue4)
    ln4  =   lines!(ax1, df_isor_Mnoig10[:, :Npts], df_isor_Mnoig10[:, :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc5  = scatter!(ax1, df_isor_Mig0[:,    :Npts], df_isor_Mig0[:,    :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln5  =   lines!(ax1, df_isor_Mig0[:,    :Npts], df_isor_Mig0[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc6  = scatter!(ax1, df_isor_Mig01[:,   :Npts], df_isor_Mig01[:,   :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln6  =   lines!(ax1, df_isor_Mig01[:,   :Npts], df_isor_Mig01[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc7  = scatter!(ax1, df_isor_Mig1[:,    :Npts], df_isor_Mig1[:,    :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln7  =   lines!(ax1, df_isor_Mig1[:,    :Npts], df_isor_Mig1[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc8  = scatter!(ax1, df_isor_Mig10[:,   :Npts], df_isor_Mig10[:,   :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :skyblue1)
    ln8  =   lines!(ax1, df_isor_Mig10[:,   :Npts], df_isor_Mig10[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc9  = scatter!(ax1, df_isor_Net0[:,    :Npts], df_isor_Net0[:,    :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln9  =   lines!(ax1, df_isor_Net0[:,    :Npts], df_isor_Net0[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc10 = scatter!(ax1, df_isor_Net01[:,   :Npts], df_isor_Net01[:,   :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln10 =   lines!(ax1, df_isor_Net01[:,   :Npts], df_isor_Net01[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc11 = scatter!(ax1, df_isor_Net1[:,    :Npts], df_isor_Net1[:,    :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln11 =   lines!(ax1, df_isor_Net1[:,    :Npts], df_isor_Net1[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc12 = scatter!(ax1, df_isor_Net10[:,   :Npts], df_isor_Net10[:,   :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln12 =   lines!(ax1, df_isor_Net10[:,   :Npts], df_isor_Net10[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc13 = scatter!(ax3, df_peak_Mnoig0[:,  :Npts], df_peak_Mnoig0[:,  :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln13 =   lines!(ax3, df_peak_Mnoig0[:,  :Npts], df_peak_Mnoig0[:,  :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc14 = scatter!(ax3, df_peak_Mnoig01[:, :Npts], df_peak_Mnoig01[:, :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln14 =   lines!(ax3, df_peak_Mnoig01[:, :Npts], df_peak_Mnoig01[:, :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc15 = scatter!(ax3, df_peak_Mnoig1[:,  :Npts], df_peak_Mnoig1[:,  :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :skyblue4)
    ln15 =   lines!(ax3, df_peak_Mnoig1[:,  :Npts], df_peak_Mnoig1[:,  :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc16 = scatter!(ax3, df_peak_Mnoig10[:, :Npts], df_peak_Mnoig10[:, :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :skyblue4)
    ln16 =   lines!(ax3, df_peak_Mnoig10[:, :Npts], df_peak_Mnoig10[:, :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue4)
    sc17 = scatter!(ax3, df_peak_Mig0[:,    :Npts], df_peak_Mig0[:,    :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln17 =   lines!(ax3, df_peak_Mig0[:,    :Npts], df_peak_Mig0[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc18 = scatter!(ax3, df_peak_Mig01[:,   :Npts], df_peak_Mig01[:,   :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln18 =   lines!(ax3, df_peak_Mig01[:,   :Npts], df_peak_Mig01[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc19 = scatter!(ax3, df_peak_Mig1[:,    :Npts], df_peak_Mig1[:,    :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :skyblue1)
    ln19 =   lines!(ax3, df_peak_Mig1[:,    :Npts], df_peak_Mig1[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc20 = scatter!(ax3, df_peak_Mig10[:,   :Npts], df_peak_Mig10[:,   :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :skyblue1)
    ln20 =   lines!(ax3, df_peak_Mig10[:,   :Npts], df_peak_Mig10[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :dash, linewidth = plt_opts.line_width,  color = :skyblue1)
    sc21 = scatter!(ax3, df_peak_Net0_CPU[:,    :Npts], df_peak_Net0_CPU[:,    :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln21 =   lines!(ax3, df_peak_Net0_CPU[:,    :Npts], df_peak_Net0_CPU[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc22 = scatter!(ax3, df_peak_Net01_CPU[:,   :Npts], df_peak_Net01_CPU[:,   :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln22 =   lines!(ax3, df_peak_Net01_CPU[:,   :Npts], df_peak_Net01_CPU[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc23 = scatter!(ax3, df_peak_Net1_CPU[:,    :Npts], df_peak_Net1_CPU[:,    :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln23 =   lines!(ax3, df_peak_Net1_CPU[:,    :Npts], df_peak_Net1_CPU[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc24 = scatter!(ax3, df_peak_Net10_CPU[:,   :Npts], df_peak_Net10_CPU[:,   :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :goldenrod1)
    ln24 =   lines!(ax3, df_peak_Net10_CPU[:,   :Npts], df_peak_Net10_CPU[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :goldenrod1)
    sc25 = scatter!(ax3, df_peak_Net0_GPU[:,    :Npts], df_peak_Net0_GPU[:,    :("median(Time) [ns]")] ./ 1e9, marker = :circle, markersize = plt_opts.marker_size, color = :black)
    ln25 =   lines!(ax3, df_peak_Net0_GPU[:,    :Npts], df_peak_Net0_GPU[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :black)
    sc26 = scatter!(ax3, df_peak_Net01_GPU[:,   :Npts], df_peak_Net01_GPU[:,   :("median(Time) [ns]")] ./ 1e9, marker = :utriangle, markersize = plt_opts.marker_size, color = :black)
    ln26 =   lines!(ax3, df_peak_Net01_GPU[:,   :Npts], df_peak_Net01_GPU[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :black)
    sc27 = scatter!(ax3, df_peak_Net1_GPU[:,    :Npts], df_peak_Net1_GPU[:,    :("median(Time) [ns]")] ./ 1e9, marker = :dtriangle, markersize = plt_opts.marker_size, color = :black)
    ln27 =   lines!(ax3, df_peak_Net1_GPU[:,    :Npts], df_peak_Net1_GPU[:,    :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :black)
    sc28 = scatter!(ax3, df_peak_Net10_GPU[:,   :Npts], df_peak_Net10_GPU[:,   :("median(Time) [ns]")] ./ 1e9, marker = :star8, markersize = plt_opts.marker_size, color = :black)
    ln28 =   lines!(ax3, df_peak_Net10_GPU[:,   :Npts], df_peak_Net10_GPU[:,   :("median(Time) [ns]")] ./ 1e9, linestyle = :solid, linewidth = plt_opts.line_width,  color = :black)
    
    # Iso resources - Memory
    df_isor_sorted = sort(df_isor, :Npts)
    dodge_vec = [1 for _ in axes(df_isor_sorted, 1)]
    idx = findall(x -> x == "MAGE iguess", df_isor_sorted[:, :Solver])
    dodge_vec[idx] .= 2
    idx = findall(x -> x == "Net", df_isor_sorted[:, :Solver])
    dodge_vec[idx] .= 3
    bp1 = barplot!(ax2, log10.(df_isor_sorted[:, :Npts]), df_isor_sorted[:, :("Mem.  [b]")] ./ 1048576, dodge = dodge_vec, color = dodge_vec, colormap = :bamako10)

    # Peak performance - Memory
    df_peak        = vcat(df_Mpeak, vcat(df_Npeak_CPU, df_Npeak_GPU))
    df_peak_sorted = sort(df_peak, :Npts)
    dodge_vec = [1 for _ in axes(df_peak_sorted, 1)]
    idx = findall(x -> x == "MAGE iguess", df_peak_sorted[:, :Solver])
    dodge_vec[idx] .= 2
    idx = findall(x -> x == "Net CPU", df_peak_sorted[:, :Solver])
    dodge_vec[idx] .= 3
    idx = findall(x -> x == "Net GPU", df_peak_sorted[:, :Solver])
    dodge_vec[idx] .= 4
    bp1 = barplot!(ax4, log10.(df_peak_sorted[:, :Npts]), df_peak_sorted[:, :("Mem.  [b]")] ./ 1048576, dodge = dodge_vec, color = dodge_vec, colormap = :bamako10, label = ["M noig", "M ig", "Net CPU", "Net GPU"])

    # Add legend for all
    cmap = cgrad(:bamako10, 4, categorical = true)
    cols = cmap[1:4]
    labels = [L"$$M no iguess", L"$$M iguess", L"$$Net CPU", L"$$Net GPU"]
    elems = [PolyElement(polycolor = cols[i]) for i in eachindex(labels)]
    Legend(
        gl5[1,1],
        [
            sc1, sc2, sc3, sc4,
            ln1, ln5, ln12, ln25,
            elems[1], elems[2], elems[3], elems[4]
        ],
        [
            L"$\sigma = 0.0$", L"$\sigma = 0.1$", L"$\sigma = 1.0$", L"$\sigma = 10.0$",
            L"$$M no iguess", L"$$M iguess", L"$$Net CPU", L"$$Net GPU",
            L"$$M no iguess", L"$$M iguess", L"$$Net CPU", L"$$Net GPU",
        ],
        halign = :left,
        tellheight = false,
        tellwidth  = false,
        rowgap = 10,
        framevisible = false,
        labelsize = 14
    )

    # Add subplot label
    axs = [ax1, ax2, ax3, ax4]
    for (idx_ax, Ax) in enumerate(axs)
        text!(  Ax, 0, 1, 
        text = string(plt_opts.spl_alpha_small[idx_ax]), 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    end

    # Clean up
    hidexdecorations!(ax1, grid = false, minorgrid = false, minorticks = false)
    hideydecorations!(ax3, grid = false, minorgrid = false, minorticks = false)
    hidexdecorations!(ax3, grid = false, minorgrid = false, minorticks = false)
    hideydecorations!(ax4, grid = false, minorgrid = false, minorticks = false)
    
    # Display some values
    time_MAGE_noig_Iso = @chain df_isor_Mnoig0 begin
        @rsubset :Npts   == 10000
    end
    time_MAGE_iguess_Iso = @chain df_isor_Mig0 begin
        @rsubset :Npts   == 10000
    end
    time_NetCPU_Iso = @chain df_isor_Net0 begin
        @rsubset :Npts == 10000
    end
    time_MAGE_noig_10000 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE"
        @rsubset :Npts   == 10000
        @rsubset :σ == 0.0
    end
    time_MAGE_iguess_10000 = @chain df_Mpeak begin
        @rsubset :Solver == "MAGE iguess"
        @rsubset :Npts   == 10000
        @rsubset :σ == 0.0
    end
    time_NetCPU_10000 = @chain df_Npeak_CPU begin
        @rsubset :Npts == 10000
        @rsubset :σ == 0.0
    end
    time_NetGPU_10000 = @chain df_Npeak_GPU begin
        @rsubset :Npts == 10000
        @rsubset :σ == 0.0
    end
    println("Speedup Peak -> Net CPU / MAGE no iguess : $(time_MAGE_noig_10000[1, :("median(Time) [ns]")] / time_NetCPU_10000[1, :("median(Time) [ns]")]) x")
    println("Speedup Peak -> Net CPU / MAGE    iguess : $(time_MAGE_iguess_10000[1, :("median(Time) [ns]")] / time_NetCPU_10000[1, :("median(Time) [ns]")]) x")
    println("Speedup Peak -> Net CPU / Net GPU        : $(time_NetCPU_10000[1, :("median(Time) [ns]")] / time_NetGPU_10000[1, :("median(Time) [ns]")]) x")
    println("Speedup IsoR -> Net CPU / MAGE no iguess : $(time_MAGE_noig_Iso[1, :("median(Time) [ns]")] / time_NetCPU_Iso[1, :("median(Time) [ns]")]) x")
    println("Speedup IsoR -> Net CPU / MAGE    iguess : $(time_MAGE_iguess_Iso[1, :("median(Time) [ns]")] / time_NetCPU_Iso[1, :("median(Time) [ns]")]) x")
    
    # Save and display figure
    save("./figures/Figure_11_Benchmark_64BLAS.png", fg1, px_per_unit = plt_opts.fig_res)
    display(fg1)

    # Return
    return 
end

# Mineral abundance comparison all
function create_mineralComparison_figure(df :: DataFrame, plt_opts :: makie_plot_options)
    # Initialize figure
    fg1 = Figure(size = plt_opts.fig_size, fontsize = plt_opts.font_size, figure_padding = plt_opts.figure_pad)
    ax1 = Axis(
        fg1[1,1], aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 65, 0, 65),
        xticks = [0, 20, 40, 60], yticks = [0, 20, 40, 60]
    )
    ax2 = Axis(
        fg1[1,2], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 45, 0, 45),
        xticks = [0, 20, 40], yticks = [0, 20, 40]
    )
    ax3 = Axis(
        fg1[1,3], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 65, 0, 65),
        xticks = [0, 20, 40, 60], yticks = [0, 20, 40, 60]
    )
    ax4 = Axis(
        fg1[1,4], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 25, 0, 25),
        xticks = [0, 10, 20], yticks = [0, 10, 20]
    )
    ax5 = Axis(
        fg1[2,1], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 65, 0, 65),
        xticks = [0, 20, 40, 60], yticks = [0, 20, 40, 60]
    )
    ax6 = Axis(
        fg1[2,2], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 100, 0, 100),
        xticks = [0, 25, 50, 75, 100], yticks = [0, 25, 50, 75, 100]
    )
    ax7 = Axis(
        fg1[2,3], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 4, 0,4),
        xticks = [0, 2, 4], yticks = [0, 2, 4]
    )
    ax8 = Axis(
        fg1[2,4], 
        aspect = 1.0,
        xlabel = L"$$Exp. [wt-%]", ylabel = L"$$MAGE [wt-%]",
        limits = (0, 6, 0, 6),
        xticks = [0, 2, 4, 6], yticks = [0, 2, 4, 6]
    )
    ln1 = lines!(ax1, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln2 = lines!(ax2, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln3 = lines!(ax3, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln4 = lines!(ax4, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln5 = lines!(ax5, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln6 = lines!(ax6, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln7 = lines!(ax7, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    ln8 = lines!(ax8, 0.00001:1:100, 0.00001:1:100, color = :goldenrod1, linewidth = plt_opts.line_width)
    scatter!(ax1, df[:, :("plag_exp")], df[:, :("fsp [wt%]")], label = L"$$plag")
    scatter!(ax2, df[:, :("ol_exp")],   df[:, :("ol [wt%]")], label = L"$$ol")
    scatter!(ax3, df[:, :("amph_exp")], df[:, :("amp [wt%]")], label = L"$$amph")
    scatter!(ax4, df[:, :("opx_exp")],  df[:, :("opx [wt%]")], label = L"$$opx")
    scatter!(ax5, df[:, :("cpx_exp")],  df[:, :("cpx [wt%]")], label = L"$$cpx")
    scatter!(ax6, df[:, :("liq_exp")],  df[:, :("liq [wt%]")], label = L"$$liq")
    scatter!(ax7, df[:, :("ilm_exp")],  df[:, :("ilm [wt%]")], label = L"$$ilm")
    scatter!(ax8, df[:, :("sp_exp")],   df[:, :("spl [wt%]")], label = L"$$sp")
    axislegend(ax1, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax2, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax3, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax4, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax5, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax6, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax7, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))
    axislegend(ax8, position = :rb, framevisible = true, labelsize = 11, backgroundcolor = (:white, 0.0), framecolor = (:gray, 0.4))

    # Add subplot label
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    for (idx_ax, Ax) in enumerate(axs)
        text!(  Ax, 0, 1, 
        text = string(plt_opts.spl_alpha_small[idx_ax]), 
        space = :relative, align = (:left, :top), 
        offset = (4, -2), 
        font = :bold, 
        fontsize = plt_opts.font_size
        )
    end
    display(fg1)

    save("./figures/Figure_5_mineral_comparison.png", fg1, px_per_unit = plt_opts.fig_res)
end

# ----------------
# Run main
postprocess_comparison();