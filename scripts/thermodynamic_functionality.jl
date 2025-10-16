# Functionality for computational thermodynamics

function run_functionality()
    
    # Define bulk and oxides
    oxides  = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "O", "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
    X1      = [49.9, 17.0, 10.0, 8.67, 9.28, 1.5, 0.49, 3.02, 1.25, 0.05, 2.0]
    X2      = [ 45.67,    13.7,   6.0,  13.0,   7.6, 4.0,   2.1,    1.8,     0.1,   1.92,   4.9]
    X3      = [ 45.67,    13.7,   6.0,  13.0,   7.6, 4.0,   2.1,    1.8,     0.1,   1.92,   4.9]
    X       = [X1, X2]

    X_FeO   = 5.0
    X_Fe2O3 = 10.0

    # Test functions
    # X_anh, ox_anh = normalize_to_anhydrous_bulk(X, oxides)
    # Xan, ox_an    = renormalize_bulk_to_anX(X, oxides, "O")
    # X_norm = normalize_bulk(X)
    # X_oversat = oversaturate_bulk(X, "O", 4.0, oxides)

    X_FeOt = convert_ironoxides_to_FeOt(X_Fe2O3, X_FeO, "wt")

    println("FeO = $X_FeO, X_Fe2O3 = $X_Fe2O3, X_FeOt = $X_FeOt")

    # println(ox_anh)
    # println(X_anh)
    # println(ox_an)
    # println(Xan)
    # println(sum.(X))
    # println(X_norm)
    # println(X_oversat)
    
    # Return 
    return nothing
end

function normalize_to_anhydrous_bulk(   bulk_in :: Union{Vector{Float64}, Vector{Vector{Float64}}}, 
                                        oxides_in :: Vector{String})
    
    # Some helper function to deal with types
    vect(A) = [A]

    # Find index of water 
    idx_H2O = findfirst(x -> x .== "H2O", oxides_in)
    idx_anh = findall(x -> x .!= "H2O", oxides_in)

    if isnothing(idx_H2O)
        error("H2O is not in the oxide list. Make sure you are using hydrous bulk rock compositions.")
    end
    
    # Convert types
    typeof(bulk_in) == Vector{Float64} ? bulk = vect(deepcopy(bulk_in)) : bulk = deepcopy(bulk_in)
    
    # Initialize
    bulk_anh = [Vector{Float64}(undef, length(oxides_in) - 1) for _ in eachindex(bulk)]

    # Normalize
    for (idx_bulk, curr_bulk) in enumerate(bulk)
        sum_anh = sum(curr_bulk[idx_anh])
        bulk_anh[idx_bulk] = curr_bulk[idx_anh] ./ sum_anh
    end

    # Return
    return bulk_anh .* 100.0, oxides_in[idx_anh]
    
end

function renormalize_bulk_to_anX( bulk_in :: Union{Vector{Float64}, Vector{Vector{Float64}}}, 
                                oxides_in :: Vector{String},
                                an_oxide :: String)

    # Some helper function to deal with types
    vect(A) = [A]

    # Find index of water 
    idx_anOx = findfirst(x -> x .== an_oxide, oxides_in)
    idx_an   = findall(x -> x .!= an_oxide, oxides_in)

    if isnothing(idx_anOx)
    error("$(an_oxide) is not in the oxide list. Make sure you are using proper bulk rock compositions.")
    end

    # Convert types
    typeof(bulk_in) == Vector{Float64} ? bulk = vect(deepcopy(bulk_in)) : bulk = deepcopy(bulk_in)

    # Initialize
    bulk_an = [Vector{Float64}(undef, length(oxides_in) - 1) for _ in eachindex(bulk)]

    # Normalize
    for (idx_bulk, curr_bulk) in enumerate(bulk)
    sum_an = sum(curr_bulk[idx_an])
    bulk_an[idx_bulk] = curr_bulk[idx_an] ./ sum_an
    end

    # Return
    return bulk_an .* 100.0, oxides_in[idx_an]

end

function normalize_bulk( bulk_in :: Union{Vector{Float64}, Vector{Vector{Float64}}})

    # Some helper function to deal with types
    vect(A) = [A]

    # Convert types
    typeof(bulk_in) == Vector{Float64} ? bulk = vect(deepcopy(bulk_in)) : bulk = deepcopy(bulk_in)

    # Initialize
    bulk_norm = [Vector{Float64}(undef, length(bulk[idx])) for idx in eachindex(bulk)]

    # Normalize
    for (idx_bulk, curr_bulk) in enumerate(bulk)
        sum_bulk = sum(curr_bulk)
        bulk_norm[idx_bulk] = curr_bulk ./ sum_bulk
    end

    # Return
    if typeof(bulk_in) == Vector{Float64}
        return bulk_norm[1] .* 100.0
    else
        return bulk_norm .* 100.0
    end

end

# Manually convert bulk rock 
function convert_bulk(bulk_in :: Union{Vector{Float64}, Vector{Vector{Float64}}}, oxides_in :: Vector{String}, conversion :: String)
    
    # Oxides and molar masses
    ref_ox          = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2",  "O", "Cr2O3",  "MnO",  "H2O",   "S"]      # Oxide list
    ref_MolarMass   = [60.08,   101.96, 56.08, 40.30, 71.85,  159.69,  94.2,  61.98,  79.88, 16.0,  151.99, 70.937, 18.015, 32.06]      # Reference molar mass [g/mol]

    # Some helper function to deal with types
    vect(A) = [A]

    # Convert types
    typeof(bulk_in) == Vector{Float64} ? bulk = vect(deepcopy(bulk_in)) : bulk = deepcopy(bulk_in)

    # Initialize
    bulk_conv = [Vector{Float64}(undef, length(bulk[idx])) for idx in eachindex(bulk)]

    # Convert
    if conversion == "wt_to_mol"
        for (idx_bulk, Bulk) in enumerate(bulk)
            for (idx_oxide, Oxide) in enumerate(oxides_in)
                idOx_ref = findfirst(x -> x == Oxide, ref_ox)
                bulk_conv[idx_bulk][idx_oxide] = Bulk[idx_oxide] / ref_MolarMass[idOx_ref]
            end
        end
    end

    # Return
    if typeof(bulk_in) == Vector{Float64}
        return bulk_conv[1] .* 100.0
    else
        return bulk_conv .* 100.0
    end

end

function oversaturate_bulk(bulk_in :: Union{Vector{Float64}, Vector{Vector{Float64}}}, oxide_sat :: String, sat_value :: Float64, oxides_in :: Vector{String})
    # Some helper function to deal with types
    vect(A) = [A]

    # Convert types
    typeof(bulk_in) == Vector{Float64} ? bulk = vect(deepcopy(bulk_in)) : bulk = deepcopy(bulk_in)

    # Initialize output array
    bulk_oversat = [Vector{Float64}(undef, length(bulk[idx])) for idx in eachindex(bulk)]

    # Set concentration [mol] at which oversaturation is typically achieved
    target_concentration = 

    # Find position of oxygen in arrays
    idO = findfirst(x -> x == oxide_sat, oxides_in)

    # Loop through bulk and start calculating
    for (idx_bulk, Bulk) in enumerate(bulk)

        # Define intervall
        lower_guess       = 0.1
        upper_guess       = 10.0
        lower_bulk        = deepcopy(Bulk); lower_bulk[idO] = lower_guess
        upper_bulk        = deepcopy(Bulk); upper_bulk[idO] = upper_guess
        guess_bulk        = deepcopy(Bulk)
        lower_bulk_mol    = convert_bulk(lower_bulk, oxides_in, "wt_to_mol")
        upper_bulk_mol    = convert_bulk(upper_bulk, oxides_in, "wt_to_mol")
        lower_bulk_mol_n  = normalize_bulk(lower_bulk_mol)
        upper_bulk_mol_n  = normalize_bulk(upper_bulk_mol)
        res_upper = upper_bulk_mol_n[idO] - sat_value
        res_lower = lower_bulk_mol_n[idO] - sat_value

        # Start bisection algorithm
        ϵ_tol = 1e-2; max_iter = 100
        for iter in 1:max_iter
            if res_upper * res_lower > 0.0
                error("The residuals must have different signs. Consider changing the starting intervals.")
            end
            guess            = (upper_guess + lower_guess) / 2.0
            guess_bulk[idO]  = guess
            guess_bulk_mol   = convert_bulk(guess_bulk, oxides_in, "wt_to_mol")
            guess_bulk_mol_n = normalize_bulk(guess_bulk_mol)
            res_guess        = guess_bulk_mol_n[idO] - sat_value

            # Check if root is found
            if abs(res_guess) < ϵ_tol || (abs(upper_guess - lower_guess) / 2.0) < ϵ_tol
                # println("Converged after $iter iterations to residual = $res_guess")
                bulk_oversat[idx_bulk] = deepcopy(guess_bulk)
                break
            end

            # Update the intervals
            if res_lower * res_guess < 0
                upper_guess = deepcopy(guess)
                res_upper   = deepcopy(res_guess)
            else
                lower_guess = deepcopy(guess)
                res_lower   = deepcopy(res_guess)
            end

            # Break if not converged sufficiently
            if iter == max_iter && res_guess > ϵ_tol
                error("Bisection algorithm did not sufficiently converge in $max_iter iterations")
            end
        end
    end
    
    # Return
    if typeof(bulk_in) == Vector{Float64}
        return bulk_oversat[1]
    else
        return bulk_oversat
    end

end

# Convert iron oxides to total FeO equivalent
function convert_ironoxides_to_FeOt(X_Fe2O3 :: Float64, X_FeO :: Float64, sys_in :: String)
    
    # Sanity checks
    if sys_in != "wt" && sys_in != "mol"
        error("Undefined unit for variable sys_in. Supported units are wt and mol")
    end

    # Create local copies
    X_Fe2O3 = deepcopy(X_Fe2O3)
    X_FeOt  = deepcopy(X_FeO  )

    # Molar mass of iron oxides
    molar_masses = (71.85, 159.69)      # (FeO, Fe2O3) [g / mol]

    # Convert to mol if necessary
    if sys_in == "wt"
        X_Fe2O3  /= molar_masses[2]
    end

    # Calculate moles of FeO equivalent
    X_FeO_mol_eq = 2.0 * X_Fe2O3
    
    # Calculate the total FeO
    if sys_in == "wt"
        X_FeOt += X_FeO_mol_eq * molar_masses[1]
    else
        X_FeOt += X_FeO_mol_eq
    end

    # Return
    return X_FeOt
end

function postprocess_MAGEMin_struct(Out_MAGEMin :: Vector{MAGEMin_C.gmin_struct{Float64, Int64}};
                                    remove_phase :: Bool = false,
                                    remove_list  :: Vector{String}
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
    if remove_phase
    idx_rm_ph = findall(in(remove_list), ph_obs)
    isnothing(idx_rm_ph) ? nothing : deleteat!(ph_obs, idx_rm_ph)
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

    if remove_phase                                                                 # In case some phase should be excluded
    idx_rm_ph   = findall(in(remove_list), target_phases)                       # ... find their indices in the out struct
    isnothing(idx_rm_ph) ? nothing : deleteat!(target_phases,    idx_rm_ph)     # ... remove them from the target
    isnothing(idx_rm_ph) ? nothing : deleteat!(target_phases_wt, idx_rm_ph)     # ... remove their weights too
    idx_ph      = findall(in(target_phases), out.ph)                            # ... find position of target phases in out struct
    sum_ph_wt   = sum(out.ph_frac_wt[idx_ph])                                   # ... calculate weight sum
    target_phases_wt ./= sum_ph_wt                                              # ... renormalize
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
    if isnan(out.bulk_M_wt[idx_Ox])
    push!(df_out_db[!, "$(iOx)_liq [wt%]"], NaN)
    else
    push!(df_out_db[!, "$(iOx)_liq [wt%]"], out.bulk_M_wt[idx_Ox] .* 100.0)
    end

    # Solid
    if isnan(out.bulk_S_wt[idx_Ox])
    push!(df_out_db[!, "$(iOx)_sol [wt%]"], NaN)
    else
    push!(df_out_db[!, "$(iOx)_sol [wt%]"], out.bulk_S_wt[idx_Ox] .* 100.0)
    end

    # Start
    push!(df_out_db[!, "$(iOx)_MAGE_start [wt%]"], out.bulk_wt[idx_Ox] .* 100.0)
    end
    end

    # Return
    return df_out_db
end

# run_functionality()

