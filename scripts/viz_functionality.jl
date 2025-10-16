# Some structures and functions that help me nicely organize my figures
using KernelDensity, CairoMakie, Statistics, MathTeXEngine

# ------------------------ #
#|      Structures        |#
# ------------------------ #

struct makie_plot_options
    fig_size        :: Tuple{Int64, Int64}
    fig_res         :: Float64
    line_width      :: Float64
    font_size       :: Float64
    marker_size     :: Float64
    cm_to_pt        :: Float64
    golden_ratio    :: Float64
    label_size      :: Float64
    tick_label_pad  :: Float64
    figure_pad      :: Float64
    spl_alpha_small :: Vector{Char}
    spl_alpha_Caps  :: Vector{Char}
end

# -------------------------- #
#|    Makie plot options    |#
# -------------------------- #

function makie_plot_options(;   fig_size       :: Union{Tuple{Int64, Int64}, Tuple{Float64, Float64}} = (397, 561),
                                fig_res        :: Union{Int64, Float64} = 2,
                                line_width     :: Union{Int64, Float64} = 2.5,
                                font_size      :: Union{Int64, Float64} = 18.0,
                                marker_size    :: Union{Int64, Float64} = 12.0,
                                cm_to_pt       :: Union{Int64, Float64} = 28.3465,
                                golden_ratio   :: Float64               = 1.618,
                                label_size     :: Union{Int64, Float64} = 16.0,
                                tick_label_pad :: Union{Int64, Float64} = 15.0,
                                figure_pad     :: Union{Int64, Float64} = 20.0
    )

    # Convert figure resolution to integer values of pixels
    typeof(fig_size) == Float64 ? Int.(fig_size   ) : nothing

    # Add subplot labels
    spl_alpha_small = [label for label in 'a':'z']
    spl_alpha_Caps  = [label for label in 'A':'Z']

    # Return structure
    return makie_plot_options(fig_size, fig_res, line_width, font_size, marker_size, cm_to_pt, golden_ratio, label_size, tick_label_pad, figure_pad, spl_alpha_small, spl_alpha_Caps)
end

# -------------------------- #
#|      Makie KDE plot      |#
# -------------------------- #

function KDE_plot(
        ax        :: Union{Axis, GridPosition},
        field     :: Vector{Float64};
        showdata  :: Bool    = false,
        kernel    :: Symbol  = :Gauss,
        bandwidth :: Float64 = 1.0,
        npoints   :: Int64   = 600
    )

    # Define kernel
    if kernel == :Gauss
        K(x) = 1.0 ./ sqrt(2π) .* exp.(- ( x ).^2 ./ 2.0)
    end

    # Compute probability density
    ρ_kernel  = Vector{eltype(field)}(undef, npoints)
    ρ_kernel .= 0.0
    x_kernel  = LinRange(minimum(field) - 0.4, maximum(field) + 0.4, npoints)
    # x_kernel  = LinRange(-3.0, 3.0, npoints)
    for idx in eachindex(field)
        ρ_kernel .+= K( (x_kernel .- field[idx]) ./ bandwidth )
    end


    # Normalize to sum to unity
    ρ_kernel ./= size(field)[1] .* bandwidth

    total = 0.0
    for idx in eachindex(ρ_kernel)
        total += ρ_kernel[idx] * (x_kernel[2] - x_kernel[1])
    end

    println(mean(ρ_kernel))
    println(std(ρ_kernel))

    lines(ax, x_kernel, ρ_kernel)
    if showdata
        scatter!(ax, field, 0.0 .* field, color =:black, markersize = 10.0, marker = :rect)
    end
    density!(ax, field, bandwidth = bandwidth, boundary = (minimum(field) - 0.4, maximum(field) + 0.4))

    # Return
    return nothing

end

# 2D KDE plot
function density_map!(
    GL :: GridLayout,
    x  :: AbstractArray,
    y  :: AbstractArray;
    σx :: Float64 = 1.0,
    σy :: Float64 = 1.0,
    npts :: Int64 = 100,
    marg :: Symbol = :hist,
    colormap :: Union{Symbol, Reverse{Symbol}} = Reverse(:bilbao),
    bar_color :: Symbol = :skyblue,
    xlabel :: AbstractString = "x",
    ylabel :: AbstractString = "y"
)
    xmin, xmax = minimum(x), maximum(x)
    ymin, ymax = minimum(y), maximum(y)
    xgrid = LinRange(xmin, xmax, npts) |> collect
    ygrid = LinRange(ymin, ymax, npts) |> collect
    x2D = repeat(xgrid, npts) |> xgrid -> reshape(xgrid, npts, npts)
    y2D = repeat(ygrid, npts) |> ygrid -> reshape(ygrid, npts, npts)' |> Matrix{Float64}

    ρ̃ = zeros(Float64, npts, npts)
    A = 1.0 / (length(x) * 2π *σx*σy)
    for idx in eachindex(x)
        ρ̃ .+= A .* exp.( -( (x2D .- x[idx]).^2 ./ σx.^2 .+ (y2D .- y[idx]).^2 ./ σy.^2 ) )
    end
    GL1 = GL[2:4, 1:3] = GridLayout()
    GL2 = GL[1,   1:3] = GridLayout()
    GL3 = GL[2:4,   4] = GridLayout()
    GL4 = GL[5,   1:3] = GridLayout()
    ax1 = Axis(GL1[1,1], xlabel = xlabel, ylabel = ylabel)
    ax2 = Axis(GL2[1,1])
    ax3 = Axis(GL3[1,1])
    cnf = contourf!(ax1, x2D, y2D, ρ̃, colormap = colormap)
    if marg == :hist
        hist!(ax2, x, color = bar_color)
        hist!(ax3, y, direction = :x, color = bar_color)
    end
    Colorbar(GL4[1, 1], cnf, vertical = false, ticks = ([minimum(ρ̃), maximum(ρ̃)], ["low", "high"]), label = L"$$Kernel Density [ ]", flipaxis = false)

    # Make the plot nice
    ax1.limits = (xmin, xmax, ymin, ymax)
    xlims!(ax2, (xmin, xmax))
    ylims!(ax3, (ymin, ymax))
end
# ------------------------- #
#|       Testing           |#
# ------------------------- #

function testing()

    # Define random data for KDE plot
    x = [1.33, 0.3, 0.97, 1.1, 0.1, 1.4, 0.4]
    # x = [0.0]

    fg1 = Figure()

    KDE_plot(fg1[1,1], x; showdata = true, bandwidth = 0.03)
    display(fg1)

    # Return
    return nothing

end

# testing();