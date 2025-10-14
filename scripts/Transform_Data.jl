using Statistics

"""
    TransformData(data; transform, dims)

Transforms highly variable data stored in `data` Matrix along dimension `dims` to the same space. Currently, unit range is the only supoorted transformation type and can be activated by setting `transform` to `:unit` or `:zscore`. Default transformation type is unit and default dimension is 1 (i.e. along the rows). Return arguments include the transformed data Matrix as well the scaling parameters. If `scale` is `true`, `data` will be scaled according to the transformation method and required parameters.
"""
function TransformData(
    data       :: AbstractArray;
    transform  :: Symbol = :unit,
    dims       :: Int  = 1,
    scale      :: Bool   = false,
    min_data   :: Union{AbstractArray, Nothing} = nothing,
    max_data   :: Union{AbstractArray, Nothing} = nothing,
    mean_data  :: Union{AbstractArray, Nothing} = nothing,
    std_data   :: Union{AbstractArray, Nothing} = nothing,
    mask       :: Union{AbstractArray, Nothing} = nothing,
    DatType    :: Type = Float32
)
    # Initialize
    data_t = copy(data)

    # Check the mask
    if isnothing(mask)
        mask = Int.(ones(size(data)))
    end

    # Unit-Range (Min-Max)
    if transform == :unit && scale == false
        min_data_new = minimum(data, dims = dims)
        max_data_new = maximum(data, dims = dims)

        if dims == 1
            for Idx in axes(data_t, dims)
                data_t[Idx, :] .-= min_data_new'
                data_t[Idx, :] ./= (max_data_new .- min_data_new)'
            end
        elseif dims == 2
            for Idx in axes(data_t, dims)
                data_t[:, Idx] .-= min_data_new
                data_t[:, Idx] ./= (max_data_new .- min_data_new)
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data_t, min_data_new, max_data_new
    end

    # Data scaling
    if transform == :unit && scale == true

        # Sanity check
        isnothing(min_data) ? error("UndefKeywordError: keyword argument `min_data` not assigned") : nothing
        isnothing(max_data) ? error("UndefKeywordError: keyword argument `max_data` not assigned") : nothing

        # Transformation
        if dims == 1
            for Idx in axes(data_t, dims)
                data_t[Idx, :] .-= min_data'
                data_t[Idx, :] ./= (max_data .- min_data)'
            end
        elseif dims == 2
            for Idx in axes(data_t, dims)
                data_t[:, Idx] .-= min_data
                data_t[:, Idx] ./= (max_data .- min_data)
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data_t
    end

    # Zscore (mean = 0, std = 1)
    if transform == :zscore && scale == false
        mean_data_new = DatType.(Statistics.mean(data, dims = dims))
        std_data_new  = DatType.(Statistics.std( data, dims = dims))

        if dims == 1
            for Idx in axes(data_t, dims)
                data_t[Idx, :] .-= mean_data_new'
                data_t[Idx, :] ./= std_data_new'
            end
        elseif dims == 2
            for Idx in axes(data_t, dims)
                data_t[:, Idx] .-= mean_data_new
                data_t[:, Idx] ./= std_data_new
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data_t, mean_data_new, std_data_new
    end

    # Data scaling
    if transform == :zscore && scale == true

        # Sanity check
        isnothing(mean_data) ? error("UndefKeywordError: keyword argument `mean_data` not assigned") : nothing
        isnothing(std_data) ? error("UndefKeywordError: keyword argument `std_data` not assigned") : nothing

        # Transformation
        if dims == 1
            for Idx in axes(data_t, dims)
                data_t[Idx, :] .-= mean_data'
                data_t[Idx, :] ./= std_data'
            end
        elseif dims == 2
            for Idx in axes(data_t, dims)
                data_t[:, Idx] .-= mean_data
                data_t[:, Idx] ./= std_data
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data_t
    end
end

"""
    TransformDataBack(data_t; transform, dims, transform parameters ...)

Transform a transformed data Matrix along its dimension `dims` back to its original data range using the transform parameters and the transformation method specified by `transform`. Currently supported transformation type is `:unit` and requires the definition of keyword arguments `min_data` and `max_data`.
"""
function TransformDataBack(
    data_t    :: AbstractArray;
    transform :: Symbol = :unit,
    dims      :: Int = 1,
    min_data  :: Union{AbstractArray, Nothing} = nothing,
    max_data  :: Union{AbstractArray, Nothing} = nothing,
    mean_data :: Union{AbstractArray, Nothing} = nothing,
    std_data  :: Union{AbstractArray, Nothing} = nothing
)
    # Initialize
    data = deepcopy(data_t)

    # Unit
    if transform == :unit

        # Dimension dependent transformation
        if dims == 1
            for Idx in axes(data, dims)
                data[Idx, :] .*= (max_data .- min_data)'
                data[Idx, :] .+= min_data'
            end
        elseif dims == 2
            for Idx in axes(data, dims)
                data[:, Idx] .*= (max_data .- min_data)
                data[:, Idx] .+= min_data
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data
    end

    # Unit
    if transform == :zscore

        # Dimension dependent transformation
        if dims == 1
            for Idx in axes(data, dims)
                data[Idx, :] .*= std_data'
                data[Idx, :] .+= mean_data'
            end
        elseif dims == 2
            for Idx in axes(data, dims)
                data[:, Idx] .*= std_data
                data[:, Idx] .+= mean_data
            end
        else
            error("DimensionError: requested dimension is higher than the currently supported no. of dimensions")
        end

        # Return
        return data
    end
end

# Run Main - THE LINES BELOW SHOULD BE USED FOR TESTING ONCE THE PACKAGE WILL BE REGISTERED
# dims = 1
# data1 = Float32.([1.0 2.0; 3.0 4.0])
# data2 = Float32.([5.0 6.0; 7.0 8.0])
# data_t1, min_data, max_data = TransformData(data1; dims = dims)
# data_t2 = TransformData(data2; dims = dims, scale = true, min_data = min_data, max_data = max_data)
# data_scb1 = TransformDataBack(data_t1, min_data = min_data, max_data = max_data, dims = dims)
# data_scb2 = TransformDataBack(data_t2, min_data = min_data, max_data = max_data, dims = dims)