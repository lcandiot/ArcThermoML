using JLD2

"""

    LoadData(; data_path)

Load the data that has already been generated and stored at the `data_path` location.
"""
function LoadData(; data_path :: String)
    
    # Introduce to scope
    input  = Float32[]
    output = Float32[]

    # Open read and return
    jldopen(data_path, "r") do file
        input  = file["data/input"]
        output = file["data/output"]
    end

    return input, output

end

# Run
# input, output = LoadData(data_path = "./data/IgneousData.jld2")