using CSV
using MLJ
using BSON
using Dates
using Downloads
using FileIO
using Impute
using DataFrames
using ShiftedArrays
using BenchmarkTools
using Random

using StatsBase: sample

const DATA_DIR = joinpath(pwd(), "..", "data")
const PROCESSED_DATA_DIR = joinpath(DATA_DIR, "processed")
const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")


"""
   download_raw_data(year_range::UnitRange{Int64}; force_download::Bool = false)

Downloads and unzips all files from INMET with respect to a given `year_range`
"""
function download_raw_data(year_range::UnitRange{Int64}; force_download::Bool = false)

    @assert length(year_range) > 0

    baseurl = "https://portal.inmet.gov.br/uploads/dadoshistoricos"
    extract_dir = joinpath(DATA_DIR, "raw")
    zip_dir = joinpath(DATA_DIR, "zip")

    !isdir(extract_dir) && mkdir(extract_dir)
    !isdir(zip_dir) && mkdir(zip_dir)

    Threads.@threads for year in year_range
        if force_download || !("$year.zip" in readdir("../data/zip"))
            @info "Downloading $year ... "
            Downloads.download("$baseurl/$year.zip", "$zip_dir/$year.zip")
        else
            @info "Skipping $year ..."
        end
    end

    # TODO regex missing here
#    run(`unzip -jo ../data/zip/\*.zip "*/INMET_SE_SP_*" -d ../data/raw/`)
#    run(`unzip -jo ../data/zip/\*.zip "INMET_SE_SP_*" -d ../data/raw/`)
end


function process_datafile_header(file::AbstractString) :: Dict

    df = CSV.read(
        file, DataFrame;
        header = ["names", "value"],
        skipto = 4,
        limit = 4, # number of lines to read (from the start set in `skipto`)
        decimal = ',',
        delim = ';',
    )

    header_info = Dict(
        "id" => df[1, "value"],
        "lat" => parse(Float64, replace(df[2, "value"], "," => ".")),
        "lon" => parse(Float64, replace(df[3, "value"], "," => ".")),
        "alt" => parse(Float64, replace(df[4, "value"], "," => ".")),
    )
    
    return header_info
end


"""
    gather_files(from::AbstractString) :: DataFrame

Loads all files in `from` and concatenate its contents in a single dataframe.
"""
function gather_files(; from::AbstractString) :: DataFrame

    all_dfs = Vector{DataFrame}()

    column_names = [
        "date", "hour", "precip", "pressure", "pressure_max", "pressure_min",
        "radiation", "air_temp", "dew_point", "air_temp_max", "air_temp_min",
        "dew_point_max", "dew_point_min", "humidity_max", "humidity_min", "humidity",
        "wind_dir", "wind_burst", "wind_speed",
        "extra_semicolon", # will be deleted later (extra delimiter in end of each row)
    ]

    all = DataFrame()

    for file in readdir(from)
        df = CSV.read(
            joinpath(from, file), DataFrame;
            typemap = Dict(Date => String, Time => String),
            skipto = 10,
            header = column_names,
            delim = ';',
            decimal = ','
        )

        select!(df, Not(:extra_semicolon))

        # standardize date-time rcord formats
        transform!(df, :date => ByRow(s -> replace(s, "/" => "-")) => :date)
        transform!(df, :hour => ByRow(s -> replace(s, " UTC" => "", ":" => "")) => :hour)

        # convert date and time to their respective types
        df.date = Date.(df.date, "yyyy-mm-dd")
        df.hour = Time.(df.hour, "HHMM")

        # convert Type to Union{Type, Missing}
        allowmissing!(df)

        # adding header infor
        header_info = process_datafile_header(joinpath(from, file))
        df.id .= header_info["id"]
        df.lat .= header_info["lat"]
        df.lon .= header_info["lon"]
        df.alt .= header_info["alt"]

        append!(all, df)
    end

    return all
end


function preprocess(df::DataFrame; interpolate::Bool = false) :: DataFrame

    ignore_features = ["doy", "hour", "id", "lat", "lon", "alt", "year"]

    # removing the trailing "A" in the IDs of each of the stations
    transform!(df, :id => ByRow(id -> parse(Float32, id[2:end])) => :id)
    
    # shifting target variable 1 step in time into the future (advanding in 1h)
    transform!(groupby(df, :id), :radiation .=> ShiftedArrays.lead .=> :next_radiation)

    # removing observations out of daytime
    filter!(row -> Time(7) <= row["hour"] <= Time(19), df)

    df.doy = dayofyear.(df.date)
    df.hour = hour.(df.hour)
    df.year = year.(df.date)
    select!(df, Not(:date))

    # converting -9999.0 to `missing`
    df = Impute.declaremissings(df; values = [-9999.0])

    # converting absurd values to `missing`s (values that should not be smaller than zero)
    zero_absurd_features = filter(feature -> !(feature in ignore_features), names(df))

    for feature in zero_absurd_features
        transform!(df, feature => ByRow(r -> r isa Missing || r < 0 ? missing : r), renamecols = false)
    end

    # imputing missing values using KNN with samples in other years in the same doy
    features_to_interpolate = filter(f -> !(f in ignore_features), names(df))

    if interpolate
        for group in groupby(df, [:id, :doy, :hour])
            nrow(group) > 2 && Impute.substitute!(group)
        end
    end

    # deleting remaining rows with at leat one missing value
    dropmissing!(df)
    disallowmissing!(df)

    #
    # feature-specific odd value handling
    #

    filter!(row -> 800 <= row.pressure <= 1100, df)
    filter!(row -> 800 <= row.pressure_max <= 1100, df)
    filter!(row -> 800 <= row.pressure_min <= 1100, df)

    filter!(row -> row.precip <= 5, df)

    filter!(row -> row[:radiation] < 4000, df)

    return df
end


function prepare_processing_datasets(df::DataFrame; sample_thresh::Integer, train_frac::Float64, valid_frac::Float64, target::Symbol)

    # selecting the stations with at least 'sample_thresh' samples for training
    nrows_by_id = combine(groupby(df, :id), nrow)
    valid_ids = filter(g -> g[:nrow] >= sample_thresh, nrows_by_id)[:, :id]
    filter!(r -> r[:id] in valid_ids, df)

    train = DataFrame()
    valid = DataFrame()
    test = DataFrame()

    # converting all features to Float64 (necessary for standardization with MLJ)
    for feature in filter(n -> n != "id", names(df))
        df[!, feature] = convert.(Float32, df[!, feature])
    end

    # selecting partitioning data between the training, test and validation sets
    for group in groupby(df, :id)
        sort!(group, [:year, :doy, :hour])

        group_train, group_valid, group_test = MLJ.partition(group, train_frac, valid_frac) 

        append!(train, group_train)
        append!(valid, group_valid)
        append!(test, group_test)
    end

    # splitting between data and target variables
    data_cols = filter(n -> !(Symbol(n) in [target, :id]), names(df))
    target_cols = [:id, :hour, :doy, :year, target]  # NOTE the order of the target

    xtrain = select(train, data_cols)
    ytrain = select(train, target_cols)

    xvalid = select(valid, data_cols)
    yvalid = select(valid, target_cols)

    xtest = select(test, data_cols)
    ytest = select(test, target_cols)

    # standardizers fitted only on train data...
    input_std_mach = fit!(machine(Standardizer(), xtrain))
    output_std_mach = fit!(machine(Standardizer(), ytrain[:, target]))

    norm_xtrain = MLJ.transform(input_std_mach, xtrain)
    norm_ytrain = copy(ytrain)
    norm_ytrain[!, "norm_$target"] = MLJ.transform(output_std_mach, ytrain[:, target])

    norm_xvalid = MLJ.transform(input_std_mach, xvalid)
    norm_yvalid = copy(yvalid)
    norm_yvalid[!, "norm_$target"] = MLJ.transform(output_std_mach, yvalid[:, target])

    norm_xtest = MLJ.transform(input_std_mach, xtest)
    norm_ytest = copy(ytest)
    norm_ytest[!, "norm_$target"] = MLJ.transform(output_std_mach, ytest[:, target])

    artifact_dict = Dict(
        "input_scaler" => input_std_mach,
        "output_scaler" => output_std_mach,
        "xtrain" => norm_xtrain,
        "ytrain" => norm_ytrain,
        "xvalid" => norm_xvalid,
        "yvalid" => norm_yvalid,
        "xtest" => norm_xtest, 
        "ytest" => norm_ytest,
    )

    # converting sets to Float32 (best suited for GPU)
    for (key, artifact) in artifact_dict
        artifact_dict[key] = occursin(r"x|y", key) ? convert.(Float32, artifact) : artifact
    end

    return artifact_dict
end


function main()

    # creating the necessary firectoies
    for dir in [ARTIFACT_DIR, DATA_DIR, PROCESSED_DATA_DIR]
        !isdir(dir) && mkdir(dir)
    end

    #
    # Download data
    #

    @info "Initializing raw data download"
    download_raw_data(2010:2021, force_download = false)

    @info "Concatenating all datafiles into one dataframe"
    df = gather_files(from = joinpath(DATA_DIR, "raw"))
    CSV.write(joinpath(DATA_DIR, "processed", "concatenated_data.csv"), df)

    #
    # Data Pre-processing
    #

    df = CSV.read(joinpath(DATA_DIR, "processed", "concatenated_data.csv"), DataFrame)

    @info "Performing preprocessing operations"
    df = preprocess(df, interpolate = true)
    CSV.write(joinpath(DATA_DIR, "processed", "ready_dataframe.csv"), df)

    #
    # Train, validation and test set preparation
    #

    df = CSV.read(joinpath(DATA_DIR, "processed", "ready_dataframe.csv"), DataFrame)

    @info "Splitting data into Training, Validation and Test sets"
    artifact_dict = prepare_processing_datasets(df;
        sample_thresh = 10_000,
        train_frac = 0.7,
        valid_frac = 0.15,
        target = :next_radiation,
    )

    @info "Saving produced artifacts..."
    FileIO.save(joinpath(ARTIFACT_DIR, "artifacts.jld2"), artifact_dict) 

    return nothing
end


main()
