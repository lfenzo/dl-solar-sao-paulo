using CSV
using Dates
using FileIO
using DataFrames
using Statistics
using ShiftedArrays


const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")


# abstract baseline
abstract type Baseline end


"""
    HourlyMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the hours in provided data.
"""
mutable struct HourlyMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
HourlyMean(; site_specific) = HourlyMean(nothing, site_specific)

function fit!(baseline::HourlyMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :hour] : [:hour]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::HourlyMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :hour] : [:hour]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    DailyMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the days of the year in `X`.
"""
mutable struct DailyMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
DailyMean(; site_specific) = DailyMean(nothing, site_specific)

function fit!(baseline::DailyMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :doy] : [:doy]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::DailyMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :doy] : [:doy]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    DateTimeMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the days of the year in `X`.
"""
mutable struct DateTimeMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
DateTimeMean(; site_specific) = DateTimeMean(nothing, site_specific)

function fit!(baseline::DateTimeMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :doy, :hour] : [:doy, :hour]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::DateTimeMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :doy, :hour] : [:doy, :hour]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    PreviousDay()

The value for the target in t+1 will be the same value for the target in the same hour
of the previous day.
"""
mutable struct PreviousDay <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
end
PreviousDay() = PreviousDay(nothing)

function fit!(baseline::PreviousDay, X::DataFrame) :: Nothing

    # similar dataframe, but copying no rows from it
    baseline.lookup_table = similar(X, 0)

    # reconstructing the missing 
    for site in groupby(X, :id)
        start_date = DateTime(minimum(site.year)) + Day(minimum(site.doy) - 1)
        end_date = DateTime(maximum(site.year)) + Day(maximum(site.doy) - 1)

        # construct a date array that will be used to fill the gaps in each of the sites
        datearray = start_date:Hour(1):end_date

        cols = [:id, :hour, :doy, :year]
        local_df = similar(select(baseline.lookup_table, cols), 0)

        local_df = DataFrame(
            id = [site[1, :id] for _ in 1:length(datearray)],
            hour = hour.(datearray),
            doy = dayofyear.(datearray),
            year = year.(datearray),
        )

        filled_df = leftjoin(local_df, X, on = cols)

        # this sort operation is here to make sure that the correct chronological order
        # is kept after the join. The fact that the rows in the resulting dataframe get
        # somehow reordered makes absolutely no sense to me
        sort!(filled_df, [:id, :year, :doy, :hour])

        # now that every day worth of observations have the exact same number of hours
        # we can safely shift the values for the radiation
        filled_df[!, :baseline] = ShiftedArrays.lag(filled_df.next_radiation, 24)

        append!(baseline.lookup_table, filled_df, cols = :union)
    end

    dropmissing!(baseline.lookup_table)

    return nothing
end

function predict(baseline::PreviousDay, X::DataFrame)
    return baseline.lookup_table
end


function main()
    xtrain, ytrain = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xtrain", "ytrain")

    baseline = PreviousDay()
    fit!(baseline, ytrain)  # yes, we are using ytrain, not xtrain...

    return predict(baseline, ytrain)
end

main()
