using CSV
using CairoMakie
using DataFrames


function main()
    df = CSV.read("../data/tmp/preprcessed.csv", DataFrame)

    fig = Figure()
    axs = Axis(fig[1, 1])
    feature = :radiation

    println(names(df))

    df = filter(row -> row.precip < 5, df)

    density!(axs, df[:, feature])

    axs.xlabel = string(feature)
    save("teste.pdf", fig)
end



main()
