using CairoMakie

# Create a simple cartoon showing the geometry of log-convexity
fig = Figure()

ax = Axis(fig[1,1],
    width = 200,
    height = 200,
    xlabel = "x",
    ylabel = "y",
    )

x = range(-1, 1, length=100)
y_parabola = @. x^2

# Line starting at (1, 1) with negative slope that crosses the parabola
# Line: y = 1 - 0.2*(x-1)
y_line = @. 1 - (x + 1)

# Plot the parabola (mixture model)
lines!(ax, x, y_parabola, 
    color = :black, 
    linewidth = 3, 
    label = "Mixture (Convex Curve)")

# Plot the line (single exponential fit)
lines!(ax, x, y_line, 
    color = :red, 
    linewidth = 3, 
    linestyle = :dash,
    label = "Single Exponential (Line)")

# Mark the starting point
scatter!(ax, [-1], [1], 
    color = :black, 
    markersize = 12,
    label = "Starting Point")
scatter!(ax, [0], [0], 
    color = :black, 
    markersize = 12,
    label = "Starting Point")



Label(fig[1, 1, TopLeft()], "A)",
    fontsize = 18,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :left)
resize_to_layout!(fig)
save("cartoon_figure.pdf", fig)
fig
