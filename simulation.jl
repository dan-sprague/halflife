using Turing,CairoMakie,LinearAlgebra,MCMCChains 
using LaTeXStrings

halflives = [0.9,0.1]
weights = [0.5,0.5]


Exponential.(halflives)

N₀ = 1.0

N(θ,t;N₀ = 1.0) = N₀ * 0.5^(t / θ)


T = 0:0.1:3


n1 = N.(Ref(halflives[1]),T)
n2 = N.(Ref(halflives[2]),T)

acc = zeros(size(T,1))
for component in 1:2
    for (i,t) ∈ enumerate(T)
        acc[i] += weights[component] * N(halflives[component],t)
    end
end

acc 

Cat = Turing.Categorical


fig = Figure()
ax = Axis(fig[1,1],xlabel = "t",ylabel=L"N(t_{1/2},t)",title="Weighted Mixture of Decays")
lines!(ax,T,n1, label = latexstring("$(halflives[1])"),linestyle=:dash)
lines!(ax,T,n2,label = latexstring("$(halflives[2])"),linestyle=:dash)
lines!(ax,T,acc,label = L"\sum_{i ∈ 1:3}{w_iN_i(t)}")

Legend(fig[1,2],ax,L"t_{1/2}")


# ax = Axis(fig[1,3],xlabel = "t",ylabel=L"N(t_{1/2},t)")
# lines!(ax,T,acc,label = "Weighted Mixture")
rowsize!(fig.layout,1,Aspect(1,1.0))
resize_to_layout!(fig)
fig


using Turing, CairoMakie, LinearAlgebra, MCMCChains
using LaTeXStrings

halflives = [0.5, 5.0] 
weights = [0.8, 0.2] 
k(th) = log(2) / th
rates = k.(halflives)

t = 0:0.05:10

term_fast = @. weights[1] * exp(-rates[1]*t)
term_stable = @. weights[2] * exp(-rates[2]*t)

y_mixture = term_fast .+ term_stable

avg_rate = sum(weights .* rates)
y_avg_rate = @. 1.0 * exp(-avg_rate * t)

frac_stable = term_stable ./ y_mixture
frac_unstable = term_fast ./ y_mixture

fig = Figure()

ax1 = Axis(fig[1,1], 
    xlabel="Time (t)", 
    ylabel="Predicted Fraction Remaining N(t)", 
    yticklabelcolor = :black)

l1 = lines!(ax1, t, y_mixture, label="Mixture", linewidth=3,)
l2 = lines!(ax1, t, y_avg_rate, label="Single Exp Fit", linewidth=3,)

ax2 = Axis(fig[1,1], 
    ylabel="Proportion of Population",
    yaxisposition = :right,
    )

hidespines!(ax2, :l, :t, :b)
hidexdecorations!(ax2)

l3 = lines!(ax2, t, frac_stable, label="Proportion Stable", linewidth=3, linestyle=:dot,
color = 3)
l4 = lines!(ax2, t, frac_unstable, label="Proportion Unstable", linewidth=3, linestyle=:dot,
color = RGBf(0.5,0.2,0.8))

legend_grid = GridLayout(fig[1,2], alignmode = Outside())
Legend(legend_grid[1,1], [l1, l2], ["Mixture of Rates", "Single Average Rate"], "Decay Models",
framevisible=false)
Legend(legend_grid[2,1], 
    [l3, l4], 
    [
        rich("Stable t", subscript("1/2"), " = 5.0"),
        rich("Unstable t", subscript("1/2"), " = 0.5")
    ], 
    "Population Composition",
    framevisible=false
)

rowsize!(fig.layout, 1, Aspect(1, 1.0))

resize_to_layout!(fig)

save("figure.png",fig)




using CairoMakie, Optim

# --- 1. Setup Ground Truth ---
halflives = [0.5, 5.0]
weights = [0.8, 0.2]
rates = log(2) ./ halflives
t = 0:0.05:8.0

# The Truth
y_mix = @. weights[1] * exp(-rates[1]*t) + weights[2] * exp(-rates[2]*t)

# --- 2. Find the "Best" Single Fit (Least Squares) ---
# We numerically find the k that minimizes sum of squared errors
sq_error(k) = sum((y_mix .- exp.(-k .* t)).^2)
res = optimize(sq_error, 0.0, 5.0) 
k_best = res.minimizer

y_fit = exp.(-k_best .* t)

# --- 3. Plotting ---
fig = Figure(size = (800, 600))

# Upper Plot: The Curves (They look close!)
ax1 = Axis(fig[1,1], title="Visual Comparison", ylabel="N(t)")
lines!(ax1, t, y_mix, color=:blue, label="True Mixture", linewidth=3)
lines!(ax1, t, y_fit, color=:red, linestyle=:dash, label="Best Single Fit", linewidth=3)
axislegend(ax1)

# Lower Plot: The Difference (The Truth Revealed)
# This shows exactly where the fit is higher or lower
ax2 = Axis(fig[2,1], title="Difference (Fit - Truth)", ylabel="Error", xlabel="Time")
diff = y_fit .- y_mix

lines!(ax2, t, diff, color=:black, linewidth=2)
hlines!(ax2, [0.0], color=:gray, linestyle=:dot) # Zero line

# Annotate the phases
text!(ax2, 0.5, 0.05, text="Phase 1: Fit > Mix\n(Fit is too slow)", color=:red, align=(:left, :center))
text!(ax2, 5.0, -0.05, text="Phase 2: Fit < Mix\n(Fit is too fast)", color=:blue, align=(:center, :top))

rowsize!(fig.layout, 1, Relative(2/3))
fig



using CairoMakie, Optim, LaTeXStrings


# --- RE-RUN THIS BLOCK FIRST ---
t = 0:0.05:8.0  # Length 161

# Re-calculate everything so dimensions match 't'
y_mix = @. weights[1] * exp(-rates[1]*t) + weights[2] * exp(-rates[2]*t)

# Least Squares Fit
sq_error(k) = sum((y_mix .- exp.(-k .* t)).^2)
res = optimize(sq_error, 0.0, 5.0) 
k_best = res.minimizer
y_fit = exp.(-k_best .* t)

# Tangent
avg_rate = sum(weights .* rates)
y_tangent = @. 1.0 * exp(-avg_rate * t)

# Composition (The source of your error)
n_unstable = @. weights[1] * exp(-rates[1]*t)
n_stable   = @. weights[2] * exp(-rates[2]*t)
frac_unstable = n_unstable ./ y_mix
frac_stable   = n_stable   ./ y_mix

# 1. Setup Data
halflives = [0.5, 5.0]
weights = [0.8, 0.2]
rates = log(2) ./ halflives
t = 0:0.05:8.0

# Mixture Model
y_mix = @. weights[1] * exp(-rates[1]*t) + weights[2] * exp(-rates[2]*t)

# 2. Models
# A. Average Rate (The Tangent at t=0) - Jensen's Lower Bound
avg_rate = sum(weights .* rates)
y_tangent = @. 1.0 * exp(-avg_rate * t)

# B. Best Fit (The Secant) - Least Squares
sq_error(k) = sum((y_mix .- exp.(-k .* t)).^2)
res = optimize(sq_error, 0.0, 5.0) 
k_fit = res.minimizer
y_fit = exp.(-k_fit .* t)

# 3. Plotting in Log Space
fig = Figure(size = (800, 500))

ax = Axis(fig[1,1], 
    yscale = log10, # <--- THE KEY
    xlabel = "Time", 
    ylabel = "Log N(t)",
    title = "Log-Convexity: Fitting a Line to a Curve",
    yminorticksvisible = true, yminorticks = IntervalsBetween(9))

# Plot Mixture (The Curve)
lines!(ax, t, y_mix, color=:blue, linewidth=3, label="Mixture (Convex)")

# Plot Best Fit (The Secant)
lines!(ax, t, y_fit, color=:red, linestyle=:dash, linewidth=2, label="Least Squares (Secant)")

# Plot Average Rate (The Tangent)
lines!(ax, t, y_tangent, color=:green, linestyle=:dot, linewidth=2, label="Avg of Rates (Tangent)")

# Annotations
text!(ax, 4.0, y_fit[end], text="Straight line must\ncross the curve", color=:red)

axislegend(ax, position=:lb)
fig



using CairoMakie, Optim

# ==========================================
# 1. Unified Data Generation
# ==========================================
halflives = [0.5, 10.0]
weights = [0.8, 0.2]
rates = log(2) ./ halflives
t = 0:0.05:8.0

# A. Ground Truth (The Mixture)
y_mix = @. weights[1] * exp(-rates[1]*t) + weights[2] * exp(-rates[2]*t)

# B. Least Squares Fit (Single Exponential)
sq_error(k) = sum((y_mix .- exp.(-k .* t)).^2)
res = optimize(sq_error, 0.0, 5.0) 
k_best = res.minimizer
y_fit = exp.(-k_best .* t)

# C. Tangent (Average Rate - Jensen's Bound)
avg_rate = sum(weights .* rates)
y_tangent = @. 1.0 * exp(-avg_rate * t)

# D. Population Composition (for Dual Axis)
# Calculate the absolute amount of Stable vs Unstable remaining
n_unstable = @. weights[1] * exp(-rates[1]*t)
n_stable   = @. weights[2] * exp(-rates[2]*t)
# Calculate fractions relative to current total N(t)
frac_unstable = n_unstable ./ y_mix
frac_stable   = n_stable   ./ y_mix

# ==========================================
# 2. Figure Setup
# ==========================================
# Setting resolution to 800x800 ensures the base aspect ratio is 1
fig = Figure(size = (800, 500), fontsize=14)

# ==========================================
# Panel 1: Linear Scale + Composition (Top Left)
# ==========================================
ax1 = Axis(fig[1,1], 
    title = "A. Decay & Composition",
    xlabel = "Time (t)", 
    ylabel = "Fraction Remaining N(t)",
    yticklabelcolor = :black)

# Primary Plot (Decay)
l1 = lines!(ax1, t, y_mix, color=:black, linewidth=3, label="True Mixture")
l2 = lines!(ax1, t, y_fit, color=:red, linestyle=:dash, linewidth=2, label="Single Exp Fit")

# Secondary Axis (Proportions)
ax1_right = Axis(fig[1,1], 
    ylabel = "Population Proportion",
    yaxisposition = :right,
    yticklabelcolor = :purple)
hidespines!(ax1_right, :l, :t, :b)
hidexdecorations!(ax1_right)

# Plot Proportions
l3 = lines!(ax1_right, t, frac_stable, color=:purple, linewidth=3, linestyle=:dot)
l4 = lines!(ax1_right, t, frac_unstable, color=(:purple, 0.5), linewidth=3, linestyle=:dot)

# ==========================================
# Panel 2: Log Scale / Convexity (Top Right)
# ==========================================
ax2 = Axis(fig[2,1], 
    title = "B. Log-Convexity",
    xlabel = "Time (t)", 
    ylabel = "Log N(t)",
    yscale = log10,
    yminorticksvisible = true, yminorticks = IntervalsBetween(9))

lines!(ax2, t, y_mix, color=:black, linewidth=3)
lines!(ax2, t, y_fit, color=:red, linestyle=:dash, linewidth=2)
l5 = lines!(ax2, t, y_tangent, color=:green, linestyle=:dashdot, linewidth=2, label="Tangent (Avg Rate)")

# ==========================================
# Panel 4: Legend Grid (Bottom Right)
# ==========================================
# We create a nested GridLayout for the legends to keep them organized
legend_subgrid = GridLayout(fig[1,2], alignmode = Outside())

# Legend for Models
Legend(fig[2,2], 
    [l1, l2, l5], 
    ["Mixture (Observed)", "Least Squares Fit (Secant)", "Avg Rate (Tangent)"], 
    "Decay Models",
    framevisible=false, titlegap=5,
    tellheight=false)

Legend(legend_subgrid[1,1], 
    [l1, l2], 
    ["Mixture (Observed)", "Least Squares Fit (Secant)"], 
    "Decay Models",
    framevisible=false, titlegap=5,
    tellheight=false)


# Legend for Composition
Legend(legend_subgrid[2,1], 
    [l3, l4], 
    [
        rich("Stable (t", subscript("1/2"), "=5.0)"),
        rich("Unstable (t", subscript("1/2"), "=0.5)")
    ], 
    "Population (Right Axis)",
    framevisible=false, titlegap=5,
    tellheight=false)

# ==========================================
# Layout Adjustments
# ==========================================
# Force columns and rows to be equal to ensure the aspect ratio balance
#colsize!(fig.layout, 1, Aspect(1,1.0))
#colsize!(fig.layout,2,Aspect(1,1.0))
resize_to_layout!(fig)

# Save
save("combined_decay_figure.png", fig)

# Return fig for display in notebook/REPL
fig