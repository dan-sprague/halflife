using Turing, CairoMakie, LinearAlgebra, MCMCChains, Optim
using LaTeXStrings, Printf, PrettyTables

struct DecayModel{T <: Real}
    λ::T
    
    function DecayModel(λ::T) where T <: Real  # Changed from DecayMixtureModel
        @assert λ > 0 "λ must be > 0."
        new{T}(λ)
    end
end

struct DecayMixtureModel{T <: Real}
    w::Vector{T}
    λ::Vector{T}
    
    function DecayMixtureModel(w::Vector{T}, λ::Vector{T}) where T <: Real
        all(w .>= 0) && sum(w) ≈ 1.0 || throw(ArgumentError("w must be non-negative and sum to 1"))
        all(λ .>= 0) || throw(ArgumentError("λ must be non-negative"))
        new{T}(w, λ)
    end
end




function N(t::Real, m::DecayMixtureModel)
    @assert t >= 0
    expectation = 0.0
    for (wᵢ, λᵢ) in zip(m.w, m.λ)
        expectation += wᵢ * exp(-λᵢ * t)
    end
    expectation
end

function N(t::Real,m::DecayModel)
    @assert t >=0

    exp(-m.λ * t)
end

function simulate_decay(T, M::DecayMixtureModel, σ::Real)
    @assert all(T .>= 0) "Time must be ≥ 0"
    @assert σ >= 0 "σ must be non-negative"
    
    observations = similar(T, Float64)
    for (i, t) ∈ enumerate(T)
        observations[i] = N(t, M) * exp(σ * randn())
    end
    observations
end

function neg_log_likelihood(T, Y, w, λ, log_σ)
    σ = exp(log_σ)
    n = length(T)
    
    Y_pred = [sum(w[j] * exp(-λ[j] * t) for j in eachindex(w)) for t in T]
    log_Y = log.(Y)
    log_Y_pred = log.(Y_pred)
    
    nll = n / 2 * log(2 * π) + n / 2 * log(σ^2) + sum((log_Y .- log_Y_pred).^2) / (2 * σ^2)
    return nll
end

function fit_single_decay(T, Y)
    nvar = 2 
    log_Y = log.(Y)
    X = hcat(ones(length(T)), collect(T))
    β = X \ log_Y  
    λ_init = -β[2]
    λ_init = max(0.01, λ_init) 
    
    func = TwiceDifferentiable(
        vars -> begin
            λ = exp(vars[1]) 
            log_σ = vars[2]
            neg_log_likelihood(T, Y, [1.0], [λ], log_σ)
        end,
        [log(λ_init), log(0.1)]; 
        autodiff = :forward,
    )
    
    opt = optimize(func, [log(λ_init), log(0.1)], BFGS(), 
                   Optim.Options(iterations=10000, g_tol=1e-6))
    
    λ_fit = exp(opt.minimizer[1]) 
    σ_fit = exp(opt.minimizer[2])
    
    return λ_fit, σ_fit, opt
end

# Simulate 4 levels of heterogeneity
T = 0.0:0.2:20.0
colors = Makie.wong_colors()

# Level 1: No heterogeneity (equal rates)
model1 = DecayMixtureModel([0.5, 0.5], [0.5, 0.5])
Y_obs1 = simulate_decay(T, model1, 0.1)
opt1 = fit_single_decay(T, Y_obs1)
fitted1 = DecayModel(opt1[1])
Y_pred1 = N.(T, Ref(fitted1))

# Level 2: Low heterogeneity
model2 = DecayMixtureModel([0.5, 0.5], [0.1, 1.0])
Y_obs2 = simulate_decay(T, model2, 0.1)
opt2 = fit_single_decay(T, Y_obs2)
fitted2 = DecayModel(opt2[1])
Y_pred2 = N.(T, Ref(fitted2))

# Level 3: Medium heterogeneity
model3 = DecayMixtureModel([0.5, 0.5], [0.01, 1.0])
Y_obs3 = simulate_decay(T, model3, 0.1)
opt3 = fit_single_decay(T, Y_obs3)
fitted3 = DecayModel(opt3[1])
Y_pred3 = N.(T, Ref(fitted3))

# Create figure
fig = Figure()

# Panel A: Linear scale
ax1 = Axis(fig[1,1],
    xlabel = "Time (t)",
    ylabel = "N(t)",
    width = 200,
    height = 200)

scatter!(ax1, T, Y_obs1, color=(colors[1], 1.0), )
lines!(ax1, T, Y_pred1, color=colors[1], linewidth=3, label="Homogenous")

scatter!(ax1, T, Y_obs2, color=(colors[2], 1.0), )
lines!(ax1, T, Y_pred2, color=colors[2], linewidth=3, label="Low heterogeneity")

scatter!(ax1, T, Y_obs3, color=(colors[3], 1.0), )
lines!(ax1, T, Y_pred3, color=colors[3], linewidth=3, label="High heterogeneity")

Label(fig[1, 1, TopLeft()], "A",
    fontsize = 18,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :left)

# Panel B: Semi-log scale
ax2 = Axis(fig[1,2],
    xlabel = "Time (t)",
    ylabel = "log N(t)",
    yscale = log10,
    width = 200,
    height = 200,
    limits = (nothing, (1e-4, 2)))  # Set y-limits to avoid log(0)

# Filter out very small values for scatter plots
min_val = 1e-6
scatter!(ax2, T[Y_obs1 .> min_val], Y_obs1[Y_obs1 .> min_val], color=(colors[1], 0.3), )
lines!(ax2, T, Y_pred1, color=colors[1], linewidth=3, label="Homogenous")

scatter!(ax2, T[Y_obs2 .> min_val], Y_obs2[Y_obs2 .> min_val], color=(colors[2], 0.3), )
lines!(ax2, T, Y_pred2, color=colors[2], linewidth=3, label="Low heterogeneity")

scatter!(ax2, T[Y_obs3 .> min_val], Y_obs3[Y_obs3 .> min_val], color=(colors[3], 0.3), )
lines!(ax2, T, Y_pred3, color=colors[3], linewidth=3, label="High heterogeneity")


Label(fig[1, 2, TopLeft()], "B",
    fontsize = 18,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :left)

# Panel D: Log-space residuals (linear scale)
ax4 = Axis(fig[1,3],
    xlabel = "Time (t)",
    ylabel = "Log-space Residuals",
    width = 200,
    height = 200)

# Compute log residuals: log(Y_obs) - log(Y_pred)
log_resid1 = log.(Y_obs1) .- log.(Y_pred1)
log_resid2 = log.(Y_obs2) .- log.(Y_pred2)
log_resid3 = log.(Y_obs3) .- log.(Y_pred3)

scatter!(ax4, T, log_resid1, color=(colors[1], 1.0))
hlines!(ax4, [0.0], color=:black, linestyle=:dash, linewidth=3)

scatter!(ax4, T, log_resid2, color=(colors[2], 1.0))

scatter!(ax4, T, log_resid3, color=(colors[3], 1.0))

Label(fig[2, 1, TopLeft()], "D",
    fontsize = 18,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :left)

Label(fig[1, 3, TopLeft()], "C",
    fontsize = 18,
    font = :bold,
    padding = (0, 5, 5, 0),
    halign = :left)

# Panel E: Sign of log-space residuals
ax5 = Axis(fig[2,1],
    xlabel = "Time (t)",
    ylabel = "Sign of Log Residuals",
    width = 200,
    height = 200,
    yticks = ([-1, 0, 1], ["-1", "0", "+1"]))

# Plot sign of residuals with offset for visibility
scatter!(ax5, T, sign.(log_resid1) .+ 0.15, color=(colors[1], 1.0), )
scatter!(ax5, T, sign.(log_resid2), color=(colors[2], 1.0), )
scatter!(ax5, T, sign.(log_resid3) .- 0.15, color=(colors[3], 1.0),)
hlines!(ax5, [0.0], color=:black, linestyle=:dash, linewidth=3)



# Wald-Wolfowitz Runs Test
using HypothesisTests

function runs_test_pvalue(residuals)
    # Convert residuals to binary sequence (above/below median)
    signs = sign.(residuals)
    # Remove zeros
    signs_nonzero = signs[signs .!= 0]
    if length(signs_nonzero) < 2
        return NaN
    end
    # Convert to binary: positive = 1, negative = 0
    binary = signs_nonzero .> 0
    test = WaldWolfowitzTest(binary)
    return pvalue(test)
end

p1 = runs_test_pvalue(log_resid1)
p2 = runs_test_pvalue(log_resid2)
p3 = runs_test_pvalue(log_resid3)

# Panel F: Runs test p-values table
ax6 = Axis(fig[2,2],
    width = 200,
    height = 200,
    limits = (0, 1, 0, 1))
hidedecorations!(ax6)
hidespines!(ax6)

# Create table text with sprintf for alignment
format_pval(p) = p < 0.001 ? "p \u226A 0.001" : @sprintf("%.3f", p)

table_text = @sprintf("""
Wald-Wolfowitz Test\nfor Homogeneity

%-13s  %10s
─────────────────
%-13s  %10s
%-13s  %10s
%-13s  %10s
""", 
"Model", "p-value",
"Hom.", format_pval(p1),
"Low het.", format_pval(p2),
"High het.", format_pval(p3))

text!(ax6, 0.05, 0.5, text=table_text, align=(:left, :center), fontsize=12, space=:data)


Legend(fig[1,4], ax1, framevisible=false)

resize_to_layout!(fig)
fig
save("figure.png",fig)





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