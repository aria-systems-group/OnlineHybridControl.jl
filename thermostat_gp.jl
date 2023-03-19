using PosteriorBounds
using GaussianProcesses
using Random
using LinearAlgebra
using Distributions

# Initialize the GP
Random.seed!(11)
# Training data
n = 10;                          #number of training points
x = [rand(Uniform(19.,24.), 1, n);   rand(1, n)];              #predictors
obs_noise = 0.01
τ = 5.   # min
αe = 8.0e-3
αH = 3.6e-3
Te = 15.0   # °C
Th = 55.0   # °C
f(x) = x[1] + τ*(αe*(Te-x[1]) + αH*(Th-x[1])*x[2]) + 0.1*randn() # standard normal dist. with var σ^2 = 0.01
y = [f(xk) for xk in eachcol(x)]

logObsNoise = log10(obs_noise)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.5, 0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

y_train = y - x[1,:]    # Essentially making training data zero-mean
gp = GP(x, y_train, mZero, kern, logObsNoise)       #Fit the GP, TODO: hyperparameter optimization?  
optimize!(gp, kernbounds=[[0.0, -0.00001], [1.0, 0.00001]])

# Setup minimal GP
gp_ex = PosteriorBounds.PosteriorGP(
    gp.dim,
    gp.nobs,
    gp.x,
    gp.cK,
    Matrix{Float64}(undef, gp.nobs, gp.nobs),
    UpperTriangular(zeros(gp.nobs, gp.nobs)),
    inv(gp.cK),
    gp.alpha,
    PosteriorBounds.SEKernel(gp.kernel.σ2, gp.kernel.ℓ2)
)
PosteriorBounds.compute_factors!(gp_ex)

#==
Point-wise bounds from Quadratic Form (sweeping)
==#

# Test point-valued bounds - first, set the range
x_L = [21.0, 0.2]
x_U = [21.1, 0.21]          # Making this upper bound smaller results in tighter bounds
x_test = [0.25; 0.25]

theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(x, gp_ex.kernel)

# Testing out the point-wise bounds
A, B, C, D = PosteriorBounds.calculate_μ_bound_values(gp_ex.alpha, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x)
res1 = PosteriorBounds.μ_bound_point(x_test, theta_vec, A, B, C, D)

# Testing out the point-wise bounds
Ã, B̃, C̃, D̃ = PosteriorBounds.calculate_μ_bound_values(gp_ex.alpha, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x, upper_bound_flag=true)
res2 = PosteriorBounds.μ_bound_point(x_test, theta_vec, Ã, B̃, C̃, D̃)

# Testing out the point-wise bounds
Aσ, Bσ, Cσ, Dσ = PosteriorBounds.calculate_σ2_bound_values(gp_ex.K_inv, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x)
res_σ2_UB = PosteriorBounds.σ2_bound_point(x_test, theta_vec, Aσ, Bσ, Cσ, Dσ)

# Testing out the point-wise bounds
Aσ_LB, Bσ_LB, Cσ_LB, Dσ_LB = PosteriorBounds.calculate_σ2_bound_values(gp_ex.K_inv, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x, min_flag=true)
res_σ2_LB = PosteriorBounds.σ2_bound_point(x_test, theta_vec, Aσ_LB, Bσ_LB, Cσ_LB, Dσ_LB)

N_samps = 100
μ_bound_norms = Vector{Float64}(undef, N_samps) 
σ2_vals_UB = Vector{Float64}(undef, N_samps) 
σ2_vals_LB = Vector{Float64}(undef, N_samps) 

# Sweep over the points
for i=1:N_samps
    xs = [rand(Uniform(x_L[1], x_U[1])); rand(Uniform(x_L[2], x_U[2])) ]
    μ_lb = PosteriorBounds.μ_bound_point(xs, theta_vec, A, B, C, D) 
    μ_ub = PosteriorBounds.μ_bound_point(xs, theta_vec, Ã, B̃, C̃, D̃)
    σ2_UB = PosteriorBounds.σ2_bound_point(xs, theta_vec, Aσ, Bσ, Cσ, Dσ) 
    σ2_LB = PosteriorBounds.σ2_bound_point(xs, theta_vec, Aσ_LB, Bσ_LB, Cσ_LB, Dσ_LB) 
    μ_bound_norms[i] = norm(μ_lb - μ_ub) 
    σ2_vals_UB[i] = σ2_UB
    σ2_vals_LB[i] = σ2_LB
end

using UnicodePlots
plt = histogram(μ_bound_norms, title="μ Bounds Norms, |LB - UB|")
show(plt)
plt2 = histogram(σ2_vals_UB, title="σ2 Upper Bounds")
show(plt2)
plt3 = histogram(σ2_vals_LB, title="σ2 Lower Bounds")
show(plt3)

#==
Interval Bounds (no sweeping)
==#
theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(x, gp_ex.kernel)
x_interval_L = [21.0, 0.2]
x_interval_U = [22.0, 0.25]         

μ_LB = PosteriorBounds.compute_μ_bounds_bnb(gp_ex, x_interval_L, x_interval_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-3)
μ_UB = PosteriorBounds.compute_μ_bounds_bnb(gp_ex, x_interval_L, x_interval_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-3, max_flag=true)
cK_inv_scaled = PosteriorBounds.scale_cK_inv(gp_ex.cK, 1.0, logObsNoise)      # Needed to scale the problem if prior variance is not 1.0
σ2_UB = PosteriorBounds.compute_σ_bounds(gp_ex, x_interval_L, x_interval_U, theta_vec_train_squared, theta_vec, cK_inv_scaled; bound_epsilon=1e-4, max_iterations=100) 
σ2_LB = PosteriorBounds.compute_σ_bounds(gp_ex, x_interval_L, x_interval_U, theta_vec_train_squared, theta_vec, cK_inv_scaled; bound_epsilon=1e-3, max_iterations=100, min_flag=true)

@info "μ bounds: ($μ_LB, $μ_UB)"
@info "σ2 lower bounds: ($σ2_LB)"
@info "σ2 upper bounds: ($σ2_UB)"
