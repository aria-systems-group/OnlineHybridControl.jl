using PosteriorBounds
using GaussianProcesses
using Random
using LinearAlgebra
using Distributions

# Initialize the GP
Random.seed!(11)
# Training data
n = 100;                          #number of training points
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
kern = Poly(0.9, 0.0, 2)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

y_train = y - x[1,:]    # Essentially making training data zero-mean
gp = GP(x, y_train, mZero, kern, logObsNoise)       #Fit the GP, TODO: hyperparameter optimization?  
optimize!(gp, kernbounds=[[-2.0, -0.00001], [2.0, 0.00001]], noise=false)

x_test = [21.0; 0.25]

# The following predict_y function includes the dynamics nosie
μ, σ2 = predict_y(gp, hcat(x_test))

# The following predict_f function does not include the dynamics nosie
μ_no_noise, σ2_no_noise = predict_f(gp, hcat(x_test))

@info "μ: ($μ)"
@info "μ no noise: ($μ_no_noise)"
@assert μ == μ_no_noise
@info "σ2: ($σ2)"
@info "σ2 no noise: ($σ2_no_noise)"
σ2_noise = σ2[1] - σ2_no_noise[1]
@info σ2_noise

#====
Get the parameters for explicit polynomial expansion
====#
alpha = gp.alpha        # alpha vector
K_inv = inv(gp.cK)      # inverse of the cK matrix
c = gp.kernel.c         # bias term in kernel
deg = gp.kernel.deg     # degree of kernel polynomial
σ2 = gp.kernel.σ2       # prior variance term (close to 1)
x_train = gp.x  # same as the training data input x above
