functions {
    real contraction_theor(real x, real m, real b) {
        return m * x + b;
    }
}
data {
    int N; // dataset size
    real x[N]; // distance from network center
    real y[N]; // contraction rate
    real slope_mean;
    real slope_std;
    real sigma_slope_mean;
    real sigma_slope_std;
    real offset_mean;
    real offset_std;
}
parameters {
    real<lower=0> slope;
    real<lower=0> sigma_slope;
    real<upper=0> offset;
}
model {
    // Priors
    slope ~ normal(slope_mean, slope_std);
    sigma_slope ~ normal(sigma_slope_mean, sigma_slope_std);
    offset ~ normal(offset_mean, offset_std);
    // Likelihood
    for (i in 1:N) {
        y[i] ~ normal(contraction_theor(x[i], slope, offset), sigma_slope);
    }
}
