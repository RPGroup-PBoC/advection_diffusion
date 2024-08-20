data {
    int N; // dataset size
    real l[N]; // length
    real shortest;
    real longest;
}
parameters {
    real log_lambda;
}
model {
    // Priors
    log_lambda ~ uniform(log(shortest), log(longest));
    // Likelihood
    for (i in 1:N) {
        l[i] ~ exponential(1 / exp(log_lambda));
    }
}
generated quantities {
    // Take the exponent of log_lambda
    real lambda;
    lambda = exp(log_lambda);
}