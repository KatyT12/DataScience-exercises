import numpy as np
import sklearn.linear_model


def fit_mrrw(x):
    X = np.column_stack([x[:-1]])
    model = sklearn.linear_model.LinearRegression()
    model.fit(X,x[1:])
    a, b = model.intercept_, *model.coef_
    λ = b
    μ = a/(1 - λ)  
    
    σ = np.sqrt(np.mean(np.square(np.array(x[1:]) - model.predict(X))))
    return (μ, λ, σ)


# probabilities of state change over time but this is fine in a Markov chain
def fit_climate0(temp, t):
    # Input: temp and t are numpy vectors
    X = np.column_stack([t-2000])
    model = sklearn.linear_model.LinearRegression().fit(X, temp)
    α, γ = model.intercept_, *model.coef_
    σ = np.sqrt(np.mean(np.square(temp - model.predict(X))))
    return (α, γ, σ)


def fit_climate1(temp, t):
    temp_trunc = temp[1:]
    X = np.column_stack([t[1:]-2000,temp[:-1]])
    model = sklearn.linear_model.LinearRegression().fit(X, temp_trunc)
    α, γ, λ = model.intercept_, *model.coef_
    σ = np.sqrt(np.mean(np.square(temp_trunc - model.predict(X))))
    return (α,γ,λ,σ)


# Is this model adequate compared to the simpler model?
def test_climate0(temp, t):
    num = 20000
    _, _, λ, _ = fit_climate1(temp, t)

    αhat, γhat, σhat = fit_climate0(temp,t)

    def rtemp_star():
        return np.random.normal(loc=αhat + γhat*(t-2000),scale=σhat)

    def readout(data):
        _,_,λ_,_ = fit_climate1(data,t)
        return λ_

    t_ = np.array([readout(rtemp_star()) for _ in range(num)])
    
    return 2*min(np.mean(λ <= t_),np.mean(λ >= t_))

def graphrw_stationary(adj):
    P = np.array(adj)/(np.sum(adj, axis=1)[:, np.newaxis])

    # Set up equations
    A = (P - np.eye(len(adj))).T
    A = np.concatenate((A, [np.ones(len(adj))]), axis=0)
    B = np.concatenate((np.zeros(len(adj)),np.array([1])), axis=0).T
    # Solve
    π = np.linalg.lstsq(A, B)[0]

    return π

P = [[0,1,0,0],[0,0,1,1],[0,1,0,1],[0,1,1,0]]
print(graphrw_stationary(P))

