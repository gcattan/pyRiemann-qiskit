import cvxpy as cvx


def fro_mean_convex(covmats):
    k, _, n = covmats.shape
    X_mean = cvx.Variable((n, n), symmetric=True)

    def dist(i):
        return cvx.norm(X_mean - covmats[i], "fro")

    expression = sum(dist(i) for i in range(k))
    constraints = [X_mean >> 0]
    prob = cvx.Problem(cvx.Minimize(expression), constraints)
    prob.solve(verbose=True)
    return X_mean.value
