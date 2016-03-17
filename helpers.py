def computeNumericalGradient(N, x, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for i in range(len(paramsInitial)):
        # set pertubation vector
        perturb[i] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(x, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(x, y)

        # compute numerical gradient
        numgrad[i] = (loss2 - loss1) / (2*e)

        # return the value we changed back to zero
        perturb[i] = 0

    # return params to original value
    N.setParams(paramsInitial)

    return numgrad
