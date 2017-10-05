def c_index(risk, T, C):
    """Calculate concordance index to evaluate model prediction.

    C-index calulates the fraction of all pairs of subjects whose predicted
    survival times are correctly ordered among all subjects that can actually
    be ordered, i.e. both of them are uncensored or the uncensored time of
    one is smaller than the censored survival time of the other.

    Parameters
    ----------
    risk: numpy.ndarray
       m sized array of predicted risk (do not confuse with predicted survival time)
    T: numpy.ndarray
       m sized vector of time of death or last follow up
    C: numpy.ndarray
       m sized vector of censored status (do not confuse with observed status)

    Returns
    -------
    A value between 0 and 1 indicating concordance index.
    """
    n_orderable = 0.0
    score = 0.0
    for i in range(len(T)):
        for j in range(i+1, len(T)):
            if(C[i] == 0 and C[j] == 0):
                n_orderable = n_orderable + 1
                if(T[i] > T[j]):
                    if(risk[j] > risk[i]):
                        score = score + 1
                elif(T[j] > T[i]):
                    if(risk[i] > risk[j]):
                        score = score + 1
                else:
                    if(risk[i] == risk[j]):
                        score = score + 1
            elif(C[i] == 1 and C[j] == 0):
                if(T[i] >= T[j]):
                    n_orderable = n_orderable + 1
                    if(T[i] > T[j]):
                        if(risk[j] > risk[i]):
                            score = score + 1
            elif(C[j] == 1 and C[i] == 0):
                if(T[j] >= T[i]):
                    n_orderable = n_orderable + 1
                    if(T[j] > T[i]):
                        if(risk[i] > risk[j]):
                            score = score + 1

    return score / n_orderable
