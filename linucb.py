import numpy as np
from contextual_bandit import ContextualBandit


def compute_ucb(x, A, b, alpha):
    """
    Compute Upper Confidence Bound for each arm.
    
    Parameters
    ----------
    x : np.ndarray
        Context vector
    A : list of np.ndarray
        Design matrices for each arm
    b : list of np.ndarray
        Reward vectors for each arm
    alpha : float
        Exploration parameter
        
    Returns
    -------
    np.ndarray
        UCB values for each arm
    """
    UCB = []
    for i in range(len(b)):
        Ainv = np.linalg.inv(A[i])
        Rhat = x @ Ainv @ b[i]
        Uhat = alpha * np.sqrt(x.T @ Ainv @ x)
        UCB.append(Rhat + Uhat)
    return np.array(UCB)


def linucb(bandit, alpha, trials, history = None):
    """
    LinUCB algorithm for contextual bandits.
    
    Parameters
    ----------
    bandit : ContextualBandit
        The bandit environment
    alpha : float
        Exploration parameter
    trials : int
        Number of trials to run
    history : String
        how to utilize historical data ("full start", "artificial replay", None)
    Returns
    -------
    np.ndarray
        Regret at each trial
    """
    K = bandit.K
    D = bandit.d
    A = [np.eye(D + 1) for k in range(K)]
    b = [np.zeros(D + 1) for k in range(K)]
    regret = np.zeros(trials)

    if history == "full start":
        context_data, arms_data, rewards_data = bandit.get_history()
        for t in range(len(context_data)):
            x = context_data[t]
            a = arms_data[t]
            r = rewards_data[t]
            A[a] += np.outer(x, x)
            b[a] += r * x

    if history == "artificial replay":
        context_data, arms_data, rewards_data = bandit.get_history()
        historical_pulls = np.bincount(arms_data, minlength=K)
        artificial_pulls = np.zeros(K, dtype=int)

        t = 0

        while t < trials:
            x = bandit.sample_context()
            UCB = compute_ucb(x, A, b, alpha)
            idx = np.argmax(UCB)

            # Check if we can use historical data for this arm
            if artificial_pulls[idx] < historical_pulls[idx]:
                x_hist, r, _ = bandit.sample_data(idx)
                A[idx] += np.outer(x_hist, x_hist)
                b[idx] += r * x_hist
                artificial_pulls[idx] += 1
                # regret[t] = bandit.compute_regret(idx)

            else:
                _, r = bandit.pull_arm(idx, x)
                A[idx] += np.outer(x, x)
                b[idx] += r * x
                regret[t] = bandit.compute_regret(idx, context=x)

                t += 1

    else:
        for t in range(trials):
            x = bandit.sample_context()
            UCB = compute_ucb(x, A, b, alpha)
            idx = np.argmax(UCB)
            _, r = bandit.pull_arm(idx, x)

            A[idx] += np.outer(x, x)
            b[idx] += r * x
            regret[t] = bandit.compute_regret(idx, context=x)
    
    print(np.sum(regret))
    return regret

