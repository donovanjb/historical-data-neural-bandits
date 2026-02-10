import numpy as np
from contextual_bandit import ContextualBandit
import pandas as pd


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
        context_data, arms_data, rewards_data = bandit.get_data()
        for t in range(len(context_data)):
            x = context_data[t]
            a = arms_data[t]
            r = rewards_data[t]
            A[a] += np.outer(x, x)
            b[a] += r * x
        
        for t in range(trials):
            x = bandit.sample_context(bandit.online_distribution)
            UCB = compute_ucb(x, A, b, alpha)
            idx = np.argmax(UCB)
            _, r = bandit.pull_arm(idx, x)

            A[idx] += np.outer(x, x)
            b[idx] += r * x
            regret[t] = bandit.compute_regret(idx, context=x)

    elif history == "artificial replay":
        context_data, arms_data, rewards_data = bandit.get_data()
        historical_pulls = np.bincount(arms_data, minlength=K)
        artificial_pulls = np.zeros(K, dtype=int)

        t = 0
        x = bandit.sample_context(bandit.online_distribution)

        while t < trials:
            UCB = compute_ucb(x, A, b, alpha)
            idx = np.argmax(UCB)

            # Check if we can use historical data for this arm
            if artificial_pulls[idx] < historical_pulls[idx]:
                x_hist, r = bandit.sample_data(idx)
                A[idx] += np.outer(x_hist, x_hist)
                b[idx] += r * x_hist
                artificial_pulls[idx] += 1
                continue

            else:
                _, r = bandit.pull_arm(idx, x)
                A[idx] += np.outer(x, x)
                b[idx] += r * x
                regret[t] = bandit.compute_regret(idx, context=x)
                t += 1
                x = bandit.sample_context(bandit.online_distribution)

    else: # no history - standard online LinUCB
        for t in range(trials):
            x = bandit.sample_context(bandit.online_distribution)
            UCB = compute_ucb(x, A, b, alpha)
            idx = np.argmax(UCB)
            _, r = bandit.pull_arm(idx, x)

            A[idx] += np.outer(x, x)
            b[idx] += r * x
            regret[t] = bandit.compute_regret(idx, context=x)
    
    return regret, artificial_pulls if history == "artificial replay" else None

def linucb_matching_context(bandit, alpha, trials):
    """
    LinUCB artificial replay algorithm for contextual bandits.
    Only pulls from historical data if there is a matching context and arm in the history, otherwise pulls from the environment.
    
    Parameters
    ----------
    bandit : ContextualBandit
        The bandit environment
    alpha : float
        Exploration parameter
    trials : int
        Number of trials to run
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

    context_data, arms_data, rewards_data = bandit.get_data()

    df = pd.DataFrame({
        'context': list(map(tuple, context_data)),
        'arm': arms_data,
        'reward': rewards_data
    })
    
    rewards_data = np.array(rewards_data)

    t = 0
    artificial_pulls = np.zeros(K, dtype=int)
    x = bandit.sample_context(bandit.online_distribution)

    while t < trials:
        UCB = compute_ucb(x, A, b, alpha)
        idx = np.argmax(UCB)

        # use historical data if there is matching context and arm in the history
        matches = df[(df['arm'] == idx) & (df['context'] == tuple(x))]
        if not matches.empty:
            r = matches['reward'].iloc[0]
            A[idx] += np.outer(x, x)
            b[idx] += r * x
            df = df.drop(matches.index[0])  # Remove the used data point
            artificial_pulls[idx] += 1
            continue
        else:
            _, r = bandit.pull_arm(idx, x)
            A[idx] += np.outer(x, x)
            b[idx] += r * x
            regret[t] = bandit.compute_regret(idx, context=x)
            t += 1
            x = bandit.sample_context(bandit.online_distribution)

    return regret, artificial_pulls