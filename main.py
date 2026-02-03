import numpy as np
import matplotlib.pyplot as plt
from contextual_bandit import ContextualBandit
from linucb import linucb

def _sample_uniform_ball(d):
    """
    Sample a vector uniformly from a d-dimensional unit ball.
    
    Parameters
    ----------
    d : int
        Dimension
        
    Returns
    -------
    np.ndarray
        A d-dimensional vector uniformly distributed in the unit ball
    """
    z = np.random.randn(d)
    z = z / np.linalg.norm(z)  # Normalize to unit sphere
    r = np.random.uniform(0, 1) ** (1.0 / d)  # Sample radius for uniform distribution in ball
    return r * z


def create_bandit(reward_fn_id, d=5, K=3, context_radius=1.0, random_seed=None, data_num_steps=None, data_distribution="uniform"):
    """
    Create a contextual bandit with a specified reward function.
    
    Parameters
    ----------
    reward_fn_id : int
        Which reward function to use (1, 2, or 3)
    d : int
        Dimension of context vectors
    K : int
        Number of arms
    context_radius : float
        Radius of the context ball
    random_seed : int, optional
        Random seed for reproducibility
    data_num_steps : int, optional
        Number of offline data samples to generate
    data_distribution : str, optional
        Distribution type for offline data ("uniform", etc.)
    Returns
    -------
    ContextualBandit
        A bandit instance with the selected reward function
    """
    
    if reward_fn_id == 1:
        # Reward function 1: r = 10(x^T a)^2
        # Sample arm parameters uniformly from unit ball
        arm_params = np.array([_sample_uniform_ball(d+1) for _ in range(K)])
        
        def reward_function_1(context, arm):
            a = arm_params[arm]
            return 10 * (np.dot(context, a) ** 2)
        
        reward_fn = reward_function_1
        
    elif reward_fn_id == 2:
        # Reward function 2: r = x^T A^T A x (quadratic form)
        # A is a d x d matrix sampled from N(0, 1) for each arm
        arm_params = [np.random.randn(d+1, d+1) for _ in range(K)]
        
        def reward_function_2(context, arm):
            A = arm_params[arm]
            ATA = A.T @ A
            return np.dot(context, ATA @ context)
        
        reward_fn = reward_function_2
        
    elif reward_fn_id == 3:
        # Reward function 3: r = cos(3x^T a)
        # Sample arm parameters uniformly from unit ball
        arm_params = np.array([_sample_uniform_ball(d+1) for _ in range(K)])
        
        def reward_function_3(context, arm):
            a = arm_params[arm]
            return np.cos(3 * np.dot(context, a))
        
        reward_fn = reward_function_3
        
    else:
        raise ValueError(f"reward_fn_id must be 1, 2, or 3, got {reward_fn_id}")
    
    # Create and return the bandit
    bandit = ContextualBandit(
        d=d,
        K=K,
        reward_function=reward_fn,
        context_radius=context_radius,
        random_seed=random_seed,
        data_num_steps=data_num_steps,
        data_distribution=data_distribution
    )
    
    return bandit


if __name__ == "__main__":
    # Example usage
    print("Creating bandits with different reward functions...\n")
    
    # Test each reward function
    for fn_id in [1, 2, 3]:
        print(f"Reward Function {fn_id}:")
        bandit = create_bandit(reward_fn_id=fn_id, d=5, K=3, random_seed=42)
        
        # Sample a context and pull each arm
        context = bandit.sample_context()
        print(f"  Context: {context[:3]}...")
        
        rewards = bandit.get_all_rewards(context)
        print(f"  Rewards for all arms: {rewards}")
        
        optimal_arm = bandit.get_optimal_arm(context)
        print(f"  Optimal arm: {optimal_arm}\n")
    
    # Experiment: Test LinUCB with artificial replay and full start
    print("\n" + "="*70)
    print("EXPERIMENT: LinUCB with Reward Function 1")
    print("="*70 + "\n")
    
    # test_dims = [5, 10, 20]
    # test_arms = [2, 5, 10]
    # test_N = [0, 500, 1000]
    # test_data_distribution = ["uniform", "lightly_unbalanced", "half_coverage"]
    # test_rew_fn_id = [1, 2, 3]

    test_dims = [20]
    test_arms = [4]
    test_N = [1000]
    test_data_distribution = ["uniform"]
    test_rew_fn_id = [1, 2, 3]

    reward_fn_titles = {
        1: "r = 10(x^T a)^2",
        2: "r = x^T A^T A x",
        3: "r = cos(3x^T a)",
    }

    alpha = 1.0
    trials = 10000

    for d in test_dims:
        for K in test_arms:
            for N in test_N:
                for data_distribution in test_data_distribution:
                    for rew_fn_id in test_rew_fn_id:
                        print("\n" + "-" * 70)
                        print(
                            f"Running: d={d}, K={K}, N={N}, dist={data_distribution}, reward_fn={rew_fn_id}"
                        )

                        bandit_with_data = create_bandit(
                            reward_fn_id=rew_fn_id,
                            d=d,
                            K=K,
                            random_seed=42,
                            data_num_steps=N,
                            data_distribution=data_distribution,
                        )

                        print(f"Generated {bandit_with_data.data_size} offline data samples")
                        print(
                            f"Arm distribution: {np.bincount(bandit_with_data.arms_data, minlength=K)}\n"
                        )

                        regret_artificial_replay = linucb(
                            bandit_with_data, alpha, trials, history="artificial replay"
                        )
                        cum_regret_artificial = np.cumsum(regret_artificial_replay)

                        bandit_for_full_start = create_bandit(
                            reward_fn_id=rew_fn_id,
                            d=d,
                            K=K,
                            random_seed=42,
                            data_num_steps=N,
                            data_distribution=data_distribution,
                        )

                        regret_full_start = linucb(
                            bandit_for_full_start, alpha, trials, history="full start"
                        )
                        cum_regret_full_start = np.cumsum(regret_full_start)

                        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                        fig.suptitle(
                            f"Reward Function {rew_fn_id}: {reward_fn_titles[rew_fn_id]}",
                            fontsize=12,
                        )

                        axes[0].plot(
                            regret_artificial_replay,
                            label="Artificial Replay",
                            linewidth=2,
                            linestyle="-",
                            marker="o",
                            markersize=3,
                            markevery=max(1, len(regret_artificial_replay) // 50),
                            alpha=0.85,
                        )
                        axes[0].plot(
                            regret_full_start,
                            label="Full Start",
                            linewidth=2,
                            linestyle="--",
                            marker="s",
                            markersize=3,
                            markevery=max(1, len(regret_full_start) // 50),
                            alpha=0.85,
                        )
                        axes[0].set_ylabel("Regret", fontsize=12)
                        axes[0].set_title(
                            "LinUCB Regret Comparison: Artificial Replay vs Full Start\n"
                            f"(d={d}, K={K}, N={N}, dist={data_distribution})",
                            fontsize=13,
                        )
                        axes[0].legend(fontsize=11)
                        axes[0].grid(True, alpha=0.3)

                        axes[1].plot(
                            cum_regret_artificial,
                            label="Artificial Replay",
                            linewidth=2,
                            linestyle="-",
                            marker="o",
                            markersize=3,
                            markevery=max(1, len(cum_regret_artificial) // 50),
                            alpha=0.85,
                        )
                        axes[1].plot(
                            cum_regret_full_start,
                            label="Full Start",
                            linewidth=2,
                            linestyle="--",
                            marker="s",
                            markersize=3,
                            markevery=max(1, len(cum_regret_full_start) // 50),
                            alpha=0.85,
                        )
                        axes[1].set_xlabel("Timestep", fontsize=12)
                        axes[1].set_ylabel("Cumulative Regret", fontsize=12)
                        axes[1].set_title("Cumulative Regret", fontsize=13)
                        axes[1].legend(fontsize=11)
                        axes[1].grid(True, alpha=0.3)

                        plt.tight_layout()
                        output_path = (
                            "experimentation results/"
                            f"linucb_d{d}_K{K}_N{N}_dist-{data_distribution}_rew{rew_fn_id}.png"
                        )
                        plt.savefig(output_path, dpi=150)
                        print(f"Plot saved as '{output_path}'")
                        plt.close(fig)
