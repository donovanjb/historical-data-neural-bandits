import numpy as np
import matplotlib.pyplot as plt
from contextual_bandit import ContextualBandit
from linucb import linucb
from linucb import linucb_matching_context
import itertools

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
    return np.abs(r * z)


def create_bandit(reward_fn_id, d=5, K=3, context_radius=1.0, random_seed=None, online_distribution="uniform", discrete_contexts=0, offline_data=None, offline_arm_distribution="uniform", offline_distribution="uniform"):
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
    online_distribution : str
        Distribution for sampling contexts online ("uniform" or "biased")
    discrete_contexts : int
        If > 0, number of discrete contexts to use (instead of continuous sampling)
    offline_data : int
        Number of offline data samples to generate (if any)
    offline_arm_distribution : str
        Distribution for sampling arms in offline data ("uniform", "lightly_unbalanced", "half_coverage")
    offline_distribution : str
        Distribution for sampling contexts in offline data ("uniform" or "biased")
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
        
    elif reward_fn_id == 4:
        # Reward function 4: r = 10(x^T a)
        # Sample arm parameters uniformly from unit ball
        arm_params = np.array([_sample_uniform_ball(d+1) for _ in range(K)])
        
        def reward_function_4(context, arm):
            a = arm_params[arm]
            return 2 * np.dot(context, a)
        
        reward_fn = reward_function_4
        
    else:
        raise ValueError(f"reward_fn_id must be 1, 2, 3, or 4, got {reward_fn_id}")
    
    # Create and return the bandit
    bandit = ContextualBandit(
        d=d,
        K=K,
        reward_function=reward_fn,
        context_radius=context_radius,
        random_seed=random_seed,
        online_distribution=online_distribution,
        discrete_contexts=discrete_contexts,
        offline_data=offline_data,
        offline_arm_distribution=offline_arm_distribution,
        offline_distribution=offline_distribution
    )
    
    return bandit


if __name__ == "__main__":

    print(np.array([_sample_uniform_ball(5+1) for _ in range(10)]))

    # Example usage
    print("Creating bandits with different reward functions...\n")
    
    # Test each reward function
    for fn_id in [1, 2, 3, 4]:
        print(f"Reward Function {fn_id}:")
        bandit = create_bandit(
            reward_fn_id=fn_id,
            d=5,
            K=3,
            # random_seed=42,
            online_distribution="uniform",
            discrete_contexts=5,
            offline_data=None,
            offline_arm_distribution="uniform",
            offline_distribution="uniform"
        )
        
        # Sample a context and pull each arm
        context = bandit.sample_context(distribution="uniform")
        print(f"  Context: {context[:3]}...")
        
        rewards = bandit.get_all_rewards(context)
        print(f"  Rewards for all arms: {rewards}")
        
        optimal_arm = bandit.get_optimal_arm(context)
        print(f"  Optimal arm: {optimal_arm}\n")
    
    # Experiment: Test LinUCB with artificial replay and full start
    print("\n" + "="*70)
    print("EXPERIMENT: LinUCB with Reward Function 4")
    print("="*70 + "\n")

    test_dims = [2]
    test_arms = [5]
    test_rew_fn_id = [4]
    test_online_distribution = ["uniform", "biased-small", "biased-large"]
    test_discrete_contexts = [3, 5]
    
    test_N = [250]
    test_offline_distribution = ["uniform", "biased-small", "biased-large"]
    test_offline_arm_distribution = ["lightly_unbalanced"]

    test_cases = list(itertools.product(
        test_dims,
        test_arms,
        test_rew_fn_id,
        test_online_distribution,
        test_discrete_contexts,
        test_N,
        test_offline_distribution,
        test_offline_arm_distribution
    ))

    print(f"Total test cases: {len(test_cases)}\n")
    print(test_cases)

    reward_fn_titles = {
        1: r"$r = 10(x^T a)^2$",
        2: r"$r = x^T A^T A x$",
        3: r"$r = \cos(3x^T a)$",
        4: r"$r = 10(x^T a)$",
    }

    alpha = 1.2
    trials = 2500
    
    random_seed = 42
    np.random.seed(random_seed)

    for test in test_cases:
        d, K, rew_fn_id, online_dist, discrete_ctx, N, offline_dist, offline_arm_dist = test
        print("\n" + "-" * 70)
        print(
            f"Running: d={d}, K={K}, reward_fn={rew_fn_id}, online_dist={online_dist}, "
            f"discrete_ctx={discrete_ctx}, N={N}, offline_dist={offline_dist}, "
            f"offline_arm_dist={offline_arm_dist}"
        )

        bandit_with_data = create_bandit(
            reward_fn_id=rew_fn_id,
            d=d,
            K=K,
            # random_seed=42,
            online_distribution=online_dist,
            discrete_contexts=discrete_ctx,
            offline_data=N,
            offline_arm_distribution=offline_arm_dist,
            offline_distribution=offline_dist
        )

        print(f"Generated {bandit_with_data.data_size} offline data samples")
        print(f"Arm distribution: {np.bincount(bandit_with_data.arms_data, minlength=K)}\n")
        
        contexts, arms, rewards = bandit_with_data.get_data()
        contexts = np.array(contexts)
        arms = np.array(arms)
        rewards = np.array(rewards)
        
        # Generate plots to visualize distribution of offline data for contexts (dim = 2 only), arms, and rewards
        if d == 2:
            fig, ax = plt.subplots(3, 1, figsize=(18, 15))
            fig.suptitle(f"Offline Data Distribution: d={d}, K={K}, N={N}, Offline Distribution={offline_dist}", fontsize=14)

            x_val = contexts[:, 1]
            y_val = contexts[:, 2]

            hist, xedges, yedges = np.histogram2d(x_val, y_val, bins=discrete_ctx)

            im = ax[0].imshow(
                hist.T, 
                interpolation='nearest', 
                origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='viridis',
                aspect='auto'
            )

            cbar = plt.colorbar(im, ax=ax[0])
            cbar.set_label('Count')

            ax[0].set_title("Contexts", fontsize=12)
            ax[0].set_xlabel("Dimension 1", fontsize=10)
            ax[0].set_ylabel("Dimension 2", fontsize=10)

            ax[1].bar(np.arange(K), np.bincount(arms, minlength=K), width=0.7)
            ax[1].set_title("Arms Pulled", fontsize=12)
            ax[1].set_xlabel("Arm", fontsize=10)
            ax[1].set_xticks(np.arange(K))
            ax[1].set_ylabel("Count", fontsize=10)
            ax[1].grid(True, alpha=0.3)
            ax[2].hist(rewards, bins=9, rwidth=1)
            ax[2].set_title("Rewards", fontsize=12)
            ax[2].set_xlabel("Reward", fontsize=10)
            ax[2].set_ylabel("Count", fontsize=10)
            ax[2].grid(True, alpha=0.3)
            plt.tight_layout()
            output_path = (
                "experimentation results/offline_data_distribution/"
                f"d{d}_K{K}_N{N}_offline-{offline_dist}-discrete{discrete_ctx}.png"
            )
            plt.savefig(
                output_path,
                dpi=150
            )
            plt.close(fig)

        regret_artificial_replay, artificial_pulls = linucb(
            bandit_with_data, alpha, trials, history="artificial replay"
        )
        cum_regret_artificial = np.cumsum(regret_artificial_replay)

        bandit_with_data.reset_history()  # Clear history

        regret_artificial_replay_match_context, artificial_pulls_match_context = linucb_matching_context(
            bandit_with_data, alpha, trials
        )
        cum_regret_artificial_match_context = np.cumsum(regret_artificial_replay_match_context)

        bandit_with_data.reset_history()  # Clear history

        regret_full_start, _ = linucb(
            bandit_with_data, alpha, trials, history="full start"
        )
        cum_regret_full_start = np.cumsum(regret_full_start)

        # plot results
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
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
            regret_artificial_replay_match_context,
            label="Artificial Replay Match Context",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=3,
            markevery=max(1, len(regret_artificial_replay_match_context) // 50),
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
            f"(d={d}, K={K}, N={N}, context={discrete_ctx}, Offline Distribution={offline_dist}, Online Distribution={online_dist})",
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
            cum_regret_artificial_match_context,
            label="Artificial Replay Match Context",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=3,
            markevery=max(1, len(cum_regret_artificial_match_context) // 50),
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

        axes[2].bar(
            np.arange(K) - 0.1,
            artificial_pulls,
            width=0.1,
            alpha=0.85,
            label="Artificial Replay",
        )
        axes[2].bar(
            np.arange(K),
            artificial_pulls_match_context,
            width=0.1,
            alpha=0.85,
            label="Artificial Replay Match Context",
        )
        axes[2].bar(
            np.arange(K) + 0.1,
            np.bincount(arms, minlength=K),
            width=0.1,
            alpha=0.85,
            label="Offline Data",
        )
        axes[2].set_xlabel("Arm", fontsize=12)
        axes[2].set_ylabel("Number of Pulls in Artificial Replay", fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].set_title("Arm Pull Distribution in Artificial Replay", fontsize=13)

        plt.tight_layout()
        output_path = (
            "experimentation results/"
            f"linucb_d{d}_K{K}_N{N}_offline-{offline_dist}-discrete{discrete_ctx}_online-{online_dist}_rew{rew_fn_id}.png"
        )
        # output_path = (
        #     "experimentation results/exact_match_artificial/"
        #     f"linucb_matching_context_d{d}_K{K}_N{N}_offline-{offline_dist}-discrete{discrete_ctx}_online-{online_dist}_rew{rew_fn_id}.png"
        # )
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved as '{output_path}'")
        plt.close(fig)
