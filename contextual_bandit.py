import numpy as np
from typing import Callable, Optional, Tuple
from statsmodels.tools.tools import add_constant


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit with d-dimensional contexts and K arms.
    
    Parameters
    ----------
    d : int
        Dimension of the context vectors
    K : int
        Number of arms
    reward_function : Callable[[np.ndarray, int], float]
        Function that takes (context, arm_index) and returns a reward
    context_radius : float, optional
        Radius of the ball from which contexts are sampled uniformly (default: 1.0)
    random_seed : int, optional
        Random seed for reproducibility
    online_distribution : str, optional
        Distribution of contexts for online pulls (default: "uniform")
    online_data : int, optional
        Number of steps to generate online contexts. If None, no data is generated (generated live). Must be >= number of trials.
    discrete_contexts : int, optional
        If > 0, contexts are sampled as discrete integers in [0, discrete_contexts)
        (default: 0, meaning continuous contexts)

    ### For data generation and historical data handling:
    offline_data : int, optional
        Number of steps to generate for offline data. If None, no data is generated.
    offline_arm_distribution : str, optional
        Distribution of arms for offline data generation (default: "uniform")
    offline_distribution : str, optional
        Distribution of contexts for offline data generation (default: "uniform")
    """
    
    def __init__(
        self,
        d: int,
        K: int,
        reward_function: Callable[[np.ndarray, int], float],
        context_radius: float = 1.0,
        random_seed: Optional[int] = None,
        online_distribution: str = "uniform",
        online_data: Optional[int] = None,
        discrete_contexts: int = 0,
        offline_data: Optional[int] = None,
        offline_arm_distribution: str = "uniform",
        offline_distribution: str = "uniform",
    ):
        self.d = d
        self.K = K
        self.reward_function = reward_function
        self.context_radius = context_radius
        self.online_distribution = online_distribution
        self.online_data = online_data
        self.discrete_contexts = discrete_contexts
        self.offline_data = offline_data
        self.offline_arm_distribution = offline_arm_distribution
        self.offline_distribution = offline_distribution

        if random_seed is not None:
            np.random.seed(random_seed)
        
        # History tracking (for live interactions)
        self.online_contexts = []
        self.online_arms = []
        self.online_rewards = []
        self.t = 0
        
        # Data tracking (for offline data collection)
        self.offline_contexts = []
        self.offline_arms = []
        self.offline_rewards = []
        self.offline_data_size = 0
        self.offline_data_pointers = np.zeros(self.K, dtype=int)
        
        # Generate initial offline / online data if specified
        if offline_data is not None:
            self.generate_data(num_steps=offline_data, arm_distribution=offline_arm_distribution, context_distribution=offline_distribution)
        
        if online_data is not None:
            self.online_contexts = [self.sample_context(distribution=online_distribution) for _ in range(online_data)]

    def sample_context(self, distribution) -> np.ndarray:
        """
        Sample a context uniformly from a d-dimensional ball of given radius, or from a discrete set of integers [0, discrete_contexts).
        
        Returns
        -------
        np.ndarray
            A d-dimensional context vector sampled uniformly from the ball, or a discrete context vector if discrete_contexts > 0
        """

        if self.discrete_contexts > 0:

            p_large = np.linspace(0.1, 1, self.discrete_contexts)
            p_small = np.linspace(1, 0.1, self.discrete_contexts)
            p_large /= p_large.sum()
            p_small /= p_small.sum()

            if distribution == "uniform":
                context = np.random.randint(0, self.discrete_contexts, size=self.d)
                context = context / self.discrete_contexts  # Normalize to [0, 1]
                context = np.insert(context, 0, 1.0)  # Add intercept term

            elif distribution == "biased-small":
                # Biased distribution: higher probability for smaller integers
                probabilities = p_small
                context = np.random.choice(self.discrete_contexts, size=self.d, p=probabilities)
                context = context / self.discrete_contexts  # Normalize to [0, 1]
                context = np.insert(context, 0, 1.0)  # Add intercept term

            elif distribution == "biased-large":
                # Biased distribution: higher probability for larger integers
                probabilities = p_large
                context = np.random.choice(self.discrete_contexts, size=self.d, p=probabilities)
                context = context / self.discrete_contexts  # Normalize to [0, 1]
                context = np.insert(context, 0, 1.0)  # Add intercept term
            
            elif distribution == "small-large":
                x = np.random.choice(self.discrete_contexts, size=1, p=p_small)
                y = np.random.choice(self.discrete_contexts, size=1, p=p_large)
                context = np.array([x[0], y[0]])
                context = context / self.discrete_contexts  # Normalize to [0, 1]
                context = np.insert(context, 0, 1.0)  # Add intercept term

            elif distribution == "large-small":
                x = np.random.choice(self.discrete_contexts, size=1, p=p_small)
                y = np.random.choice(self.discrete_contexts, size=1, p=p_large)
                context = np.array([y[0], x[0]])
                context = context / self.discrete_contexts  # Normalize to [0, 1]
                context = np.insert(context, 0, 1.0)  # Add intercept term

        # Continuous contexts
        # https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere#:~:text=The%20best%20way%20to%20generate,that%20radius%20in%20d%20dimensions.
        else:
            if distribution == "uniform":
                z = np.random.randn(self.d)
                z = z / np.linalg.norm(z)  # Normalize to unit sphere
                r = self.context_radius * np.random.uniform(0, 1) ** (1.0 / self.d)
                
            elif distribution == "biased-small":
                # Biased distribution: higher probability for smaller values in continuous space
                # For simplicity, we can bias the radius r to be smaller values more likely
                z = np.random.randn(self.d)
                z = z / np.linalg.norm(z)  # Normalize to unit sphere
                probabilities = np.linspace(1, 0.1, 1000)
                probabilities /= probabilities.sum()
                r = self.context_radius * np.random.choice(np.linspace(0, 1, 1000), p=probabilities)
                
            elif distribution == "biased-large":
                # Biased distribution: higher probability for larger values in continuous space
                z = np.random.randn(self.d)
                z = z / np.linalg.norm(z)  # Normalize to unit sphere
                probabilities = np.linspace(0.1, 1, 1000)
                probabilities /= probabilities.sum()
                r = self.context_radius * np.random.choice(np.linspace(0, 1, 1000), p=probabilities)
            context = r * z
            context = np.insert(context, 0, 1.0)  # Add intercept term
                
        return context
    
    def get_online_sample(self) -> np.ndarray:
        """
        Get the next context for online interaction. If online_data was specified, return the pre-generated contexts in order; otherwise, sample a new context.
        """

        return self.online_contexts[self.t] if self.t < len(self.online_contexts) else self.sample_context(distribution=self.online_distribution)

    def get_reward(self, context: np.ndarray, arm: int) -> float:
        """
        Get the reward for pulling a specific arm given a context.
        
        Parameters
        ----------
        context : np.ndarray
            The d-dimensional context vector
        arm : int
            The arm index (0 to K-1)
            
        Returns
        -------
        float
            The reward value
        """
        if arm < 0 or arm >= self.K:
            raise ValueError(f"Arm index must be between 0 and {self.K-1}")
        
        return self.reward_function(context, arm) + np.random.normal(0, 0.1) # noise term
    
    def pull_arm(self, arm: int, context: Optional[np.ndarray] = None, data: bool = False) -> Tuple[np.ndarray, float]:
        """
        Pull an arm and observe the reward. If no context is provided, sample a new one.
        
        Parameters
        ----------
        arm : int
            The arm to pull (0 to K-1)
        context : np.ndarray, optional
            The context to use. If None, a new context is sampled.
        data : bool, optional
            If True, store the interaction in data history; otherwise, in live history.
            
        Returns
        -------
        Tuple[np.ndarray, float]
            The context used and the reward observed
        """
        if context is None:
            context = self.sample_context(distribution=self.online_distribution)
        
        reward = self.get_reward(context, arm)
        
        if data:
            self.offline_contexts.append(context)
            self.offline_arms.append(arm)
            self.offline_rewards.append(reward)
            self.offline_data_size += 1
        else:
            self.online_contexts.append(context)
            self.online_arms.append(arm)
            self.online_rewards.append(reward)
            self.t += 1
        
        return context, reward
    
    def get_all_rewards(self, context: np.ndarray) -> np.ndarray:
        """
        Get the expected rewards for all arms given a context.
        
        Parameters
        ----------
        context : np.ndarray
            The d-dimensional context vector
            
        Returns
        -------
        np.ndarray
            Array of shape (K,) containing rewards for each arm
        """
        rewards = np.array([self.reward_function(context, arm) for arm in range(self.K)])
        return rewards
    
    def get_optimal_arm(self, context: np.ndarray) -> int:
        """
        Get the optimal arm for a given context.
        
        Parameters
        ----------
        context : np.ndarray
            The d-dimensional context vector
            
        Returns
        -------
        int
            The index of the optimal arm
        """
        rewards = self.get_all_rewards(context)
        return np.argmax(rewards)

    def compute_regret(self, chosen_arm: int, context: Optional[np.ndarray] = None) -> float:
        """
        Compute the regret of choosing a specific arm given a context.
        
        Parameters
        ----------
        chosen_arm : int
            The arm that was chosen
        context : np.ndarray, optional
            The context to use. If None, the last context in history is used.
            
        Returns
        -------
        float
            The regret value
        """
        if context is None:
            if self.t == 0:
                raise ValueError("No context available in online history.")
            context = self.online_contexts[-1]
        
        optimal_arm = self.get_optimal_arm(context)
        optimal_reward = self.reward_function(context, optimal_arm)
        chosen_reward = self.reward_function(context, chosen_arm)
        
        regret = optimal_reward - chosen_reward
        return regret
    
    def reset_online_history(self):
        """Reset the bandit's online pulls while keeping the same offline data & contexts."""
        self.online_arms = []
        self.online_rewards = []
        self.t = 0

    def generate_data(self, num_steps: int = None, arm_distribution: str = "uniform", context_distribution: str = "uniform") -> None:
        """
        Generate data by pulling arms according to a specified distribution.
        
        Parameters
        ----------
        num_steps : int
        arm_distribution : str, optional
            Distribution of arms for data generation (default: "uniform")
        context_distribution : str, optional
            Distribution of contexts for data generation (default: "uniform")
        """

        if num_steps is None:
            num_steps = self.K * 10  # Default to 10 pulls per arm                

        # Generate arm sequence based on distribution
        if arm_distribution == "uniform":
            # All arms pulled equally
            arms = np.tile(np.arange(self.K), num_steps // self.K + 1)[:num_steps]
            np.random.shuffle(arms)
            
        elif arm_distribution == "lightly_unbalanced":
            # Some arms pulled more, but all covered
            arm_counts = np.ones(self.K, dtype=int)
            remaining = num_steps - self.K
            # Distribute remaining pulls with slight bias
            for i in range(remaining):
                arm = np.random.randint(0, self.K)
                arm_counts[arm] += 1
            arms = np.repeat(np.arange(self.K), arm_counts)[:num_steps]
            np.random.shuffle(arms)
            
        elif arm_distribution == "half_coverage":
            # Only half the arms are pulled
            covered_arms = np.random.choice(self.K, size=self.K // 2, replace=False)
            arms = np.random.choice(covered_arms, size=num_steps)
            
        else:
            raise ValueError(f"Invalid arm distribution: {arm_distribution}")
        
        # Pull arms according to the generated sequence
        for arm in arms:
            context = self.sample_context(distribution=context_distribution)
            self.pull_arm(arm, context, data=True)

    def sample_data(self, arm: int) -> Tuple[np.ndarray, float, int]:
        """
        Sample a context and reward from the stored data for a specific arm.
        Uses a per-arm pointer to return samples in order.
        
        Parameters
        ----------
        arm : int
            The arm index to sample data for
            
        Returns
        -------
        Tuple[np.ndarray, float, int]
            A tuple of (context, reward, index in data)
        """
        indices = [i for i, a in enumerate(self.offline_arms) if a == arm]
        if not indices:
            raise ValueError(f"No offline data available for arm {arm}")
        
        pointer = self.offline_data_pointers[arm]
        if pointer >= len(indices):
            raise ValueError(f"No more samples left for arm {arm}")
        
        idx = indices[pointer]
        self.offline_data_pointers[arm] += 1
        context = self.offline_contexts[idx]
        reward = self.offline_rewards[idx]
        
        return context, reward

    def get_history(self) -> Tuple[list, list, list]:
        """
        Get the bandit's history.
        
        Returns
        -------
        Tuple[list, list, list]
            Lists of contexts, arms pulled, and rewards observed
        """
        return self.contexts_history, self.arms_history, self.rewards_history
    
    def get_offline_data(self) -> Tuple[list, list, list]:
        """
        Get the bandit's offline generated data.
        
        Returns
        -------
        Tuple[list, list, list]
            Lists of contexts, arms pulled, and rewards observed in the generated data
        """
        return self.offline_contexts, self.offline_arms, self.offline_rewards