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
    data_num_steps : int, optional
        Number of steps to generate for offline data. If None, no data is generated.
    data_distribution : str, optional
        Distribution of arms for data generation (default: "uniform")
    """
    
    def __init__(
        self,
        d: int,
        K: int,
        reward_function: Callable[[np.ndarray, int], float],
        context_radius: float = 1.0,
        random_seed: Optional[int] = None,
        data_num_steps: Optional[int] = None,
        data_distribution: str = "uniform"
    ):
        self.d = d
        self.K = K
        self.reward_function = reward_function
        self.context_radius = context_radius
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # History tracking (for live interactions)
        self.contexts_history = []
        self.arms_history = []
        self.rewards_history = []
        self.t = 0
        
        # Data tracking (for offline data collection)
        self.contexts_data = []
        self.arms_data = []
        self.rewards_data = []
        self.data_size = 0
        self.data_pointers = np.zeros(self.K, dtype=int)
        
        # Generate initial data if specified
        if data_num_steps is not None:
            self.generate_data(num_steps=data_num_steps, distribution=data_distribution)
    
    def sample_context(self) -> np.ndarray:
        """
        Sample a context uniformly from a d-dimensional ball of given radius.
        
        Returns
        -------
        np.ndarray
            A d-dimensional context vector sampled uniformly from the ball
        """
        # https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere#:~:text=The%20best%20way%20to%20generate,that%20radius%20in%20d%20dimensions.

        z = np.random.randn(self.d)
        z = z / np.linalg.norm(z)  # Normalize to unit sphere
        r = self.context_radius * np.random.uniform(0, 1) ** (1.0 / self.d)
        
        context = r * z
        context = np.insert(context, 0, 1.0)  # Add intercept term
        return context
    
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
            context = self.sample_context()
        
        reward = self.get_reward(context, arm)
        
        if data:
            self.contexts_data.append(context)
            self.arms_data.append(arm)
            self.rewards_data.append(reward)
            self.data_size += 1
        else:
            self.contexts_history.append(context)
            self.arms_history.append(arm)
            self.rewards_history.append(reward)
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
        rewards = np.array([self.get_reward(context, arm) for arm in range(self.K)])
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
                raise ValueError("No context available in history.")
            context = self.contexts_history[-1]
        
        optimal_arm = self.get_optimal_arm(context)
        optimal_reward = self.get_reward(context, optimal_arm)
        chosen_reward = self.get_reward(context, chosen_arm)
        
        regret = optimal_reward - chosen_reward
        return regret
    
    def reset_history(self):
        """Reset the bandit's history."""
        self.contexts_history = []
        self.arms_history = []
        self.rewards_history = []
        self.t = 0

    def generate_data(self, num_steps: int = None, distribution: str = "uniform"):
        """
        Generate data by pulling arms according to a specified distribution.
        
        Parameters
        ----------
        num_steps : int
        distribution : str, optional
            Distribution of arms to pull. Options:
            - "uniform": all arms pulled equally (default)
            - "lightly_unbalanced": some arms pulled more, but all covered
            - "half_coverage": only half the arms are pulled
        """
        if num_steps is None:
            num_steps = self.K * 10  # Default to 10 pulls per arm                
        
        # Generate arm sequence based on distribution
        if distribution == "uniform":
            # All arms pulled equally
            arms = np.tile(np.arange(self.K), num_steps // self.K + 1)[:num_steps]
            np.random.shuffle(arms)
            
        elif distribution == "lightly_unbalanced":
            # Some arms pulled more, but all covered
            arm_counts = np.ones(self.K, dtype=int)
            remaining = num_steps - self.K
            # Distribute remaining pulls with slight bias
            for i in range(remaining):
                arm = np.random.randint(0, self.K)
                arm_counts[arm] += 1
            arms = np.repeat(np.arange(self.K), arm_counts)[:num_steps]
            np.random.shuffle(arms)
            
        elif distribution == "half_coverage":
            # Only half the arms are pulled
            covered_arms = np.random.choice(self.K, size=self.K // 2, replace=False)
            arms = np.random.choice(covered_arms, size=num_steps)
            
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        # Pull arms according to the generated sequence
        for arm in arms:
            context = self.sample_context()
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
        indices = [i for i, a in enumerate(self.arms_data) if a == arm]
        if not indices:
            raise ValueError(f"No data available for arm {arm}")
        
        pointer = self.data_pointers[arm]
        if pointer >= len(indices):
            raise ValueError(f"No more samples left for arm {arm}")
        
        idx = indices[pointer]
        self.data_pointers[arm] += 1
        context = self.contexts_data[idx]
        reward = self.rewards_data[idx]
        
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
