import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from typing import Union, Tuple, Optional

class DistributionSimilarity:
    """
    A class to calculate similarity measures between a distribution and a single observation.
    """
    
    def __init__(self, population_data: np.ndarray):
        """
        Initialize with population data.
        
        Args:
            population_data: numpy array of shape (n_samples, n_features)
        """
        self.population_data = population_data
        self.pop_mean = np.mean(population_data, axis=0)
        self.pop_cov = np.cov(population_data, rowvar=False)
        self.pop_std = np.std(population_data, axis=0)
    
    def mahalanobis_distance(self, observation: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance between population and observation.
        
        Args:
            observation: Single observation vector of shape (n_features,)
            
        Returns:
            float: Mahalanobis distance
        """
        inv_cov = np.linalg.inv(self.pop_cov)
        return mahalanobis(observation, self.pop_mean, inv_cov)
    
    def zscore_similarity(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate Z-scores for each feature and mean absolute z-score.
        
        Args:
            observation: Single observation vector of shape (n_features,)
            
        Returns:
            Tuple containing:
                - Array of z-scores for each feature
                - Mean absolute z-score
        """
        z_scores = (observation - self.pop_mean) / self.pop_std
        mean_abs_zscore = np.mean(np.abs(z_scores))
        return z_scores, mean_abs_zscore
    
    def kernel_density_estimate(self, observation: np.ndarray, 
                              bandwidth: Optional[Union[str, float]] = 'scott') -> float:
        """
        Calculate kernel density estimate at the observation point.
        
        Args:
            observation: Single observation vector of shape (n_features,)
            bandwidth: Bandwidth method or value for KDE
            
        Returns:
            float: KDE score at the observation point
        """
        kde = stats.gaussian_kde(self.population_data.T, bw_method=bandwidth)
        return kde.evaluate(observation.reshape(-1, 1))[0]
    
    def probability_score(self, observation: np.ndarray) -> float:
        """
        Calculate approximate probability score assuming multivariate normal distribution.
        
        Args:
            observation: Single observation vector of shape (n_features,)
            
        Returns:
            float: Probability density at the observation point
        """
        return stats.multivariate_normal.pdf(observation, 
                                           mean=self.pop_mean, 
                                           cov=self.pop_cov)

# Example usage
if __name__ == "__main__":
    # Generate example population data
    np.random.seed(42)
    population = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=1000
    )
    
    # Create single observation
    observation = np.array([1.5, 1.0])
    
    # Initialize similarity calculator
    sim_calc = DistributionSimilarity(population)
    
    # Calculate different similarity measures
    mah_dist = sim_calc.mahalanobis_distance(observation)
    z_scores, mean_zscore = sim_calc.zscore_similarity(observation)
    kde_score = sim_calc.kernel_density_estimate(observation)
    prob_score = sim_calc.probability_score(observation)
    
    print(f"Mahalanobis distance: {mah_dist:.3f}")
    print(f"Z-scores: {z_scores}")
    print(f"Mean absolute Z-score: {mean_zscore:.3f}")
    print(f"KDE score: {kde_score:.3f}")
    print(f"Probability score: {prob_score:.3f}")
