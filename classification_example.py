import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from distribution_similarity import DistributionSimilarity

# Assume we have data for different classes
class_0_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=500)
class_1_data = np.random.multivariate_normal(mean=[2, 2], cov=[[1.5, 0.3], [0.3, 1.5]], size=500)

# Create similarity calculators for each class
sim_calc_0 = DistributionSimilarity(class_0_data)
sim_calc_1 = DistributionSimilarity(class_1_data)

# Function to create features from similarity measures
def create_similarity_features(observation):
    features = []
    
    # Class 0 similarities
    features.append(sim_calc_0.mahalanobis_distance(observation))
    _, mean_zscore_0 = sim_calc_0.zscore_similarity(observation)
    features.append(mean_zscore_0)
    features.append(sim_calc_0.kernel_density_estimate(observation))
    features.append(sim_calc_0.probability_score(observation))
    
    # Class 1 similarities
    features.append(sim_calc_1.mahalanobis_distance(observation))
    _, mean_zscore_1 = sim_calc_1.zscore_similarity(observation)
    features.append(mean_zscore_1)
    features.append(sim_calc_1.kernel_density_estimate(observation))
    features.append(sim_calc_1.probability_score(observation))
    
    return np.array(features)

# Example of using similarity measures in classification
X = np.vstack([class_0_data, class_1_data])
y = np.hstack([np.zeros(500), np.ones(500)])

# Create similarity-based features
X_similarity = np.array([create_similarity_features(obs) for obs in X])

# Split data and train classifier
X_train, X_test, y_train, y_test = train_test_split(X_similarity, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Classification accuracy: {score:.3f}")
