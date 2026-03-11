import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

class my_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.scaler = StandardScaler()

    def standardize(self, X):
        return self.scaler.fit_transform(X)

    def calculate_covariance_matrix(self, X):
        return np.cov(X.T)

    def calculate_eigenvalues_and_eigenvectors(self, covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return eigenvalues, eigenvectors

    def fit(self, X):
        X_standardized = self.scaler.fit_transform(X)
        covariance_matrix = self.calculate_covariance_matrix(X_standardized)
        eigenvalues, eigenvectors = self.calculate_eigenvalues_and_eigenvectors(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.components = self.eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        # Project data onto principal components
        X_standardized = self.scaler.transform(X)
        #return as dataframe
        result = pd.DataFrame(X_standardized @ self.components, columns=[f'PC{i+1}' for i in range(self.n_components)])
        return result

def test_pca():
    breast_canser = load_breast_cancer()
    X = breast_canser.data
    pca = my_PCA(n_components=2)
    pca.fit(X)
    X_transformed = pca.transform(X)
    # plot the classes colored by the first two principal components
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=breast_canser.target)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Breast Cancer Dataset')
    plt.show()

    #compare to sklearn PCA
    pca_sklearn = PCA(n_components=2)
    X_sklearn = pca_sklearn.fit_transform(X)
    plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=breast_canser.target)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Sklearn PCA of Breast Cancer Dataset')
    plt.show()

