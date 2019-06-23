from sklearn.decomposition import PCA
class PCAPreprocessor:
    def __init__(self):
        self.pca = None

    def process(self, x_data, y_data = 0, topk = 0):
        if self.pca == None:
            self.pca = PCA(n_components = topk)
            self.pca.fit(x_data)
        return self.pca.transform(x_data)

    def report(self):
        print("components_")
        print(self.pca.components_)
        print("explanined_variance_ratio")
        print(self.pca.explained_variance_ratio_)
        print("mean")
        print(self.pca.mean_)
        print("noise_variance")
        print(self.pca.noise_variance_)
