import numpy as np
from scipy.special import logsumexp


class NaiveBayesClassifier:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.array, y: np.array) -> list[str]:
        # count class number
        self.num_classes = len(set(y))

        # set class encoder and decoder (dicts and functions)
        self.encoder_dict = {class_name: ind for ind,
                             class_name in enumerate(sorted(set(y)))}
        self.decoder_dict = {v: k for k, v in self.encoder_dict.items()}

        self.encode_transform = self._transform(self.encoder_dict)
        self.decode_transform = self._transform(self.decoder_dict)
        # set design matrice and target values
        self.data = X
        self.targets = self.encode_transform(y)

        self.prior_proba = [np.mean(self.targets == class_num)
                            for class_num in range(self.num_classes)]

        # fit Normal distribution parameters
        self.class_parameters = list()
        for class_num in range(self.num_classes):
            params = self._get_params(class_num)
            self.class_parameters.append(params)

        self.mu, self.var = zip(*self.class_parameters)
        self.mu = np.array(self.mu)
        self.var = np.array(self.var)

        return self.encoder_dict

    def _get_params(self, class_num: int) -> tuple[np.array]:
        mean = np.mean(self.data[self.targets == class_num], axis=0)
        # compute unbiased variance `\sum_{j = 1}^n \left( x_j - \bar{x} \right)^2 / (n - 1)`
        # for each column
        var = np.var(self.data[self.targets == class_num], axis=0, ddof=1)
        return (mean, var)

    def predict(self, X: np.array) -> np.array:
        jll = self._joint_log_likelihood(X)
        return self.decode_transform(np.argmax(jll, axis=1))
        # return np.argmax(jll, axis=1)

    def predict_proba(self, X: np.array) -> np.array:
        return np.exp(self._predict_log_proba(X))

    def _predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(self.num_classes):
            jointi = np.log(self.prior_proba[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var[i, :]))
            n_ij -= 0.5 * \
                np.sum(((X - self.mu[i, :]) ** 2) / (self.var[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    # Encode `class names` into numeric values
    def _transform(self, selectors: dict):
        def get_encoded_val(val, selectors=selectors):
            assert val in selectors.keys(), '{} not in keys!'.format(val)
            return selectors[val]

        def transformation(data):
            new_data = list(map(get_encoded_val, data))
            return np.array(new_data)

        return transformation
