from signature_detection.feature_extractor import FeatureExtractor


class Classifier:
    def __init__(self):
        self.__feature_extractor = feature_extractor

    def fit(self, data, labels):
        ...

    def predict(self, data) -> bool:
        ...
