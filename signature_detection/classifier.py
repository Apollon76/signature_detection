from typing import List

from sklearn.ensemble import RandomForestClassifier

from signature_detection.feature_extractor import FeatureExtractor


class Classifier:
    def __init__(self):
        self.__feature_extractor = FeatureExtractor()
        self.__classifier = RandomForestClassifier()

    def fit(self, data, labels: List[bool]):
        data = self.__convert_data(data)
        self.__classifier.fit(data, labels)

    def predict(self, data) -> bool:
        data = self.__convert_data(data)
        return self.__classifier.predict(data)

    def __convert_data(self, data):
        return [self.__feature_extractor.extract(element) for element in data]
