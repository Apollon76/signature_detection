class Classifier:
    def __init__(self,
                 edges_detector: EdgesDetector):
        self.__edges_detector = edges_detector

    def fit(self, data, labels):
        ...

    def predict(self, data):
        ...
