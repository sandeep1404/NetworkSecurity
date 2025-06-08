from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import precision_score, recall_score, f1_score
import sys

def get_classification_score(y_true: list, y_pred: list) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics and return them as an artifact.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :return: ClassificationMetricArtifact containing precision, recall, and F1 score.
    """
    try:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        return ClassificationMetricArtifact(precision=precision, recall=recall, f1_score=f1)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e