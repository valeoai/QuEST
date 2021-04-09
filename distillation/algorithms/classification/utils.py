from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
from distillation.utils import compute_top1_and_top5_accuracy


def extract_features(feature_extractor, images, feature_name=None):
    if feature_name:
        if isinstance(feature_name, str):
            feature_name = [feature_name,]
        assert isinstance(feature_name, (list, tuple))
        return feature_extractor(images, out_feat_keys=feature_name)
    else:
        return feature_extractor(images)


def object_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    feature_name=None):

    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train: # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = extract_features(
            feature_extractor, images, feature_name=feature_name)
        # Perform the object classification task.
        scores = classifier(features)
        loss = F.cross_entropy(scores, labels)
        record['loss'] = loss.item()

    with torch.no_grad(): # Compute accuracies.
        accur_top1, accur_top5 = compute_top1_and_top5_accuracy(scores, labels)
        record['AccuracyTop1'] = accur_top1
        record['AccuracyTop5'] = accur_top5

    if is_train: # Backward loss and apply gradient steps.
        loss.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record
