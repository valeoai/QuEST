from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np

def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float).mean() * 100, top5_correct.astype(float).mean() * 100


def extract_features(feature_extractor, images, feature_name=None):
    if feature_name:
        if isinstance(feature_name, str):
            feature_name = [feature_name,]
        assert isinstance(feature_name, (list, tuple))

        features = feature_extractor(images, out_feat_keys=feature_name)
    else:
        features = feature_extractor(images)

    return features


def classification_task(classifier, features, labels):
    scores = classifier(features)
    loss = F.cross_entropy(scores, labels)

    return scores, loss


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
        scores_classification, loss_classsification = classification_task(
            classifier, features, labels)
        loss_total = loss_classsification
        record['loss'] = loss_total.item()

    with torch.no_grad(): # Compute accuracies.
        AccuracyTop1, AccuracyTop5 = compute_top1_and_top5_accuracy(
            scores_classification, labels)
        record['AccuracyTop1'] = AccuracyTop1
        record['AccuracyTop5'] = AccuracyTop5

    if is_train: # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record
