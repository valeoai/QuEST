from __future__ import print_function

import torch
import torch.nn.functional as F

import distillation.algorithms.algorithm as algorithm
import distillation.algorithms.classification.utils as cls_utils
import distillation.algorithms.clustering.cluster_utils as cluster_utils

import distillation.utils as utils
import os
import tempfile

def visual_word_dense_prediction_soft(
    predictor,
    features,
    targets,
    record):

    if not isinstance(features, (list, tuple)):
        features = [features,]
    if not isinstance(targets, (list, tuple)):
        targets = [targets,]

    assert len(features) == len(targets)
    num_target_levels = len(features)
    loss_weights = [1.0 for _ in range(num_target_levels)]
    assert isinstance(loss_weights, (list, tuple))
    assert len(loss_weights) == num_target_levels

    for i in range(num_target_levels):
        assert features[i].dim() == 4
        assert targets[i].dim() == 4

    # targets shape: batch_size x 1 x height x width
    # features shape: batch_size x num_channels x height x width
    if len(features) == 1:
        scores = [predictor(features[0]),]
    else:
        scores = predictor(features)
    # scores shape: batch_size x num_words x height x width

    loss_total = 0.0
    for i in range(num_target_levels):
        assert scores[i].dim() == 4
        assert targets[i].size(0) == scores[i].size(0) # batch size
        assert targets[i].size(1) == scores[i].size(2) # height
        assert targets[i].size(2) == scores[i].size(3) # width
        assert targets[i].size(3) == scores[i].size(1) # channels/centroids

        targets[i] = targets[i].view(-1, targets[i].size(3))
        scores[i] = scores[i].permute(0, 2, 3, 1).contiguous()
        scores[i] = scores[i].view(-1, scores[i].size(3))

        scores_log = F.log_softmax(scores[i], dim=1)
        loss = F.kl_div(scores_log, targets[i], reduction='batchmean')

        loss_total = loss_total + loss_weights[i] * loss
        if num_target_levels > 1:
            record[f'loss_vword_l{i}'] = loss.item()

        with torch.no_grad():
            key = 'Accur_vword' if (num_target_levels == 1) else f'Accur_vword_l{i}'
            targets[i] = targets[i].max(dim=1)[1]
            record[key] = utils.top1accuracy(scores[i], targets[i])

    record[f'loss_vword'] = loss_total.item()
    return scores, loss_total, targets, record


def object_classification_multiple_features_with_BoW_transfer_simpler(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    BoW_predictor,
    BoW_predictor_optimizer,
    feature_extractor_target,
    vector_quantizer_target,
    images,
    labels,
    is_train,
    feature_name=None,
    BoW_loss_coef=1.0,
    cls_loss_coef=1.0):

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    record = {}
    if is_train: # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        BoW_predictor_optimizer.zero_grad()

    # Extrack knowledge from teacher network.
    with torch.no_grad():
        feature_extractor_target.eval()
        features_target = cls_utils.extract_features(
            feature_extractor_target, images, feature_name=feature_name)
        if not isinstance(features_target, (list, tuple)):
            features_target = [features_target, ]

    if not isinstance(vector_quantizer_target, (list, tuple)):
        vector_quantizer_target = [vector_quantizer_target, ]
    assignments, encodings_1hot = [], []

    num_vq_targets = len(features_target)
    for k in range(num_vq_targets):
        vector_quantizer_target[k].eval()
        _, _, perp_trg, assign_this, enc_1hot_this, mean_assign_score = (
            vector_quantizer_target[k](features_target[k], True))
        assignments.append(assign_this)
        encodings_1hot.append(enc_1hot_this)
        record[f'perp_trg_{k}'] = perp_trg.item()
        record[f'mean_assign_dist_{k}'] = mean_assign_score.item()

    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name)

        if not isinstance(features, (list, tuple)):
            features = [features,]
        assert len(features) == len(assignments)
        assert len(features) == len(encodings_1hot)

        # Perform the object classification task.
        scores_cls, loss_cls = cls_utils.classification_task(
            classifier, features[-1], labels)
        record['loss_cls'] = loss_cls.item()
        loss_total = loss_cls * cls_loss_coef

        _, loss_vword, _, record = visual_word_dense_prediction_soft(
            BoW_predictor, features, encodings_1hot, record)

        loss_total = loss_total + loss_vword * BoW_loss_coef
        record['loss_total'] = loss_total.item()

    with torch.no_grad(): # Compute accuracies.
        AccTop1 = utils.top1accuracy(scores_cls, labels)
        record['AccuracyTop1'] = AccTop1
        record['Error'] = 100.0 - AccTop1

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        BoW_predictor_optimizer.step()

    return record


class ClassificationBoWtransfer(algorithm.Algorithm):
    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        self.BoW_loss_coef = opt.get('BoW_loss_coef', 1.0)
        self.cls_loss_coef = opt.get('cls_loss_coef', 1.0)
        self.keep_best_model_metric_name = 'AccuracyTop1'
        feature_name = opt.get('feature_name', None)

        if feature_name:
            assert isinstance(feature_name, (list, tuple))

        self.feature_name = feature_name

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def set_tensors(self, batch):
        assert len(batch) == 2
        images, labels = batch
        self.tensors['images'].resize_(images.size()).copy_(images)
        self.tensors['labels'].resize_(labels.size()).copy_(labels)

        return 'classification'

    def train_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=True)

    def evaluation_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=False)

    def process_batch_classification_task(self, batch, is_train):
        self.set_tensors(batch)

        if (isinstance(self.feature_name, (list, tuple)) and
            (len(self.feature_name) > 1)):
            vector_quantizer_target = [
                self.networks.get(f'vector_quantizer_target_{key}')
                for key in self.feature_name]
        else:
            vector_quantizer_target = self.networks.get(
                'vector_quantizer_target')

        record = object_classification_multiple_features_with_BoW_transfer_simpler(
            feature_extractor=self.networks['feature_extractor'],
            feature_extractor_optimizer=self.optimizers.get('feature_extractor'),
            classifier=self.networks['classifier'],
            classifier_optimizer=self.optimizers.get('classifier'),
            BoW_predictor=self.networks.get('BoW_predictor'),
            BoW_predictor_optimizer=self.optimizers.get('BoW_predictor'),
            feature_extractor_target=self.networks.get('feature_extractor_target'),
            vector_quantizer_target=vector_quantizer_target,
            images=self.tensors['images'],
            labels=self.tensors['labels'],
            is_train=is_train,
            feature_name=self.feature_name,
            BoW_loss_coef=self.BoW_loss_coef,
            cls_loss_coef=self.cls_loss_coef)

        return record

    def apply_kmeans_to_dataset(
        self,
        dataloader,
        num_embeddings,
        feature_name=None,
        memmap=False,
        memmap_dir=None):
        feature_extractor = self.networks['feature_extractor']
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f'==> Extract features from dataset.')

        if memmap_dir is not None:
            os.makedirs(memmap_dir, exist_ok=True)
            tempfile.tempdir = memmap_dir
        with tempfile.TemporaryFile() as fp:
            all_features_dataset, _ = cluster_utils.extract_features_from_dataset(
                feature_extractor=feature_extractor,
                dataloader_iterator=dataloader_iterator,
                feature_name=feature_name,
                logger=self.logger,
                memmap_filename=(fp if memmap else None))

            # clustering algorithm to use
            self.logger.info(f'==> Apply kmeans to dataset')
            deepcluster = cluster_utils.Kmeans(
                num_embeddings, preprocess=False)
            clustering_loss = deepcluster.cluster(
                all_features_dataset, verbose=True)
            centroids = deepcluster.centroids
            cluster_size = deepcluster.cluster_size
            points2ids = deepcluster.point2ids

            vector_quantizer = cluster_utils.initialize_vector_quantizer(
                centroids=centroids, cluster_size=cluster_size,
                commitment_cost=0.25, decay=0.99, epsilon=1e-5)
            vector_quantizer = vector_quantizer.to(self.device)

            prefix = f'vector_quantizer_kmeansK{num_embeddings}'
            filename = self._get_net_checkpoint_filename(
                prefix, self.curr_epoch)
            state = {
                'epoch': self.curr_epoch,
                'network': vector_quantizer.state_dict(),
                'metric': None,}
            self.logger.info(
                f'==> Saving vector quantizer with kmeans centroids to {filename}')
            torch.save(state, filename)
