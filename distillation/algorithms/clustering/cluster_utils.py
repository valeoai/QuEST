import time

import faiss
import random
import numpy as np
import torch
import distillation.algorithms.classification.utils as cls_utils

from tqdm import tqdm
from distillation.architectures.miscellaneous.vector_quantization import VectorQuantizerEMA


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False, useFloat16=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    print('==> Initialize faiss Clustering.')
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)
    if verbose:
        print(f'==> k-means seed: {clus.seed}')

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = useFloat16
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = np.array([clus.iteration_stats.at(i).obj for i in range(clus.niter)]) / n_data
    if verbose:
        print(f'==> k-means loss evolution: {losses}')

    centroids = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, d)

    return [int(n[0]) for n in I], losses[-1], centroids


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k, preprocess=False):
        self.k = k
        self.preprocess = preprocess

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        if self.preprocess:
            print('==> Preprocess data...')
            xb = preprocess_features(data)
        else:
            print('==> Convert data to float32')
            xb = data if (data.dtype == 'float32') else data.astype('float32')

        # cluster the data
        print('==> Start kmeans clustering.')
        I, loss, centroids = run_kmeans(xb, self.k, verbose)
        self.centroids = centroids
        self.images_lists = [[] for i in range(self.k)]
        self.point2ids = I

        for i in range(len(data)):
            cluster_id = I[i]
            self.images_lists[cluster_id].append(i)

        self.cluster_size = [len(self.images_lists[i]) for i in range(self.k)]

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def extract_features_from_dataset(
    feature_extractor,
    dataloader_iterator,
    feature_name=None,
    downsample2=False,
    logger=None,
    memmap_filename=None):

    if isinstance(feature_name, (list, tuple)):
        assert len(feature_name) == 1

    feature_extractor.eval()

    num_clusters = None

    all_features_dataset = None
    count = 0
    shape_per_img = [1, 1]
    for i, batch in enumerate(tqdm(dataloader_iterator)):
        with torch.no_grad():
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.cuda()
            assert images.dim()==4

            features = cls_utils.extract_features(
                feature_extractor, images, feature_name=feature_name)

            if (features.dim() == 4 and
                features.size(2) == 1 and
                features.size(3) == 1):
                features = features.view(features.size(0), -1)
            elif features.dim() == 4:
                if downsample2:
                    x_offset = 0 if (random.uniform(0, 1) < 0.5) else 1
                    y_offset = 0 if (random.uniform(0, 1) < 0.5) else 1
                    features = (
                        features[:, :, y_offset::2, x_offset::2].contiguous())

                features = features.permute(0, 2, 3, 1).contiguous()
                shape_per_img = [1, features.size(1), features.size(2)]
                features = features.view(-1, features.size(3))

            batch_size = features.size()[0]
            features_shape = list(features.size()[1:])
            features = features.cpu().numpy()

        if all_features_dataset is None:
            max_count = len(dataloader_iterator) * batch_size
            dataset_shape = [max_count,] + features_shape
            if memmap_filename is None:
                all_features_dataset = np.zeros(dataset_shape, dtype='float32')
            else:
                if logger:
                    logger.info(f'Use memmap with file: {memmap_filename}')
                all_features_dataset = np.memmap(
                    memmap_filename, shape=tuple(dataset_shape),
                    dtype='float32', mode='w+')

            if logger:
                logger.info(f'image size: {images.size()}')
                logger.info(f'feature shape: {features_shape}')
                logger.info(f'max count: {max_count}')
                logger.info(f'batch size: {batch_size}')


        all_features_dataset[count:(count + batch_size)] = features
        count += batch_size

    if memmap_filename is None:
        all_features_dataset = all_features_dataset[:count]
    else:
        dataset_shape[0] = count
        del all_features_dataset
        all_features_dataset = np.memmap(
            memmap_filename, shape=tuple(dataset_shape),
            dtype='float32', mode='r')

    if logger:
        logger.info(f'Shape of extracted dataset: {all_features_dataset.shape}')

    return all_features_dataset, shape_per_img


def initialize_vector_quantizer(
    centroids,
    cluster_size,
    commitment_cost=0.25,
    decay=0.99,
    epsilon=1e-5):

    vector_quantizer = VectorQuantizerEMA(
        num_embeddings=centroids.shape[0],
        embedding_dim=centroids.shape[1],
        commitment_cost=commitment_cost,
        decay=decay,
        epsilon=epsilon)
    vector_quantizer.eval()
    with torch.no_grad():
        centroids = torch.from_numpy(centroids)
        vector_quantizer._embedding.weight.copy_(centroids)
        ema_cluster_size = torch.Tensor(cluster_size)
        vector_quantizer._ema_cluster_size.copy_(ema_cluster_size)
        ema_w = centroids * ema_cluster_size.unsqueeze(1)
        vector_quantizer._ema_w.copy_(ema_w)

    return vector_quantizer
