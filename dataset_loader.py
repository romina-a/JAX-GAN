import numpy as np
import jax.numpy as jnp

# ~~~~~~~~~~~~~~~~~~~~~~~~~ UTILS FOR LOADING MNIST WITH PYTORCH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def get_NumpyLoader(batch_size, digit=None):
    data_adr = ""
    mnist_dataset = MNIST('./tmp/mnist/', download=True, transform=FlattenAndCast())
    # load training with the generator (makes batch easier I think)
    if digit is not None:
        idx = mnist_dataset.targets == digit
        mnist_dataset.data = mnist_dataset.data[idx]
        mnist_dataset.targets = mnist_dataset.targets[idx]
    training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
    return training_generator


# ~~~~~~~~~~~~~~~~~~~~~~~~ UTILS FOR LOADING MNIST WITH TF ~~~~~~~~~~~~~~~~~
import tensorflow_datasets as tfds
import tensorflow as tf


def make_mnist_dataset(batch_size, seed=1, digit=None):
    mnist = tfds.load("mnist")

    def _preprocess(sample):
        image = tf.image.convert_image_dtype(sample["image"], tf.float32)
        image = tf.image.resize(image, (32, 32))
        return 2.0 * image - 1.0

    ds = mnist["train"]
    if digit is not None:
        ds = ds.filter(lambda fd: fd['label'] == digit)
    ds = ds.map(map_func=_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(10 * batch_size, seed=seed).repeat().batch(batch_size)
    return iter(tfds.as_numpy(ds))


def make_cifar10_dataset(batch_size, seed=1, digit=None):
    cifar10 = tfds.load("cifar10")

    def _preprocess(sample):
        image = tf.image.convert_image_dtype(sample["image"], tf.float32)
        image = tf.image.resize(image, (32, 32))
        return 2.0 * image - 1.0

    ds = cifar10["train"]
    if digit is not None:
        ds = ds.filter(lambda fd: fd['label'] == digit)
    ds = ds.map(map_func=_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(10 * batch_size, seed=seed).repeat().batch(batch_size)
    return iter(tfds.as_numpy(ds))
