import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BalancedImageDataGenerator:

    def __init__(self, **image_data_generator_kwargs):
        self.augmentor = ImageDataGenerator(**image_data_generator_kwargs)


    def flow(self, pos_xs, pos_ys, neg_xs, neg_ys, desired_pos_ratio, batch_size):
        pos_count = int(batch_size * desired_pos_ratio)
        neg_count = int(batch_size * (1-desired_pos_ratio))

        pos_generator = self.augmentor.flow(pos_xs, pos_ys, batch_size=pos_count)
        neg_generator = self.augmentor.flow(neg_xs, neg_ys, batch_size=neg_count)

        batch_idx = np.arange(batch_size)
        while True:
            pos_xs, pos_ys = next(pos_generator)
            neg_xs, neg_ys = next(neg_generator)

            batch_xs = np.concatenate((pos_xs, neg_xs))
            batch_ys = np.concatenate((pos_ys, neg_ys))

            np.random.shuffle(batch_idx)
            yield batch_xs[batch_idx], batch_ys[batch_idx]