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


if __name__=="__main__":
    import time

    desired_pos_ratio = 0.5
    batch_size = 20

    pos_xs = np.random.normal(size=[160, 32, 32, 1])
    pos_ys = np.ones([160])

    neg_xs = np.random.normal(size=[840, 32, 32, 1])
    neg_ys = np.zeros([840])

    datagen = BalancedImageDataGenerator()

    gen = datagen.flow(pos_xs, pos_ys, neg_xs, neg_ys, desired_pos_ratio, batch_size)

    times = []
    ratios = []
    for i in range(20000):
        start = time.time()
        xs, ys = next(gen)
        ratios.append(ys.mean())
        times.append(time.time()-start)

    print(f"Average batch ratio {np.mean(ratios)}")
    print(f"Batches take on average {np.mean(times)} seconds")


