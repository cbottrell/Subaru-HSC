import tensorflow as tf

# comet doesn't work with tf 2.0 so I had to write it
class CometLogger(tf.keras.callbacks.BaseLogger):
    def __init__(self, experiment):
        super(CometLogger, self).__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        print(logs)
        for k in logs:
            if "val" in k:
                with self.experiment.test():
                    self.experiment.log_metric(k.replace("val_", ""), logs[k], step=epoch)
            else:
                with self.experiment.train():
                    self.experiment.log_metric(k, logs[k], step=epoch)
