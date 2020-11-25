"""Trainer class for all training-related stuffs."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """This class contains modules for training, testing, validation, loading checkpoints & other train-sy stuffs."""

    def __init__(self, model, train_dataset, batch_size, learning_rate):
        """Initialize model, dataloader class, batch size and learning rate."""
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.model = model
        self.dataset_size = {"train": len(train_dataset)}

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": learning_rate},
            {"params": model.get_10x_lr_params(), "lr": learning_rate * 10},
        ]

        self.optimizer = torch.optim.SGD(
            train_params, momentum=0.9, weight_decay=5e-4, nesterov=False
        )
        # TODO: Setting a dummy value of 50 now. Decays the learning rate after the 50th epoch by a factor of gamma.
        #  Should contain (in future) a comma separated list of epoch values where it is intended to decay the
        #  learning rate.
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[int(x) for x in "50".split(",")], gamma=0.1
        )
        self.batch_size = batch_size

    def training(self, epoch):
        """Train, train, train."""
        train_loss = 0.0
        self.model.train()
        # num_img_tr = len(self.train_dataloader)
        # visualization_index = int(random.random() * len(self.train_dataloader))
        # vis_img, vis_tgt, vis_out = None, None, None
        progress_bar = tqdm(self.train_dataloader)

        for i, sample in enumerate(progress_bar):
            image, target = sample["image"], sample["label"]
            self.optimizer.zero_grad()
            output = self.model(
                image
            )  # the values are negative, what is the model outputting?
            loss = self.cross_entropy_loss(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description("Train loss: %.3f" % (train_loss / (i + 1)))

        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.batch_size + image.data.shape[0])
        )
        print("Loss: %.3f" % train_loss)

    def cross_entropy_loss(self, logit, target):
        """Define cross entropy loss, with ignore_index as an extra parameter to mask out labels not selected."""
        n, c, h, w = logit.size()
        criterion = torch.nn.CrossEntropyLoss(
            weight=None, ignore_index=255, reduction="mean"
        )
        loss = criterion(logit, target.long())
        loss /= n

        return loss
