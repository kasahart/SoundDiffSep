from asteroid.engine.system import System
import torch
from schedulefree import RAdamScheduleFree


class TwoChSepSystem(System):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        lr=1e-4,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            config=config,
        )
        self.lr = lr


    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        mixed, noise_clean, tgt = batch
        _mixed = torch.stack([mixed, noise_clean], dim=1)
        est_targets = self(_mixed)
        loss = self.loss_func(est_targets, tgt.unsqueeze(1))
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb, train=True)
        loss = loss.mean()
        self.log("loss", loss, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb, train=False)
        self.log("val_loss", loss.mean(), logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # schedule-freeなoptimizerを例としてAdamWScheduleFreeを使用
        optimizer = RAdamScheduleFree(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer

    # 以下フックでoptimizerのモードを適切に切り替える
    def on_train_epoch_start(self):
        # 学習時はoptimizerをtrainモードに
        opt = self.optimizers(use_pl_optimizer=False)
        if hasattr(opt, "train"):
            opt.train()

    def on_validation_epoch_start(self):
        # 検証時はoptimizerをevalモードに
        opt = self.optimizers(use_pl_optimizer=False)
        if hasattr(opt, "eval"):
            opt.eval()

    def on_test_epoch_start(self):
        # テスト時もevalモード
        opt = self.optimizers(use_pl_optimizer=False)
        if hasattr(opt, "eval"):
            opt.eval()

    def on_predict_epoch_start(self):
        # 推論時もevalモード
        opt = self.optimizers(use_pl_optimizer=False)
        if hasattr(opt, "eval"):
            opt.eval()

    # チェックポイント保存時にもevalモードが推奨
    def on_save_checkpoint(self, checkpoint):
        opt = self.optimizers(use_pl_optimizer=False)
        if hasattr(opt, "eval"):
            opt.eval()
