#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
import json

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        # save the last checkpoints
        if self.cfg.MODEL.SAVE_CKPT_FINALRUNS:
            Checkpointer(
                self.model,
                save_dir=self.cfg.OUTPUT_DIR,
                save_to_disk=True
            ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

    
    def calculate_importance(self, cfg, model, train_loader, n_pieces_token=16, n_soft_tokens=20):
        """
        Train a classifier using epoch
        """
        
        for name, _ in model.named_parameters():
            print(name)
        list(model.parameters())
        
        prompt_model = model
        self.cls_weights = train_loader.dataset.get_class_weights(self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # TODO:  其实上面有包含 这里这么写冗余了 之后可以考虑去掉
        Checkpointer(
            prompt_model
        ).load(cfg.OUTPUT_DIR + '/last_model.pth') # /home/ch7858/vpt/output_val/vtab-caltech101_P32_VK5_SHARED_1_INIT_1_ACC_0/sup_vitb16_224/lr12.5_wd0.01/run1/
        # print('path', cfg.OUTPUT_DIR)
        prompt_model.eval()
        soft_tokens_importance = torch.zeros(n_soft_tokens).cuda()
        soft_tokens_pieces_importance = torch.zeros(n_soft_tokens, n_pieces_token).cuda()
        
        total_len = 0
        for step, inputs in enumerate(train_loader):
            X, targets = self.get_input(inputs)

            loss, _ = self.forward_one_batch(X, targets, True)
            
            # soft_tokens_importance += prompt_model.enc.transformer.prompt_embeddings.grad
            # print(prompt_model.enc.transformer.prompt_soft_tokens_mask_cls_token.grad)
            if self.cfg.MODEL.P_VK.MASK_CLS_TOKEN_ON_VK == False:
                soft_tokens_importance += prompt_model.enc.transformer.prompt_soft_tokens_mask_cls_token.grad
                for token_i in range(n_soft_tokens):
                    # changed here
                    soft_tokens_pieces_importance[token_i] += prompt_model.enc.transformer.prompt_soft_tokens_pieces_mask_cls_token.grad[token_i]
                total_len += 1
            else:
                # 这里也会不一样 layer变了
                # print('pass self.cfg.MODEL.P_VK.MASK_CLS_TOKEN_ON_VK at trainer')
                # print('0', soft_tokens_importance)
                # print('sss', prompt_model.enc.transformer.encoder.prompt_soft_tokens_mask_cls_token.grad)
                soft_tokens_importance += prompt_model.enc.transformer.encoder.prompt_soft_tokens_mask_cls_token.grad
                for token_i in range(n_soft_tokens):
                    # print('ggg', prompt_model.enc.transformer.encoder.prompt_soft_tokens_pieces_mask_cls_token.grad)
                    soft_tokens_pieces_importance[token_i] += prompt_model.enc.transformer.encoder.prompt_soft_tokens_pieces_mask_cls_token.grad[token_i]
                total_len += 1
            
        soft_tokens_importance /= total_len
        soft_tokens_pieces_importance /= total_len
        
        
        # normalize_scores_by_token
        if self.cfg.MODEL.P_VK.NORMALIZE_SCORES_BY_TOKEN:
            exp = 2
            norm_by_token = torch.pow(torch.pow(soft_tokens_importance, exp).sum(), 1/exp)
            soft_tokens_importance /= norm_by_token.unsqueeze(-1) + 1e-20
            norm_by_token_piece = torch.pow(torch.pow(soft_tokens_pieces_importance, exp).sum(-1), 1/exp)
            soft_tokens_pieces_importance /= norm_by_token_piece.unsqueeze(-1) + 1e-20
            
        return soft_tokens_importance, soft_tokens_pieces_importance
        
    def determine_mask_sequence(self, cfg, n_pieces_token=16, n_soft_tokens=20):
        soft_token_mask_number = cfg.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT_NUM
        if soft_token_mask_number is None:
            soft_token_mask_number = []
            # print(cfg.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT)
            for prune_percent in cfg.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT:
                total_soft_tokens = n_soft_tokens
                n_to_mask = int(total_soft_tokens * prune_percent / 100)
                if cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN > 0:
                    if n_to_mask > total_soft_tokens - cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN:
                        n_to_mask = total_soft_tokens - cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN
                        soft_token_mask_number.append(n_to_mask)
                        break
                soft_token_mask_number.append(n_to_mask)
        soft_token_mask_number = sorted(soft_token_mask_number)
        soft_token_mask_sequence = soft_token_mask_number[:]
        for idx in range(1, len(soft_token_mask_number)):
            soft_token_mask_sequence[idx] = soft_token_mask_number[idx] - soft_token_mask_number[idx-1]
        assert soft_token_mask_number[-1] == sum(soft_token_mask_sequence)
        
        token_piece_mask_number = cfg.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT_NUM
        if token_piece_mask_number is None:
            token_piece_mask_number = []
            for prune_percent in cfg.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT:
                total_soft_tokens_pieces = n_pieces_token
                n_to_mask = int(total_soft_tokens_pieces * prune_percent / 100)
                if cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE > 0:
                    if n_to_mask > total_soft_tokens_pieces - cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE:
                        n_to_mask = total_soft_tokens_pieces - cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE
                        token_piece_mask_number.append(n_to_mask)
                        break
                token_piece_mask_number.append(n_to_mask)
        token_piece_mask_number = sorted(token_piece_mask_number)
        token_piece_mask_sequence = token_piece_mask_number[:]
        for idx in range(1, len(token_piece_mask_number)):
            token_piece_mask_sequence[idx] = token_piece_mask_number[idx] - token_piece_mask_number[idx-1]
        
        assert token_piece_mask_number[-1] == sum(token_piece_mask_sequence)
        
        # print('1', soft_token_mask_sequence)
        # print('2', token_piece_mask_sequence)
        # print('3', soft_token_mask_number)
        # print('4', token_piece_mask_number)
        return soft_token_mask_sequence, token_piece_mask_sequence
    
    def dump(self, path, data, convert_key_type=False):
        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj.item(), np.integer):
                return int(obj.item())
            elif isinstance(obj.item(), float):
                return obj.item()
            raise TypeError

        if convert_key_type:
            d = {}
            for k, v1 in data.items():
                v1 =  {int(k):v2 for k,v2 in v1.items()}
                d[int(k)] = v1
        else:
            d = data
        
        with open(path, 'w') as f:
            json.dump(d, f, default=set_default)
    
    def what_tokens_pieces_to_mask(self,
                 soft_tokens_pieces_importance, 
                 n_to_mask, 
                 soft_tokens_pieces_to_mask, 
                 min_num_soft_tokens_pieces, 
                 n_pieces_token = 16, 
                 n_soft_tokens = 20,
                 reverse = False
                ):
        assert n_soft_tokens == soft_tokens_pieces_importance.size()[0]
        assert n_pieces_token == soft_tokens_pieces_importance.size()[1]
        
        for soft_token_idx in range(n_soft_tokens):
            score = soft_tokens_pieces_importance[soft_token_idx]
            
            if soft_token_idx not in soft_tokens_pieces_to_mask:
                soft_tokens_pieces_to_mask[soft_token_idx] = set()
            soft_token_pieces_and_score = [
                (soft_token_piece, score[soft_token_piece]) 
                for soft_token_piece in range(n_pieces_token)
            ]
            soft_token_pieces_and_score = sorted(soft_token_pieces_and_score, key=lambda x:x[1], reverse=reverse)
            sorted_soft_token_pieces = [soft_token_piece_and_score[0] for soft_token_piece_and_score in soft_token_pieces_and_score]
            
            sorted_soft_token_pieces = [
                soft_token_piece
                for soft_token_piece in sorted_soft_token_pieces
                if soft_token_piece not in soft_tokens_pieces_to_mask[soft_token_idx]
            ]
            for soft_token_piece in sorted_soft_token_pieces[:n_to_mask]:
                soft_tokens_pieces_to_mask[soft_token_idx].add(soft_token_piece)
    
        return soft_tokens_pieces_to_mask


    def what_tokens_to_mask(self,
                    soft_tokens_importance, 
                    n_to_mask, 
                    soft_tokens_to_mask, 
                    min_num_soft_tokens, 
                    n_pieces_token = 16, 
                    n_soft_tokens = 20,
                    reverse = False
                    ):
        soft_tokens_and_score = [(soft_token, soft_tokens_importance[soft_token]) for soft_token in range(n_soft_tokens)]
        soft_tokens_and_score = sorted(soft_tokens_and_score, key=lambda x:x[1], reverse=reverse)
        sorted_soft_tokens = [soft_token_and_score[0] for soft_token_and_score in soft_tokens_and_score]
        sorted_soft_tokens = [
                soft_token
                for soft_token in sorted_soft_tokens
                if soft_token not in soft_tokens_to_mask
        ]
        for soft_token in sorted_soft_tokens[:n_to_mask]:
            soft_tokens_to_mask.add(soft_token)
        
        return soft_tokens_to_mask
