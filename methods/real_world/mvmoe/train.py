import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_autocast_context(opts):
    enabled = opts.use_cuda and opts.train_precision in ('bf16', 'fp16')
    dtype = torch.bfloat16 if opts.train_precision == 'bf16' else torch.float16
    return torch.amp.autocast('cuda', enabled=enabled, dtype=dtype)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, grad_scaler=None):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    optimizer.zero_grad()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        is_update_step = (
            (batch_id + 1) % opts.grad_accumulation_steps == 0
            or (batch_id + 1) == len(training_dataloader)
        )
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            grad_scaler,
            is_update_step
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
                'grad_scaler': grad_scaler.state_dict() if grad_scaler is not None else None
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        grad_scaler=None,
        is_update_step=True
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    with get_autocast_context(opts):
        # Evaluate model, get costs and log probabilities
        cost, log_likelihood, distance, tardiness = model(x)

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        moe_aux_loss = get_inner_model(model).moe_aux_loss
        loss = reinforce_loss + bl_loss + opts.moe_aux_loss_weight * moe_aux_loss
        backward_loss = loss / opts.grad_accumulation_steps

    # Perform backward pass and optimization step
    if grad_scaler is not None:
        grad_scaler.scale(backward_loss).backward()
    else:
        backward_loss.backward()

    if is_update_step:
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    else:
        grad_norms = ([torch.tensor(0.0, device=opts.device)], [torch.tensor(0.0, device=opts.device)])

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, distance, tardiness, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
        print('moe_aux_loss: {}'.format(moe_aux_loss.item()))
        if not opts.no_tensorboard:
            tb_logger.log_value('moe_aux_loss', moe_aux_loss.item(), step)

