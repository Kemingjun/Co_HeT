import os
import time
import json
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from amp_utils import (
    autocast_context,
    require_finite_tensors,
    scaled_optimizer_step,
)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def _bytes_to_mib(value):
    return value / (1024 ** 2)


def _bytes_to_gib(value):
    return value / (1024 ** 3)


def _reset_cuda_peak_memory(opts):
    if not opts.use_cuda:
        return
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_id)


def _cuda_memory_snapshot(opts):
    if not opts.use_cuda:
        return None

    torch.cuda.synchronize(opts.device)
    devices = []
    for device_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(device_id)
        reserved = torch.cuda.max_memory_reserved(device_id)
        devices.append({
            'device': device_id,
            'name': torch.cuda.get_device_name(device_id),
            'max_allocated_bytes': allocated,
            'max_reserved_bytes': reserved,
            'max_allocated_mib': _bytes_to_mib(allocated),
            'max_reserved_mib': _bytes_to_mib(reserved),
            'max_allocated_gib': _bytes_to_gib(allocated),
            'max_reserved_gib': _bytes_to_gib(reserved),
        })

    return {
        'devices': devices,
        'total_max_allocated_bytes': sum(d['max_allocated_bytes'] for d in devices),
        'total_max_reserved_bytes': sum(d['max_reserved_bytes'] for d in devices),
        'total_max_allocated_mib': sum(d['max_allocated_mib'] for d in devices),
        'total_max_reserved_mib': sum(d['max_reserved_mib'] for d in devices),
        'total_max_allocated_gib': sum(d['max_allocated_gib'] for d in devices),
        'total_max_reserved_gib': sum(d['max_reserved_gib'] for d in devices),
    }


def _append_memory_record(opts, record):
    if not opts.use_cuda:
        return

    path = os.path.join(opts.save_dir, 'gpu_memory.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            payload = json.load(f)
    else:
        payload = {
            'ablation_variant': getattr(opts, 'ablation_variant', None),
            'run_name': opts.run_name,
            'device_count': torch.cuda.device_count(),
            'epochs': [],
            'summary': {}
        }

    payload['epochs'].append(record)
    train_records = [r['train'] for r in payload['epochs'] if r.get('train') is not None]
    if train_records:
        peak_train_allocated = max(r['total_max_allocated_bytes'] for r in train_records)
        peak_train_reserved = max(r['total_max_reserved_bytes'] for r in train_records)
        payload['summary'] = {
            'peak_train_allocated_bytes': peak_train_allocated,
            'peak_train_reserved_bytes': peak_train_reserved,
            'peak_train_allocated_mib': _bytes_to_mib(peak_train_allocated),
            'peak_train_reserved_mib': _bytes_to_mib(peak_train_reserved),
            'peak_train_allocated_gib': _bytes_to_gib(peak_train_allocated),
            'peak_train_reserved_gib': _bytes_to_gib(peak_train_reserved),
        }

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def _print_cuda_memory(prefix, snapshot):
    if snapshot is None:
        return
    print(
        "{} GPU memory peak: allocated={:.2f} MiB ({:.3f} GiB), reserved={:.2f} MiB ({:.3f} GiB)".format(
            prefix,
            snapshot['total_max_allocated_mib'],
            snapshot['total_max_allocated_gib'],
            snapshot['total_max_reserved_mib'],
            snapshot['total_max_reserved_gib'],
        )
    )


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
            with autocast_context(torch, opts.precision):
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


def train_epoch(
        model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem,
        tb_logger, opts, scaler):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    _reset_cuda_peak_memory(opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()  # 设置模型为训练模式
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

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
            scaler
        )

        step += 1

    epoch_duration = time.time() - start_time
    train_memory = _cuda_memory_snapshot(opts)
    _print_cuda_memory("Epoch {} train".format(epoch), train_memory)
    _append_memory_record(opts, {
        'epoch': epoch,
        'epoch_duration_seconds': epoch_duration,
        'graph_size': opts.graph_size,
        'batch_size': opts.batch_size,
        'epoch_size': opts.epoch_size,
        'ablation_variant': getattr(opts, 'ablation_variant', None),
        'precision': opts.precision,
        'checkpoint_encoder': opts.checkpoint_encoder,
        'amp_overflow_count': getattr(opts, 'amp_overflow_count', 0),
        'grad_scale': getattr(opts, 'final_grad_scale', 1.0),
        'train': train_memory,
    })
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'grad_scaler': scaler.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
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
        scaler
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)   # 将变量移动到指定的设备上
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    optimizer.zero_grad(set_to_none=True)

    # Keep model-heavy kernels under autocast while calculating the policy loss in FP32.
    with autocast_context(torch, opts.precision):
        cost, log_likelihood, distance, tardiness = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    cost = cost.float()
    log_likelihood = log_likelihood.float()
    distance = distance.float()
    tardiness = tardiness.float()
    if torch.is_tensor(bl_val):
        bl_val = bl_val.float()
    if torch.is_tensor(bl_loss):
        bl_loss = bl_loss.float()
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    require_finite_tensors(
        cost=cost,
        log_likelihood=log_likelihood,
        distance=distance,
        tardiness=tardiness,
        loss=loss,
    )

    max_norm_threshold = 1e4  # 设置梯度“异常”的判断标准

    def inspect_and_clip_gradients():
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if not math.isfinite(grad_norm):
                    print(f"[非有限梯度警告] 参数名: {name}")
                elif grad_norm > max_norm_threshold:
                    print(f"[梯度过大警告] 参数名: {name}, 梯度范数: {grad_norm:.2e}")
        return clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    step_result = scaled_optimizer_step(
        loss=loss,
        optimizer=optimizer,
        scaler=scaler,
        clip_callback=inspect_and_clip_gradients,
    )
    grad_norms = step_result['grad_norms']
    if step_result['step_skipped']:
        opts.amp_overflow_count = getattr(opts, 'amp_overflow_count', 0) + 1
    opts.final_grad_scale = step_result['scale_after']

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, distance, tardiness, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
