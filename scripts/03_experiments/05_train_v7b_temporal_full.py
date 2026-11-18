#!/usr/bin/env python3
"""Entrenamiento completo V7b con penalizacion temporal diferenciable.

Experimento separado del V7b original y del piloto de ajuste fino.
Entrena desde CERO, usa FIT/MONITOR por meses, ranking con mineria dinamica
de negativos, early stopping y evaluacion final en el fold externo.

Solo datos MAG/SWEA previamente guardados en cache; sin consultas BigQuery.
La MPB catalogada se usa SOLO en la loss FIT y en MONITOR para seleccionar
checkpoint, nunca como caracteristica de entrada ni para elegir hiperparametros
en validacion externa.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.v7b import MagSweaRegionAutoencoder
from src.model.ranking import (
    event_pairwise_ranking_loss, select_rank_event_groups,
    event_group_batches, update_hard_memory, numpy_event_rank_metrics,
)
from src.preprocessing import bigquery_source as bq
from src.preprocessing import cv_splits
from src.preprocessing.catalog import read_manifest
from src.preprocessing.mag_cache import MagEventCache
from src.preprocessing.swea_cache import SweaEventCache
from src.preprocessing.native_windows import fit_unique_epochs


def load_trainer():
    src = ROOT / 'scripts/02_training/01_train_v7b.py'
    spec = importlib.util.spec_from_file_location('v7b_original', src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path,
                   default=ROOT/'data/splits/cv4_monthly_manifest.csv')
    p.add_argument('--mag-cache-dir', type=Path,
                   default=ROOT/'data/cache/mag_events')
    p.add_argument('--swea-cache-dir', type=Path,
                   default=ROOT/'data/cache/swea_events')
    p.add_argument('--output-dir', type=Path,
                   default=ROOT/'results/metrics/v7b_temporal_full_weight5')
    p.add_argument('--folds', choices=('all', '1', '2', '3', '4'), default='all')
    p.add_argument('--max-epochs', type=int, default=1000)
    p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--region-weight', type=float, default=0.5)
    p.add_argument('--reconstruction-weight', type=float, default=0.15)
    p.add_argument('--rank-weight', type=float, default=2.)
    p.add_argument('--rank-margin', type=float, default=0.25)
    p.add_argument('--rank-control-step-s', type=float, default=120.)
    p.add_argument('--rank-positives-per-event', type=int, default=2)
    p.add_argument('--rank-hard-negatives', type=int, default=2)
    p.add_argument('--rank-random-negatives', type=int, default=2)
    p.add_argument('--evaluation-stride-s', type=float, default=32.)
    p.add_argument('--temporal-weight', type=float, default=5.)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--temporal-scale-s', type=float, default=600.)
    p.add_argument('--resume', action='store_true',
                   help='Reanudar folds en curso o verificar folds terminados.')
    a = p.parse_args()
    positive = (a.max_epochs, a.min_epochs, a.patience,
                a.learning_rate, a.batch_size, a.threads, a.rank_weight,
                a.rank_control_step_s, a.evaluation_stride_s,
                a.temperature, a.temporal_scale_s,
                a.rank_positives_per_event, a.rank_hard_negatives,
                a.rank_random_negatives)
    if (min(positive) <= 0 or a.min_epochs > a.max_epochs
            or min(a.region_weight, a.reconstruction_weight,
                   a.rank_margin, a.temporal_weight) < 0):
        p.error('Hiperparametros o epocas invalidos.')
    if a.batch_size < (
        a.rank_positives_per_event + a.rank_hard_negatives
        + a.rank_random_negatives
    ):
        p.error('El batch debe admitir un grupo de ranking completo.')
    return a


def loss_temporal(logits, examples, *, temperature, scale_s):
    """E_softmax(|centro - tMPB|)/scale_s, promedio equilibrado por evento."""
    by_event = defaultdict(list)
    for i, item in enumerate(examples):
        by_event[item[0]].append((i, item[1].center_s-item[0].timestamp()))
    if not by_event:
        raise ValueError('No hay grupos para la loss temporal.')
    event_losses = []
    for pairs in by_event.values():
        idx = torch.tensor([i for i, _ in pairs], device=logits.device)
        dist = torch.tensor([abs(d)/scale_s for _, d in pairs],
                            device=logits.device, dtype=logits.dtype)
        scores = logits.index_select(0, idx)
        p = torch.softmax(scores/temperature, dim=0)
        event_losses.append((p*dist).sum())
    return torch.stack(event_losses).mean()


def load_cached_blocks(rows, args, trainer):
    """Exige cobertura TOTAL del catalogo elegible. Nunca descarga ni consulta."""
    queries = trainer.query_sources()
    cache = {
        'MAG': MagEventCache(
            args.mag_cache_dir, project=bq.PROJECT,
            table=bq.MAG, sql=queries['MAG']
        ),
        'SWEA': SweaEventCache(
            args.swea_cache_dir, project=bq.PROJECT,
            table=bq.SWEA+' + '+bq.AXES, sql=queries['SWEA']
        ),
    }
    blocks = {}
    missing = []
    for i, row in enumerate(rows, 1):
        start, end = trainer.event_bounds(row)
        parts = {}
        for name in ('MAG', 'SWEA'):
            limit = 25000 if name == 'MAG' else 2000
            found = cache[name].load(
                row['time'], start, end, max_rows=limit
            )
            if found is None or not len(found[0]):
                missing.append((row['time'].isoformat(), name))
            else:
                parts[name] = found
        if len(parts) == 2:
            blocks[row['time']] = parts
        if i % 50 == 0 or i == len(rows):
            print(f'Cache {i}/{len(rows)} eventos inspeccionados.', flush=True)
    if missing:
        detail = ', '.join(f'{ts} {sensor}' for ts, sensor in missing[:8])
        raise RuntimeError(
            f'Cobertura de cache insuficiente: {len(missing)} intervalos '
            f'MAG/SWEA faltantes para {len(rows)} eventos. Primeros: {detail}. '
            'Para preservar una evaluacion completa no se omiten eventos ni '
            'se realizan consultas BigQuery de manera implicita.'
        )
    return blocks


def experiment_signature(args, fold, fit, monitor, val):
    options = {
        'experiment': 'v7b-full-training-temporal-v1',
        'fold': fold,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'fit': [v['time'].isoformat() for v in fit],
        'monitor': [v['time'].isoformat() for v in monitor],
        'val': [v['time'].isoformat() for v in val],
        'optimization': {
            k: getattr(args, k) for k in (
                'max_epochs', 'min_epochs', 'patience', 'learning_rate',
                'batch_size', 'seed', 'region_weight', 'reconstruction_weight',
                'rank_weight', 'rank_margin', 'rank_control_step_s',
                'rank_positives_per_event', 'rank_hard_negatives',
                'rank_random_negatives', 'evaluation_stride_s',
                'temporal_weight', 'temperature', 'temporal_scale_s'
            )
        }
    }
    return hashlib.sha256(
        json.dumps(options, sort_keys=True).encode()
    ).hexdigest()


def train_one_epoch(model, optimizer, fit_pool, norm, cfg, trainer,
                    epoch, fold, hard_memory):
    groups = select_rank_event_groups(
        fit_pool, epoch=epoch, seed=cfg.seed+fold*10000,
        hard_memory=hard_memory,
        n_positives=cfg.rank_positives_per_event,
        n_hard=cfg.rank_hard_negatives,
        n_random=cfg.rank_random_negatives
    )
    if not groups:
        raise RuntimeError('No hay eventos FIT con positivo y control.')
    totals = defaultdict(float)
    steps = 0
    model.train()
    for examples in event_group_batches(groups, cfg.batch_size):
        batch = trainer.batch_for_examples(examples, norm, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        output, cls, terms = trainer.classification_losses(
            model, batch, cfg
        )
        rank, accuracy, top1, _ = event_pairwise_ranking_loss(
            output['transition_logit'], batch['transition'],
            [x[0].isoformat() for x in examples], margin=cfg.rank_margin
        )
        temporal = loss_temporal(
            output['transition_logit'], examples,
            temperature=cfg.temperature, scale_s=cfg.temporal_scale_s
        )
        total = cls + cfg.rank_weight*rank + cfg.temporal_weight*temporal
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Loss no finita.')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        update_hard_memory(
            hard_memory, examples, output['transition_logit'],
            n_hard=cfg.rank_hard_negatives
        )
        values = {
            **terms, 'rank': float(rank.detach()),
            'temporal': float(temporal.detach()),
            'total': float(total.detach()),
            'pair_accuracy': accuracy,
            'event_top1': top1
        }
        for name, v in values.items():
            totals[name] += v
        steps += 1
    return {k: v/steps for k,v in totals.items()}


@torch.no_grad()
def monitor_one_epoch(model, monitor_pool, norm, cfg, trainer):
    """Loss completa en MONITOR sin gradientes ni uso del fold externo."""
    examples = trainer.entries_from_candidates(monitor_pool)
    if not examples:
        raise RuntimeError('Sin ejemplos de MONITOR.')
    model.eval()
    totals = defaultdict(float)
    logits = []
    scores = []
    labels = []
    ranking_rows = []
    for start in range(0, len(examples), cfg.batch_size):
        chunk = examples[start:start+cfg.batch_size]
        batch = trainer.batch_for_examples(chunk, norm, 'cpu')
        output, _, terms = trainer.classification_losses(model, batch, cfg)
        raw = output['transition_logit'].detach().cpu().numpy()
        logits.extend(raw.tolist())
        scores.extend(torch.sigmoid(
            output['transition_logit']).cpu().numpy().tolist())
        labels.extend(x[3] for x in chunk)
        ranking_rows.extend(
            (x[0].isoformat(), x[3], float(s))
            for x, s in zip(chunk, raw)
        )
        for name, v in terms.items():
            totals[name] += v*len(chunk)
    result = {name: val/len(examples) for name,val in totals.items()}
    rank = numpy_event_rank_metrics(
        ranking_rows, margin=cfg.rank_margin
    )
    result.update(rank)
    temporal = loss_temporal(
        torch.tensor(logits, dtype=torch.float32),
        examples,
        temperature=cfg.temperature, scale_s=cfg.temporal_scale_s
    )
    result['temporal'] = float(temporal.detach())
    result['select_loss'] = (
        result['det'] + cfg.region_weight*result['region']
        + cfg.rank_weight*result['rank']
        + cfg.temporal_weight*result['temporal']
    )
    result['total'] = (
        result['total'] + cfg.rank_weight*result['rank']
        + cfg.temporal_weight*result['temporal']
    )
    return result, np.asarray(scores), np.asarray(labels)


def train_fold(fold, fit, monitor, val, blocks, rows, cfg, trainer):
    folder = cfg.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(cfg, fold, fit, monitor, val)
    last_path = folder/f'fold_{fold:02d}_last.pt'
    best_path = folder/f'fold_{fold:02d}_best.pt'
    final_path = folder/f'fold_{fold:02d}_final.pt'
    metrics_path = folder/f'fold_{fold:02d}_metrics.json'
    if cfg.resume and metrics_path.exists() and final_path.exists():
        record = json.loads(metrics_path.read_text(encoding='utf-8'))
        if record['signature'] != signature:
            raise RuntimeError(f'Fold {fold}: resultados incompatibles.')
        print(f'Fold {fold}: ya terminado; se reutiliza.', flush=True)
        return record
    if last_path.exists() and not cfg.resume:
        raise FileExistsError(
            f'{last_path} existe. Usá --resume para continuar.'
        )

    normalizers = {
        name: fit_unique_epochs(
            blocks, [r['time'] for r in fit], instrument=name
        ) for name in ('MAG','SWEA')
    }
    all_mpb = np.asarray([r['time'].timestamp() for r in rows])
    all_bs = np.asarray([r['bs_time'].timestamp() for r in rows])
    fit_pool, fit_skip = trainer.gather_candidates(
        fit, blocks, all_mpb, all_bs,
        ranking=True, rank_control_step_s=cfg.rank_control_step_s
    )
    mon_pool, mon_skip = trainer.gather_candidates(
        monitor, blocks, all_mpb, all_bs,
        ranking=True, rank_control_step_s=cfg.rank_control_step_s
    )
    val_pool, val_skip = trainer.gather_candidates(
        val, blocks, all_mpb, all_bs,
        evaluate=True, stride_s=cfg.evaluation_stride_s
    )
    if not fit_pool or not mon_pool or not val_pool:
        raise RuntimeError(f'Fold {fold}: pools vacíos.')
    torch.manual_seed(cfg.seed+fold)
    model = MagSweaRegionAutoencoder()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3
    )
    epoch_start, best_epoch, stale, best = 0, 0, 0, float('inf')
    history = []
    hard_memory = {}
    if last_path.exists():
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        if last['signature'] != signature:
            raise RuntimeError(f'Fold {fold}: resume incompatible.')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        epoch_start = int(last['epoch'])
        best_epoch = int(last['best_epoch'])
        best = float(last['best_select'])
        stale = int(last['stale'])
        hard_memory = last['hard_memory']
        history = last['history']
        torch.set_rng_state(last['torch_rng'])
        random.setstate(last['python_rng'])
        np.random.set_state(last['numpy_rng'])
        print(f'Fold {fold}: retomado en época {epoch_start}.', flush=True)

    for epoch in range(epoch_start+1, cfg.max_epochs+1):
        if epoch >= cfg.min_epochs and stale >= cfg.patience:
            break
        train_log = train_one_epoch(
            model, optimizer, fit_pool, normalizers, cfg, trainer,
            epoch, fold, hard_memory
        )
        monitor_log, _, _ = monitor_one_epoch(
            model, mon_pool, normalizers, cfg, trainer
        )
        if epoch >= cfg.min_epochs:
            if monitor_log['select_loss'] < best-1e-4:
                best = monitor_log['select_loss']
                best_epoch = epoch
                stale = 0
                trainer.safe_checkpoint(best_path, {
                    'signature': signature, 'epoch': epoch,
                    'model': model.state_dict()
                })
            else:
                stale += 1
        history.append({
            'epoch': epoch, 'best_epoch': best_epoch,
            'stale': stale,
            **{f'train_{k}': v for k,v in train_log.items()},
            **{f'val_{k}': v for k,v in monitor_log.items()}
        })
        trainer.safe_checkpoint(last_path, {
            'signature': signature, 'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_epoch': best_epoch, 'best_select': best,
            'stale': stale, 'hard_memory': hard_memory,
            'history': history,
            'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state()
        })
        print(
            f'Fold {fold} época {epoch}: '
            f'FIT total={train_log["total"]:.4f}, '
            f'FIT temporal={train_log["temporal"]:.4f}, '
            f'MONITOR temporal={monitor_log["temporal"]:.4f}, '
            f'MONITOR selección={monitor_log["select_loss"]:.4f}, '
            f'mejor época={best_epoch}, paciencia={stale}/{cfg.patience}',
            flush=True
        )
        if epoch >= cfg.min_epochs and stale >= cfg.patience:
            print(f'Fold {fold}: early stopping.', flush=True)
            break
    if not best_path.exists():
        raise RuntimeError(f'Fold {fold}: no hubo mejor checkpoint.')

    state = torch.load(best_path, map_location='cpu', weights_only=False)
    if state['signature'] != signature:
        raise RuntimeError(f'Fold {fold}: checkpoint incompatible.')
    model.load_state_dict(state['model'])
    monitor_log, monitor_scores, monitor_labels = monitor_one_epoch(
        model, mon_pool, normalizers, cfg, trainer
    )
    threshold, monitor_f1 = trainer.monitor_threshold(
        monitor_scores, monitor_labels
    )
    metrics, event_rows = trainer.evaluate_fold(
        model, val_pool, normalizers, cfg, 'cpu', threshold, val
    )
    record = {
        'schema': 1,
        'experiment': 'v7b-full-training-temporal-v1',
        'signature': signature, 'fold': fold,
        'best_epoch': best_epoch, 'epochs_completed': len(history),
        'monitor_best_loss': best,
        'monitor_temporal_loss': monitor_log['temporal'],
        'monitor_threshold': threshold,
        'monitor_weak_candidate_f1': monitor_f1,
        'train_events': len(fit), 'inner_monitor_events': len(monitor),
        'train_candidate_events': len(fit_pool),
        'monitor_candidate_events': len(mon_pool),
        'training_configuration': {
            key:getattr(cfg, key) for key in (
                'seed', 'learning_rate', 'batch_size',
                'max_epochs', 'min_epochs', 'patience',
                'region_weight', 'reconstruction_weight',
                'rank_weight', 'rank_margin', 'rank_control_step_s',
                'temporal_weight', 'temperature', 'temporal_scale_s',
                'evaluation_stride_s'
            )
        },
        'train_skip_counts': fit_skip,
        'monitor_skip_counts': mon_skip,
        'val_skip_counts': val_skip,
        'normalizers': {
            k:{key: v[key].tolist() if hasattr(v[key], 'tolist') else v[key]
               for key in ('mean','scale','n_epochs')}
            for k,v in normalizers.items()
        },
        'train_history': history,
        **metrics
    }
    trainer.safe_json(metrics_path, record)
    trainer.write_csv(folder/f'fold_{fold:02d}_event_metrics.csv', event_rows)
    trainer.safe_checkpoint(final_path, {
        'schema': 1, 'fold': fold, 'signature': signature,
        'best_epoch': best_epoch, 'model': model.state_dict(),
        'normalizers': record['normalizers'],
        'monitor_threshold': threshold
    })
    print(
        f'Fold {fold} TERMINADO: época {best_epoch}, '
        f'MPB ≤60s {metrics["fraction_within_60s"]:.1%}, '
        f'error mediano {metrics["median_abs_peak_error_s"]:.1f}s.',
        flush=True
    )
    return record


def main():
    cfg = parse_args()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = load_trainer()

    rows = read_manifest(cfg.manifest)
    for row in rows:
        if not row.get('bs_timestamp_utc'):
            raise ValueError('Falta asociación BS para calcular las etiquetas.')
        row['bs_time'] = bq.to_utc(row['bs_timestamp_utc'])
    folds = [1,2,3,4] if cfg.folds == 'all' else [int(cfg.folds)]
    selections = {}
    all_used = {}
    for fold in folds:
        val = trainer.chosen_events(rows, fold, pilot_count=0)
        train = sorted(
            (r for r in rows if r['fold'] != fold
             and r['mag_available'] == '1'
             and r['swea_available'] == '1'),
            key=lambda r: r['time']
        )
        cv_splits.check_disjoint_supports(train, val)
        fit, monitor, excluded = cv_splits.split_inner_monitor_months(
            train, fold, cfg.seed
        )
        selections[fold] = (fit, monitor, val)
        for row in fit+monitor+val:
            all_used[row['time']] = row
        print(
            f'Fold {fold}: FIT={len(fit)}, MONITOR={len(monitor)}, '
            f'VAL={len(val)}, solapes FIT descartados={excluded}',
            flush=True
        )

    # No se incluye informacion del fold externo en la normalizacion
    # ni en la seleccion de checkpoint. La cache NO ejecuta BigQuery.
    blocks = load_cached_blocks(
        sorted(all_used.values(), key=lambda r:r['time']), cfg, trainer
    )
    completed = {}
    for fold in folds:
        fit, monitor, val = selections[fold]
        completed[fold] = train_fold(
            fold, fit, monitor, val, blocks, rows, cfg, trainer
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'experiment': 'v7b-full-training-temporal-v1',
        'completed_folds': sorted(completed),
        'temporal_weight': cfg.temporal_weight,
        'max_epochs': cfg.max_epochs, 'patience': cfg.patience,
        'warning': (
            'Los 4 folds tienen modelos entrenados desde cero. '
            'La evaluacion se limita a vecindades +/-30 minutos de '
            'MPB catalogadas; no es deteccion autonoma de toda la orbita.'
        ),
        'fold_results': [{
            'fold': k,
            'best_epoch': completed[k]['best_epoch'],
            'val_evaluable_events': completed[k]['val_evaluable_events'],
            'median_abs_peak_error_s':
                completed[k]['median_abs_peak_error_s'],
            'fraction_within_60s':
                completed[k]['fraction_within_60s'],
            'fraction_within_120s':
                completed[k]['fraction_within_120s'],
        } for k in sorted(completed)]
    }
    trainer.safe_json(cfg.output_dir/'cv_summary.json', summary)
    print('CV COMPLETADA: ', cfg.output_dir, flush=True)


if __name__ == '__main__':
    main()
