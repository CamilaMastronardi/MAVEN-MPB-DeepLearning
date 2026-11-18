#!/usr/bin/env python3
"""V7b: piloto independiente para penalizar el error temporal de la MPB.

Compara el checkpoint original con dos ajustes finos desde el MISMO
checkpoint: control (loss V7b) y temporal (loss V7b + w*E_softmax|dt|/600).
Los ejemplos FIT y de validacion externa son identicos para ambas variantes.
SOLO utiliza checkpoints y caches locales. Nunca consulta BigQuery ni
sobrescribe modelos originales. Es una prueba exploratoria, no CV completa.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.model.ranking import (
    event_pairwise_ranking_loss, event_group_batches, select_rank_event_groups,
)
from src.model.v7b import MagSweaRegionAutoencoder
from src.preprocessing import bigquery_source as bq
from src.preprocessing.catalog import read_manifest
from src.preprocessing.cv_splits import split_inner_monitor_months
from src.preprocessing.mag_cache import MagEventCache
from src.preprocessing.swea_cache import SweaEventCache


def temporal_distance_loss(logits, examples, *, temperature=1., scale_s=600.):
    """Esperanza diferenciable del error temporal absoluto por evento.

    Cada evento aporta el mismo peso. E|dt| evita que errores positivos y
    negativos se cancelen en el softargmax, aunque la inferencia final usa
    argmax duro. Los tiempos de la MPB solo se usan en FIT, no como input.
    """
    if logits.ndim != 1 or len(examples) != logits.numel():
        raise ValueError('Logits y ejemplos desalineados.')
    if temperature <= 0 or scale_s <= 0:
        raise ValueError('Temperatura y escala deben ser positivas.')
    groups = defaultdict(list)
    for i, (event, candidate, *_) in enumerate(examples):
        groups[event].append((i, candidate.center_s - event.timestamp()))
    if not groups:
        raise ValueError('La loss temporal requiere eventos.')
    losses = []
    for group in groups.values():
        idx = torch.tensor([p for p, _ in group], device=logits.device)
        dt = torch.tensor([v for _, v in group],
                          dtype=logits.dtype, device=logits.device)
        probs = torch.softmax(logits.index_select(0, idx)/temperature, 0)
        losses.append((probs * dt.abs()).sum()/scale_s)
    return torch.stack(losses).mean()


def trainer_module():
    path = ROOT/'scripts/02_training/01_train_v7b.py'
    spec = importlib.util.spec_from_file_location('pilot_v7b_trainer', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metrics-dir', 'manifest', 'mag-cache-dir',
                 'swea-cache-dir', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fold', type=int, choices=(1,2,3,4), default=1)
    p.add_argument('--train-events', type=int, default=0,
                   help="0 = todos los eventos FIT disponibles en caché.")
    p.add_argument('--validation-events', type=int, default=0,
                   help="0 = todos los eventos externos disponibles en caché.")
    p.add_argument('--scan-limit', type=int, default=0,
                   help="0 = sin límite de búsqueda de cachés.")
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--weight', type=float, default=5.0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--scale-s', type=float, default=600.)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if (min(a.train_events, a.validation_events, a.scan_limit) < 0
            or min(a.epochs, a.weight, a.temperature,
                   a.scale_s, a.learning_rate, a.threads) <= 0):
        p.error('Los limites deben ser >= 0 y los hiperparametros > 0.')
    return a


def cached_pair(row, cache, trainer):
    begin, end = trainer.event_bounds(row)
    mag = cache['MAG'].load(row['time'], begin, end, max_rows=25000)
    swea = cache['SWEA'].load(row['time'], begin, end, max_rows=2000)
    if mag is None or swea is None or not len(mag[0]) or not len(swea[0]):
        return None
    return {'MAG': mag, 'SWEA': swea}


def select_cached(rows, n, *, cache, trainer, mpb, bs, rank,
                  stride, rank_stride, seed, scan_limit):
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    selected, candidates = [], {}
    scanned = cached = 0
    for row in (pool if scan_limit == 0 else pool[:scan_limit]):
        if n > 0 and len(selected) >= n:
            break
        scanned += 1
        pair = cached_pair(row, cache, trainer)
        if pair is None:
            continue
        cached += 1
        possible, _ = trainer.gather_candidates(
            [row], {row['time']: pair}, mpb, bs, evaluate=not rank,
            ranking=rank, stride_s=stride, rank_control_step_s=rank_stride
        )
        items = possible.get(row['time'], [])
        if rank and not (
            any(c.transition == 1 for c, _ in items) and
            any(c.transition == 0 for c, _ in items)
        ):
            continue
        if not items:
            continue
        selected.append(row)
        candidates[row['time']] = items
    print(f'{"FIT" if rank else "VAL"}: {len(selected)}/'
          f'{"todos" if n == 0 else n} usable, '
          f'{cached}/{scanned} with both caches.', flush=True)
    return selected, candidates


def summarize(records):
    errors = np.asarray(
        [float(r['absolute_error_s']) for r in records
         if r['absolute_error_s'] != ''], dtype=float
    )
    return {
        'n_events': int(len(errors)),
        'median_abs_error_s': float(np.median(errors)) if len(errors) else None,
        'within_60s': float(np.mean(errors <= 60)) if len(errors) else None,
        'within_120s': float(np.mean(errors <= 120)) if len(errors) else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a = arguments()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    trainer = trainer_module()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fold = a.fold
    metrics = json.loads(
        (a.metrics_dir/f'fold_{fold:02d}_metrics.json').read_text()
    )
    checkpoint = torch.load(
        a.metrics_dir/f'fold_{fold:02d}_final.pt',
        map_location='cpu', weights_only=False
    )
    if checkpoint['signature'] != metrics['signature']:
        raise ValueError('Firma de checkpoint y metricas incompatibles.')
    conf = metrics['training_configuration']
    if conf.get('rank_weight', 0.) <= 0:
        raise ValueError('Se requiere un checkpoint V7b con ranking.')
    if hashlib.sha256(a.manifest.read_bytes()).hexdigest() != conf['manifest_sha256']:
        raise ValueError('Manifest distinto del entrenamiento de partida.')
    if a.seed != conf['seed']:
        raise ValueError('Debe coincidir la semilla del checkpoint.')
    norm = {
        k: {'mean': np.asarray(v['mean']), 'scale': np.asarray(v['scale'])}
        for k, v in checkpoint['normalizers'].items()
    }
    rows = read_manifest(a.manifest)
    for row in rows:
        row['bs_time'] = bq.to_utc(row['bs_timestamp_utc'])
    all_mpb = np.array([r['time'].timestamp() for r in rows])
    all_bs = np.array([r['bs_time'].timestamp() for r in rows])
    train = [
        r for r in rows if r['fold'] != fold and
        r['mag_available'] == '1' and r['swea_available'] == '1'
    ]
    val = [
        r for r in rows if r['fold'] == fold and
        r['mag_available'] == '1' and r['swea_available'] == '1'
    ]
    fit, monitor, _ = split_inner_monitor_months(train, fold, a.seed)
    print(f'Fold {fold} | FIT={len(fit)}, MONITOR excluido={len(monitor)}, '
          f'VAL={len(val)}', flush=True)
    sql = trainer.query_sources()
    cache = {
        'MAG': MagEventCache(
            a.mag_cache_dir, project=bq.PROJECT, table=bq.MAG, sql=sql['MAG']
        ),
        'SWEA': SweaEventCache(
            a.swea_cache_dir, project=bq.PROJECT,
            table=bq.SWEA+' + '+bq.AXES, sql=sql['SWEA']
        ),
    }
    sel_fit, fit_pool = select_cached(
        fit, a.train_events, cache=cache, trainer=trainer,
        mpb=all_mpb, bs=all_bs, rank=True,
        stride=conf['evaluation_stride_s'],
        rank_stride=conf['rank_control_step_s'],
        seed=a.seed+1000*fold, scan_limit=a.scan_limit
    )
    sel_val, val_pool = select_cached(
        val, a.validation_events, cache=cache, trainer=trainer,
        mpb=all_mpb, bs=all_bs, rank=False,
        stride=conf['evaluation_stride_s'],
        rank_stride=conf['rank_control_step_s'],
        seed=a.seed+2000*fold, scan_limit=a.scan_limit
    )
    if len(sel_fit) < 4 or len(sel_val) < 3:
        raise RuntimeError(
            'Faltan caches MAG/SWEA validas: se requieren al menos 4 FIT '
            'y 3 VAL. No se consulto BigQuery.'
        )
    if a.train_events == 0 or a.validation_events == 0:
        print('MODO COMPLETO: incluye todos los eventos elegibles encontrados '
              'en las caches, sin consultar BigQuery. Si faltan caches, '
              'la evaluacion sera parcial.', flush=True)

    class Config:
        pass
    cfg = Config()
    for key in (
        'rank_weight', 'rank_margin', 'reconstruction_weight',
        'region_weight', 'rank_positives_per_event',
        'rank_hard_negatives', 'rank_random_negatives', 'batch_size'
    ):
        setattr(cfg, key, conf[key])

    # Se preparan una vez los mismos grupos, sin memoria dinamica:
    # controla la unica diferencia experimental (la nueva loss).
    epoch_batches = []
    for epoch in range(1, a.epochs+1):
        groups = select_rank_event_groups(
            fit_pool, epoch=epoch, seed=a.seed+fold*10000,
            hard_memory={}, n_positives=cfg.rank_positives_per_event,
            n_hard=cfg.rank_hard_negatives,
            n_random=cfg.rank_random_negatives
        )
        if not groups:
            raise RuntimeError('No se pudieron formar grupos de ranking FIT.')
        epoch_batches.append(list(event_group_batches(groups, cfg.batch_size)))

    original_state = copy.deepcopy(checkpoint['model'])
    histories, models = {}, {}
    for name, weight in (('control', 0.), ('temporal', a.weight)):
        torch.manual_seed(a.seed+fold+555)
        model = MagSweaRegionAutoencoder()
        model.load_state_dict(original_state)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=a.learning_rate, weight_decay=1e-3
        )
        history = []
        for epoch, batches in enumerate(epoch_batches, 1):
            model.train()
            acc = defaultdict(float)
            for examples in batches:
                batch = trainer.batch_for_examples(examples, norm, 'cpu')
                optimizer.zero_grad(set_to_none=True)
                output, classification, _ = trainer.classification_losses(
                    model, batch, cfg
                )
                logits = output['transition_logit']
                ranking, _, _, _ = event_pairwise_ranking_loss(
                    logits, batch['transition'],
                    [item[0].isoformat() for item in examples],
                    margin=cfg.rank_margin
                )
                temporal = temporal_distance_loss(
                    logits, examples, temperature=a.temperature, scale_s=a.scale_s
                )
                loss = classification + cfg.rank_weight*ranking + weight*temporal
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('La loss no es finita.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                acc['total'] += float(loss.detach())
                acc['time'] += float(temporal.detach())
            rec = {'variant': name, 'epoch': epoch,
                   'loss': acc['total']/len(batches),
                   'normalized_expected_error': acc['time']/len(batches)}
            history.append(rec)
            print(f'{name}, epoca {epoch}: loss={rec["loss"]:.4f}, '
                  f'L_time={rec["normalized_expected_error"]:.4f}', flush=True)
        histories[name] = history
        models[name] = model

    threshold = float(checkpoint['monitor_threshold'])
    evaluations = {}
    for name in ('original', 'control', 'temporal'):
        model = models.get(name)
        if model is None:
            model = MagSweaRegionAutoencoder()
            model.load_state_dict(original_state)
        model.eval()
        with torch.inference_mode():
            _, records = trainer.evaluate_fold(
                model, val_pool, norm, cfg, 'cpu', threshold, sel_val
            )
        evaluations[name] = records
        print(name, summarize(records), flush=True)

    keyed = {
        name: {r['mpb_timestamp_utc']: r for r in records}
        for name, records in evaluations.items()
    }
    paired = []
    for row in sel_val:
        t = row['time'].isoformat()
        rec = {'mpb_timestamp_utc': t}
        for name in ('original', 'control', 'temporal'):
            v = keyed[name][t]
            rec[name+'_error_s'] = v['absolute_error_s']
            rec[name+'_offset_s'] = v['peak_offset_s']
        paired.append(rec)
    write_csv(a.output_dir/'paired_event_errors.csv', paired)
    write_csv(
        a.output_dir/'training_history.csv',
        histories['control']+histories['temporal']
    )
    summary = {
        'pilot': bool(a.train_events or a.validation_events or a.scan_limit),
        'fold': fold, 'n_fit': len(sel_fit),
        'n_val': len(sel_val),
        'n_fit_catalog_eligible': len(fit),
        'n_val_catalog_eligible': len(val),
        'train_limit': a.train_events,
        'validation_limit': a.validation_events,
        'scan_limit': a.scan_limit,
        'epochs': a.epochs,
        'learning_rate': a.learning_rate, 'temporal_weight': a.weight,
        'temperature': a.temperature, 'scale_s': a.scale_s,
        'new_loss': 'E_softmax[abs(candidate_center - catalog_MPB)] / scale_s',
        'evaluation': 'unchanged hard argmax on outer fold',
        'sampling': 'fixed event batches, no dynamic hard-negative mining',
        'model_signature': checkpoint['signature'],
        'results': {k: summarize(v) for k, v in evaluations.items()}
    }
    (a.output_dir/'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )

    cmap = plt.get_cmap('inferno')
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for name, color in [('control', cmap(.25)), ('temporal', cmap(.80))]:
        hist = histories[name]
        ax.plot([r['epoch'] for r in hist], [r['loss'] for r in hist],
                '-o', color=color, label=name)
    ax.set(xlabel='Epoca de ajuste fino',
           ylabel='Loss total (objetivos diferentes)',
           title='Piloto: control vs penalizacion temporal')
    ax.legend()
    fig.savefig(a.output_dir/'training_losses.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for name, color in [
        ('original', cmap(.12)), ('control', cmap(.45)),
        ('temporal', cmap(.85))
    ]:
        errors = sorted(
            float(r['absolute_error_s']) for r in evaluations[name]
            if r['absolute_error_s'] != ''
        )
        axes[0].plot(errors, np.arange(1, len(errors)+1)/len(errors),
                     '-o', ms=3, lw=1.5, color=color, label=name)
    axes[0].axvline(60, color=cmap(.72), ls='--')
    axes[0].set(
        xlabel='Error temporal absoluto (s)',
        ylabel='Fraccion acumulada',
        title='Error en validacion externa'
    )
    axes[0].legend()
    x = np.array([float(r['control_error_s']) for r in paired])
    y = np.array([float(r['temporal_error_s']) for r in paired])
    limit = max(100., x.max(), y.max())*1.04
    axes[1].scatter(x, y, color=cmap(.40), s=40)
    axes[1].plot([0, limit], [0, limit], '--', color=cmap(.75))
    axes[1].set(
        xlim=(0, limit), ylim=(0, limit),
        xlabel='Error control (s)',
        ylabel='Error con penalizacion temporal (s)',
        title='Comparacion evento por evento'
    )
    fig.suptitle('Variacion del error de localizacion por la loss temporal')
    fig.savefig(a.output_dir/'temporal_localization_comparison.png', dpi=180)
    plt.close(fig)
    print('Resultados guardados:', a.output_dir.resolve(), flush=True)


if __name__ == '__main__':
    main()
