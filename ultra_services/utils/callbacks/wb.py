# Ultralytics YOLO 🚀, AGPL-3.0 license

"""Weights & Biases integration with failure isolation.

Optional experiment loggers must never terminate model training.  The helper
also converts absolute filesystem output paths into valid W&B project names.
"""

from __future__ import annotations

import re
from pathlib import Path

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING
    assert SETTINGS['wandb'] is True
    import wandb as wb

    assert hasattr(wb, '__version__')
    import numpy as np
    import pandas as pd

    _processed_plots = {}
except (ImportError, AssertionError):
    wb = None
    _processed_plots = {}

_WB_DISABLED_FOR_RUN = False


def _safe_project_name(value) -> str:
    """Return a W&B-safe project name from a path or arbitrary user value."""
    raw = str(value or 'YOLOv8').strip()
    # Absolute/relative output directories should become their final directory name.
    name = Path(raw).name if any(char in raw for char in ('/', '\\')) else raw
    name = re.sub(r'[/\\,#?%:]+', '-', name).strip(' .-')
    return name or 'YOLOv8'


def _disable_after_error(exc: Exception) -> None:
    global _WB_DISABLED_FOR_RUN
    _WB_DISABLED_FOR_RUN = True
    LOGGER.warning(f'WARNING ⚠️ Weights & Biases disabled for this run after an integration error: {exc}')


def _active_run():
    return None if wb is None or _WB_DISABLED_FOR_RUN else getattr(wb, 'run', None)


def _custom_table(x, y, classes, title='Precision Recall Curve', x_title='Recall', y_title='Precision'):
    df = pd.DataFrame({'class': classes, 'y': y, 'x': x}).round(3)
    fields = {'x': 'x', 'y': 'y', 'class': 'class'}
    string_fields = {'title': title, 'x-axis-title': x_title, 'y-axis-title': y_title}
    return wb.plot_table('wandb/area-under-curve/v0', wb.Table(dataframe=df), fields=fields,
                         string_fields=string_fields)


def _plot_curve(x, y, names=None, id='precision-recall', title='Precision Recall Curve',
                x_title='Recall', y_title='Precision', num_x=100, only_mean=False):
    if not _active_run():
        return
    names = names or []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()
    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ['mean'] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)
            y_log.extend(np.interp(x_new, x, yi))
            classes.extend([names[i]] * len(x_new))
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    if not _active_run():
        return
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    if wb is None or _WB_DISABLED_FOR_RUN:
        return
    try:
        if not getattr(wb, 'run', None):
            wb.init(project=_safe_project_name(trainer.args.project), name=trainer.args.name,
                    config=vars(trainer.args))
    except Exception as exc:  # optional logger failures must not stop training
        _disable_after_error(exc)


def on_fit_epoch_end(trainer):
    if not _active_run():
        return
    try:
        wb.run.log(trainer.metrics, step=trainer.epoch + 1)
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        if trainer.epoch == 0:
            wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    except Exception as exc:
        _disable_after_error(exc)


def on_train_epoch_end(trainer):
    if not _active_run():
        return
    try:
        wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
        wb.run.log(trainer.lr, step=trainer.epoch + 1)
        if trainer.epoch == 1:
            _log_plots(trainer.plots, step=trainer.epoch + 1)
    except Exception as exc:
        _disable_after_error(exc)


def on_train_end(trainer):
    if not _active_run():
        return
    try:
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
        if trainer.best.exists():
            art.add_file(trainer.best)
            wb.run.log_artifact(art, aliases=['best'])
        for curve_name, curve_values in zip(trainer.validator.metrics.curves,
                                            trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(x, y, names=list(trainer.validator.metrics.names.values()), id=f'curves/{curve_name}',
                        title=curve_name, x_title=x_title, y_title=y_title)
        wb.run.finish()
    except Exception as exc:
        _disable_after_error(exc)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end,
} if wb else {}
