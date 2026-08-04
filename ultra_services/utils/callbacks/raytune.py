# Ultralytics YOLO 🚀, AGPL-3.0 license

"""Ray Tune integration that is a no-op outside an active Ray session."""

from __future__ import annotations

from ultralytics.utils import LOGGER, SETTINGS

try:
    assert SETTINGS['raytune'] is True
    import ray
    from ray import tune
    try:
        from ray.air import session as air_session
    except (ImportError, AttributeError):
        air_session = None
except (ImportError, AssertionError):
    ray = None
    tune = None
    air_session = None


def _ray_session_enabled() -> bool:
    """Detect an active Tune/Train session without depending on one Ray version."""
    if ray is None or tune is None:
        return False
    try:
        legacy = getattr(tune, 'is_session_enabled', None)
        if callable(legacy):
            return bool(legacy())
    except Exception:
        return False
    try:
        if air_session is not None:
            getter = getattr(air_session, 'get_session', None)
            if callable(getter):
                return getter() is not None
    except Exception:
        return False
    try:
        from ray.train import get_context
        context = get_context()
        trial_id = getattr(context, 'get_trial_id', lambda: None)()
        return bool(trial_id)
    except Exception:
        return False


def _report(metrics: dict) -> None:
    """Report through whichever Ray API is available."""
    if air_session is not None and callable(getattr(air_session, 'report', None)):
        air_session.report(metrics)
        return
    try:
        from ray import train
        train.report(metrics)
    except Exception as exc:
        LOGGER.warning(f'WARNING ⚠️ Ray Tune metrics were not reported: {exc}')


def on_fit_epoch_end(trainer):
    """Send metrics only when executing inside a real Ray Tune session."""
    if not _ray_session_enabled():
        return
    try:
        metrics = dict(trainer.metrics)
        metrics['epoch'] = trainer.epoch
        _report(metrics)
    except Exception as exc:
        # Optional tuning integrations must not invalidate a completed epoch.
        LOGGER.warning(f'WARNING ⚠️ Ray Tune callback disabled for this epoch: {exc}')


callbacks = {'on_fit_epoch_end': on_fit_epoch_end} if tune else {}
