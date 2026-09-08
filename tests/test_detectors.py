"""
Regression tests for the detector contract and the monitor's data handling.

Every test here pins down a defect that was actually present. The suite runs
without xgboost, tensorflow or PyQt5: those are optional, and the tests that
need them skip.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors import (                                    # noqa: E402
    FEATURE_ORDER,
    EnsembleDetector,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    PCADetector,
    RandomForestDetector,
)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


@pytest.fixture(scope='module')
def normal_df():
    return pd.read_csv(DATA_DIR / 'normal_training.csv')[FEATURE_ORDER]


@pytest.fixture(scope='module')
def labeled():
    df = pd.read_csv(DATA_DIR / 'labeled_training.csv')
    # collect_labeled writes 0 = normal, 1 = anomaly; detectors use 1 / -1.
    y = df['label'].map(lambda v: 1 if v == 0 else -1)
    return df[FEATURE_ORDER], y


@pytest.fixture(scope='module')
def sample(normal_df):
    return normal_df.iloc[0].to_dict()


@pytest.fixture(scope='module')
def stress_sample(labeled):
    X, y = labeled
    return X[y == -1].iloc[0].to_dict()


# ── Package import ──────────────────────────────────────────────────────────

def test_package_imports_without_optional_backends():
    """detectors/__init__ used to import tensorflow and xgboost eagerly, so
    the whole package was unimportable unless both were installed."""
    import importlib
    mod = importlib.import_module('detectors')
    assert mod.IsolationForestDetector is IsolationForestDetector


@pytest.mark.parametrize('attr, backend', [('XGBoostDetector', 'xgboost'),
                                           ('NeuralDetector', 'tensorflow')])
def test_optional_backend_failure_is_deferred_to_use(attr, backend):
    """The cost of a missing backend is one detector, raised where it is
    used — not an unimportable package."""
    import importlib.util
    import detectors

    if importlib.util.find_spec(backend) is None:
        with pytest.raises(ImportError, match=backend):
            getattr(detectors, attr)
    else:
        cls = getattr(detectors, attr)
        assert cls.__name__ == attr


def test_unknown_attribute_still_raises_attribute_error():
    import detectors
    with pytest.raises(AttributeError):
        detectors.NoSuchDetector


# ── Shared score contract ───────────────────────────────────────────────────

UNSUPERVISED = [IsolationForestDetector, OneClassSVMDetector,
                LocalOutlierFactorDetector, PCADetector]


@pytest.mark.parametrize('cls', UNSUPERVISED, ids=lambda c: c.__name__)
def test_score_is_finite_and_agrees_with_pred(cls, normal_df, sample,
                                              stress_sample):
    """The sign of the score must match the verdict, for every detector.

    Note the bound is deliberately not asserted here: Isolation Forest,
    One-Class SVM and LOF report their estimator's raw decision function,
    which base.py documents as normalised 'where possible'.
    """
    det = cls()
    det.train(normal_df)

    for features in (sample, stress_sample):
        pred, score, latency = det.predict(features)
        assert np.isfinite(score), f'{cls.__name__} produced {score}'
        assert pred in (1, -1)
        assert (pred == 1) == (score > 0) or score == 0.0
        assert latency >= 0.0


def test_pca_score_stays_bounded_on_extreme_input(normal_df, sample):
    """PCA is the one detector whose score had no bound at all: it returned
    -(reconstruction_error / threshold), and the error grows without limit as
    a sample moves away from the training distribution. Both consumers that
    average scores — EnsembleDetector and the monitor's aggregate — were then
    decided by PCA alone."""
    det = PCADetector()
    det.train(normal_df)

    absurd = {f: v * 10_000 for f, v in sample.items()}
    pred, score, _ = det.predict(absurd)
    assert pred == -1
    assert -1.0 <= score <= 1.0, f'PCA score escaped its range: {score}'


def test_pca_score_ranking_matches_reconstruction_error(normal_df, labeled):
    """The bound must not cost ranking information: score has to stay a
    strictly decreasing function of reconstruction error, or ROC-AUC moves."""
    det = PCADetector()
    det.train(normal_df)

    X, _ = labeled
    errors, scores = [], []
    for _, row in X.head(120).iterrows():
        features = row.to_dict()
        x = np.array([[features[f] for f in FEATURE_ORDER]], dtype=float)
        errors.append(det._reconstruct_error(det.scaler.transform(x)))
        scores.append(det.predict(features)[1])

    order_by_error = np.argsort(errors)
    ranked_scores = np.asarray(scores)[order_by_error]
    assert np.all(np.diff(ranked_scores) <= 1e-12), \
        'higher reconstruction error must never produce a higher score'


@pytest.mark.parametrize('cls', UNSUPERVISED, ids=lambda c: c.__name__)
def test_predict_before_train_raises_runtime_error(cls, sample):
    """PCADetector skipped health_check() and died with AttributeError on a
    None scaler instead of the contract's RuntimeError."""
    with pytest.raises(RuntimeError):
        cls().predict(sample)


# ── Ensemble ────────────────────────────────────────────────────────────────

def _ensemble(normal_df, labeled):
    X, y = labeled
    ens = EnsembleDetector([
        ('IF', IsolationForestDetector()),
        ('PCA', PCADetector()),
        ('RF', RandomForestDetector()),
    ])
    ens.train(X, y)
    return ens


def test_ensemble_counts_every_member(normal_df, labeled, sample):
    ens = _ensemble(normal_df, labeled)
    pred, score, latency = ens.predict(sample)
    breakdown = ens.vote_breakdown()

    assert set(breakdown) == {'IF', 'PCA', 'RF'}
    assert all(v['pred'] != 0 for v in breakdown.values()), \
        f'a member was dropped from the vote: {breakdown}'
    assert -1.0 <= score <= 1.0
    assert pred in (1, -1)


def test_ensemble_aggregate_is_not_dominated_by_one_member(normal_df, labeled,
                                                           stress_sample):
    """With PCA unbounded, one large reconstruction error could pull the mean
    past -1 no matter how the other members voted."""
    ens = _ensemble(normal_df, labeled)
    _, score, _ = ens.predict(stress_sample)
    member_scores = [v['score'] for v in ens.vote_breakdown().values()]
    assert min(member_scores) - 1e-9 <= score <= max(member_scores) + 1e-9


def test_ensemble_load_from_empty_directory_raises(tmp_path, normal_df, labeled):
    """load() marked the ensemble ready even when it loaded nothing, so
    predict() then returned 0.0 as if it were a measurement."""
    ens = _ensemble(normal_df, labeled)
    fresh = EnsembleDetector([('IF', IsolationForestDetector())])
    with pytest.raises(RuntimeError):
        fresh.load(str(tmp_path / 'nothing'))
    assert ens.model is True


def test_ensemble_save_then_load_round_trips(tmp_path, normal_df, labeled,
                                             sample):
    ens = _ensemble(normal_df, labeled)
    expected = ens.predict(sample)[1]

    ens.save(str(tmp_path))
    restored = EnsembleDetector([
        ('IF', IsolationForestDetector()),
        ('PCA', PCADetector()),
        ('RF', RandomForestDetector()),
    ])
    restored.load(str(tmp_path))
    assert restored.predict(sample)[1] == pytest.approx(expected)


# ── Monitor: missing sensor readings ────────────────────────────────────────

def test_missing_readings_are_imputed_not_passed_to_models():
    """data_collector reports an absent sensor as None. Passing that straight
    through made every detector raise, and after three samples the monitor
    disabled all of them permanently."""
    from monitor import DataCollector

    fallbacks = {f: 1.0 for f in FEATURE_ORDER}
    collector = DataCollector([], {}, fallbacks=fallbacks)

    raw = {f: 5.0 for f in FEATURE_ORDER}
    raw['cpu_temp'] = None
    raw['gpu_temp'] = float('nan')
    raw['per_cpu'] = [{'logical_id': 0, 'usage': 3.0, 'type': 'P'}]

    clean = collector._model_input(raw)

    assert clean['cpu_temp'] == 1.0
    assert clean['gpu_temp'] == 1.0
    assert all(np.isfinite(clean[f]) for f in FEATURE_ORDER)
    # The raw dict must keep the missing readings for the plots and the CSV.
    assert raw['cpu_temp'] is None
    assert clean['per_cpu'] == raw['per_cpu']


def test_every_detector_survives_a_missing_reading(normal_df, labeled, sample):
    """Without imputation, One-Class SVM, LOF and PCA raise on a NaN feature.
    Three of seven detectors failing is enough to put the monitor below
    MIN_HEALTHY_FOR_VOTE, so no verdict is ever produced again."""
    from monitor import DataCollector

    X, y = labeled
    detectors = [
        IsolationForestDetector(), OneClassSVMDetector(),
        LocalOutlierFactorDetector(), PCADetector(),
    ]
    for det in detectors:
        det.train(normal_df)
    rf = RandomForestDetector()
    rf.train(X, y)
    detectors.append(rf)

    fallbacks = {f: float(normal_df[f].median()) for f in FEATURE_ORDER}
    collector = DataCollector([], {}, fallbacks=fallbacks)

    broken = dict(sample)
    broken['cpu_temp'] = None
    broken['gpu_temp'] = None
    features = collector._model_input(broken)

    for det in detectors:
        pred, score, _ = det.predict(features)
        assert pred in (1, -1)
        assert np.isfinite(score)


def test_imputed_sample_still_predicts(normal_df, sample):
    from monitor import DataCollector

    det = IsolationForestDetector()
    det.train(normal_df)

    fallbacks = {f: float(normal_df[f].median()) for f in FEATURE_ORDER}
    collector = DataCollector([], {}, fallbacks=fallbacks)

    broken = dict(sample)
    broken['cpu_temp'] = None
    pred, score, _ = det.predict(collector._model_input(broken))
    assert pred in (1, -1)
    assert np.isfinite(score)


# ── Monitor: GUI formatting helpers ─────────────────────────────────────────

def test_fmt_num_handles_missing_readings():
    """f'{value:.1f}' on a None temperature raised TypeError inside the Qt
    timer callback, which stopped the dashboard refreshing."""
    from monitor import _fmt_num, _plottable

    assert _fmt_num(41.26, '.1f', '°C') == '41.3°C'
    assert _fmt_num(None, '.1f', '°C') == '—'
    assert _fmt_num(None, '.1f', '°C', 'n/a') == 'n/a'
    assert _fmt_num('not a number', '.3f') == '—'

    plotted = _plottable([1.0, None, 3.0])
    assert plotted[0] == 1.0 and np.isnan(plotted[1]) and plotted[2] == 3.0


def test_summarise_marks_thin_verdicts_invalid():
    from monitor import DataCollector, MIN_HEALTHY_FOR_VOTE

    def result(name, pred, score):
        return {'name': name, 'pred': pred, 'score': score,
                'latency': 1.0, 'ok': True, 'disabled': False,
                'error': None, 'ph_stat': None}

    thin = [result(f'd{i}', -1, -0.5) for i in range(MIN_HEALTHY_FOR_VOTE - 1)]
    summary = DataCollector._summarise(thin, jitter_ms=0.0)
    assert summary['valid'] is False
    assert summary['is_anomaly'] is None
    assert summary['agg_score'] is None

    enough = [result(f'd{i}', -1, -0.5) for i in range(MIN_HEALTHY_FOR_VOTE + 3)]
    summary = DataCollector._summarise(enough, jitter_ms=0.0)
    assert summary['valid'] is True
    assert summary['is_anomaly'] is True
    assert summary['agg_score'] == pytest.approx(-0.5)


# ── Drift detector ──────────────────────────────────────────────────────────

def test_drift_detector_ignores_non_finite_values():
    from anomaly_detector import DriftDetector

    d = DriftDetector()
    for _ in range(50):
        d.update(0.1)
    seen = d.n_seen
    d.update(float('nan'))
    d.update(None)
    assert d.n_seen == seen, 'a NaN would poison the cumulative sum forever'


def test_drift_detector_fires_on_a_shifted_stream():
    from anomaly_detector import DriftDetector

    d = DriftDetector(delta=0.005, threshold=5.0, warmup=50)
    for _ in range(100):
        assert not d.update(0.0)
    fired = False
    for _ in range(200):
        fired = d.update(1.0) or fired
    assert fired
    assert d.statistic > 0


# ── CSV logger ──────────────────────────────────────────────────────────────

def test_logger_writes_missing_values_as_empty_fields(tmp_path):
    from logger import CSVLogger

    path = tmp_path / 'log.csv'
    with CSVLogger(str(path), detector_names=['Isolation Forest']) as log:
        log.log(
            timestamp=1000.0,
            metrics={f: 1.0 for f in FEATURE_ORDER} | {'cpu_temp': None},
            anomaly=-1, score=-0.25, latency=2.0, jitter=0.5,
            n_healthy=1, verdict_valid=True,
            per_detector={'Isolation Forest': {
                'score': 0.0, 'latency': 2.0, 'ok': True, 'ph_stat': 0.0}},
        )

    rows = list(pd.read_csv(path).to_dict('records'))
    assert len(rows) == 1
    row = rows[0]
    assert pd.isna(row['cpu_temp'])          # missing, not zero
    assert row['isolation_forest_score'] == 0.0   # a real zero survives
    assert row['score'] == pytest.approx(-0.25)
    assert (path.with_suffix('.summary.txt')).exists()


def test_logger_column_order_tracks_feature_order():
    """logger.py kept its own copy of FEATURE_ORDER; the two could drift and
    silently mislabel every feature column."""
    import logger
    from detectors.base import FEATURE_ORDER as canonical
    assert logger.FEATURE_ORDER is canonical


# ── Optional backends ───────────────────────────────────────────────────────

def test_neural_detector_satisfies_the_base_contract():
    """NeuralDetector had no health_check/save/load and did not subclass
    BaseDetector, so EnsembleDetector's health_check() call raised
    AttributeError and dropped it from every vote."""
    tf = pytest.importorskip('tensorflow')
    from detectors.base import BaseDetector
    from detectors.neural_detector import NeuralDetector

    assert issubclass(NeuralDetector, BaseDetector)
    det = NeuralDetector()
    det.health_check()
    for attr in ('train', 'predict', 'save', 'load', 'health_check'):
        assert callable(getattr(det, attr))


def test_neural_detector_scores_stress_below_normal(sample, stress_sample):
    pytest.importorskip('tensorflow')
    from detectors.neural_detector import NeuralDetector

    det = NeuralDetector()
    _, normal_score, _ = det.predict(sample)
    _, stress_score, _ = det.predict(stress_sample)
    assert -1.0 <= stress_score <= 1.0
    assert stress_score < normal_score
