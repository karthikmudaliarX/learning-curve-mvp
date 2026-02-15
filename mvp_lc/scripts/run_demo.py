# File: mvp_lc/scripts/run_demo.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from mvp_lc.data.make_dataset import make_demo_dataset
from mvp_lc.core.experiment import ExperimentRunner, get_default_scorer
from mvp_lc.core.models import (
    BestCurveModel, MonotoneSplineModel, PowerLawModel,
    ExponentialSaturatingModel, InversePowerModel
)
from mvp_lc.core.policies import DataAcquisitionPolicy, EarlyStoppingPolicy, EarlyDiscardingPolicy
from mvp_lc.core.metrics import evaluate_curve_fit

def main():
    # 1) Data
    X, y = make_demo_dataset(n_samples=1200, n_features=30, random_state=123)
    runner = ExperimentRunner(dataset=(X, y), valid_size=0.3, random_state=42)

    # 2) Algorithms
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    logreg.name = "LogReg"
    rf = RandomForestClassifier(n_estimators=300, random_state=0)
    rf.name = "RF"
    algos = [logreg, rf]

    # 3) Resource schedule (sample sizes as fractions of training data)
    fractions = np.linspace(0.1, 1.0, 10).tolist()
    scorer = get_default_scorer(metric="accuracy")

    # 4) Collect learning curves
    curves = {}
    for algo in algos:
        lc = runner.run_samplewise_curve(algorithm=algo, sample_sizes=fractions, scorer=scorer)
        curves[algo.name] = lc
        print(f"[{algo.name}] collected {len(lc.points)} points, last metric: {lc.points[-1].metric:.4f}")

    # 5) Fit models per curve
    models = {}
    for name, lc in curves.items():
        x, y = lc.resources(), lc.metrics()
        candidate_models = [
            MonotoneSplineModel(increasing=True),
            PowerLawModel(),
            ExponentialSaturatingModel(),
            InversePowerModel()
        ]
        best_model = BestCurveModel(models=candidate_models)
        best_model.fit(x, y)
        models[name] = best_model
        fit_stats = evaluate_curve_fit(best_model, x, y)
        print(f"[{name}] best model fit: RMSE={fit_stats['rmse']:.4f}, R2={fit_stats['r2']:.3f}")

    # 6) Decision: Data Acquisition
    target_acc = 0.83
    da_policy = DataAcquisitionPolicy(target_metric=target_acc)
    for name, lc in curves.items():
        req, ok = da_policy.required_samples(models[name], lc.resources(), max(lc.resources()))
        print(f"[DataAcq {name}] target={target_acc} -> "
              f"{'need ' + str(int(req)) + ' samples' if ok else 'unreachable at observed scales'}")

    # 7) Decision: Early Stopping
    es_policy = EarlyStoppingPolicy(k=3, slope_tol=1e-4)
    for name, lc in curves.items():
        stop, reason = es_policy.should_stop(lc.resources(), lc.metrics())
        print(f"[EarlyStop {name}] stop={stop} reason={reason}")

    # 8) Decision: Early Discarding
    # Find incumbent
    incumbent_name = max(curves.keys(), key=lambda n: curves[n].points[-1].metric)
    incumbent_score = curves[incumbent_name].points[-1].metric
    remaining_budget = int(0.2 * len(runner.y_train_full))  # assume 20% more samples available
    ed_policy = EarlyDiscardingPolicy(incumbent_score=incumbent_score, remaining_budget=remaining_budget, safety_margin=0.01)
    for name in curves.keys():
        discard, info = ed_policy.should_discard(models[name], curves[name].resources())
        decision = "discard" if discard else "keep"
        print(f"[EarlyDiscard {name}] {decision} details={info}")

    # 9) Summary
    print("\n=== Summary ===")
    print(f"Incumbent: {incumbent_name} (val-acc {incumbent_score:.4f})")
    print("Recommended actions:")
    for name in curves.keys():
        lc = curves[name]
        stop, _ = es_policy.should_stop(lc.resources(), lc.metrics())
        discard, _ = ed_policy.should_discard(models[name], lc.resources())
        if discard:
            print(f"- {name}: discard (cannot beat incumbent)")
        elif stop:
            print(f"- {name}: stop early (plateau)")
        else:
            print(f"- {name}: continue training")

if __name__ == "__main__":
    main()
