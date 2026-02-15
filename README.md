# Learning Curve Analysis Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An MVP implementation of learning curve analysis for decision making in supervised machine learning, based on the survey paper by [Mohr & van Rijn (2022)](https://arxiv.org/abs/2201.12150).

## Overview

This toolkit provides practical implementations of learning curve modeling and decision-making policies to optimize machine learning workflows. It helps answer critical questions like:

- **Data Acquisition**: How much more data do I need to reach my target performance?
- **Early Stopping**: Should I continue training, or have I hit a plateau?
- **Early Discarding**: Should I abandon this configuration in favor of better alternatives?

## Features

### Curve Fitting Models
- **Monotone Spline**: Non-parametric interpolation with monotonicity constraints
- **Power Law**: `y ≈ a - b * x^(-c)` - classic learning curve model
- **Exponential Saturating**: `y ≈ L - b * exp(-c*x)` - smooth saturation model
- **Inverse Power**: `y ≈ L - b / x^c` - alternative saturation model
- **Best-of Ensemble**: Automatically selects the best-fitting model

### Decision Policies
- **Data Acquisition Policy**: Estimates required samples to reach target performance
- **Early Stopping Policy**: Detects performance plateaus using slope analysis
- **Early Discarding Policy**: Compares against incumbent to decide if a model is worth pursuing

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/learning-curve-mvp.git
cd learning-curve-mvp
pip install -r requirements.txt
```

### Run the Demo

```bash
python -m mvp_lc.scripts.run_demo
```

This demonstrates a complete workflow:
1. Generates a synthetic dataset
2. Trains Logistic Regression and Random Forest models
3. Collects learning curves at different sample sizes
4. Fits parametric curve models to the data
5. Applies decision policies (data acquisition, early stopping, early discarding)
6. Summarizes recommendations

### Example Output

```
[LogReg] collected 10 points, last metric: 0.6472
[RF] collected 10 points, last metric: 0.7611

[DataAcq LogReg] target=0.83 -> unreachable at observed scales
[DataAcq RF] target=0.83 -> unreachable at observed scales

[EarlyStop LogReg] stop=True (slope tolerance reached)
[EarlyStop RF] stop=True (slope tolerance reached)

[EarlyDiscard LogReg] discard (cannot beat incumbent)
[EarlyDiscard RF] keep (incumbent)

=== Summary ===
Incumbent: RF (val-acc 0.7611)
Recommended actions:
- LogReg: discard (cannot beat incumbent)
- RF: stop early (plateau detected)
```

## Usage

### Basic Example

```python
from mvp_lc.data.make_dataset import make_demo_dataset
from mvp_lc.core.experiment import ExperimentRunner, get_default_scorer
from mvp_lc.core.models import BestCurveModel, PowerLawModel, ExponentialSaturatingModel
from mvp_lc.core.policies import DataAcquisitionPolicy, EarlyStoppingPolicy

# 1. Prepare data
X, y = make_demo_dataset(n_samples=1000, n_features=20)
runner = ExperimentRunner(dataset=(X, y), valid_size=0.3)

# 2. Train and collect learning curve
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
scorer = get_default_scorer("accuracy")
lc = runner.run_samplewise_curve(model, fractions, scorer)

# 3. Fit learning curve model
x, y_curve = lc.resources(), lc.metrics()
candidates = [PowerLawModel(), ExponentialSaturatingModel()]
best_model = BestCurveModel(models=candidates)
best_model.fit(x, y_curve)

# 4. Make decisions
target = 0.85
dacq = DataAcquisitionPolicy(target_metric=target)
required, reachable = dacq.required_samples(best_model, x, max(x))

estop = EarlyStoppingPolicy(k=3, slope_tol=1e-4)
should_stop, info = estop.should_stop(x, y_curve)
```

## API Reference

### Core Components

#### `ExperimentRunner`
Manages incremental training on increasing sample sizes.

```python
runner = ExperimentRunner(dataset=(X, y), valid_size=0.3)
lc = runner.run_samplewise_curve(
    algorithm=model,
    sample_sizes=[0.1, 0.3, 0.5, 1.0],  # fractions of training data
    scorer=accuracy_score
)
```

#### Curve Models

All models implement:
- `fit(x, y)`: Fit to learning curve data
- `predict(x)`: Predict performance at given sample sizes
- `asymptote()`: Estimate final performance limit
- `resource_for_target(target, x_min, x_max)`: Estimate samples needed for target

#### Policies

**DataAcquisitionPolicy**: `required_samples(model, x_observed, x_max)`

**EarlyStoppingPolicy**: `should_stop(x, y)` returns (stop_boolean, info_dict)

**EarlyDiscardingPolicy**: `should_discard(model, x_observed)` compares against incumbent

## Testing

```bash
python -m pytest tests/ -v
```

## Background

This implementation is based on:

> **Learning Curves for Decision Making in Supervised Machine Learning: A Survey**  
> Felix Mohr, Jan N. van Rijn  
> arXiv:2201.12150, 2022

The survey provides a comprehensive overview of learning curve models and their application in automated machine learning (AutoML) for decisions like:
- Determining optimal training set sizes
- Early stopping of underperforming configurations
- Selecting between competing algorithms

## Project Structure

```
mvp_lc/
├── core/
│   ├── experiment.py    # Learning curve generation
│   ├── models.py        # Parametric curve models
│   ├── policies.py      # Decision policies
│   └── metrics.py       # Evaluation metrics
├── data/
│   └── make_dataset.py  # Dataset utilities
└── scripts/
    └── run_demo.py      # Demonstration script

tests/
└── test_models.py       # Unit tests
```

## Dependencies

- numpy >= 1.21
- scikit-learn >= 1.0
- scipy >= 1.8 (optional, for monotone cubic splines)
- pytest >= 7.0 (for testing)

## Contributing

This is an MVP implementation. Contributions welcome! Areas for improvement:
- Additional curve models (e.g., Weibull, logarithmic)
- Bayesian approaches to uncertainty quantification
- Visualization utilities
- Integration with popular ML frameworks

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{mohr2022learning,
  title={Learning Curves for Decision Making in Supervised Machine Learning: A Survey},
  author={Mohr, Felix and van Rijn, Jan N},
  journal={arXiv preprint arXiv:2201.12150},
  year={2022}
}
```
