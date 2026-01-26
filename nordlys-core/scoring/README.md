# Scoring Module

Model scoring with cost-accuracy optimization using configurable lambda parameters.

## Overview

The scoring module implements the cost-accuracy trade-off algorithm that ranks models based on:
- Error rate (accuracy)
- Cost per token
- User-provided cost bias (lambda)

## Headers

```cpp
#include <nordlys/scoring/scorer.hpp>
```

## CMake

```cmake
target_link_libraries(your_target PRIVATE Nordlys::Scoring)
```

## Usage

```cpp
#include <nordlys/scoring/scorer.hpp>

// Create scorer with default parameters
ModelScorer scorer;

// Or with custom lambda parameters
ModelScorer scorer(0.3f, 0.7f);  // lambda_error, lambda_cost

// Score models for a cluster
std::vector<ModelFeatures> models = /* ... */;
int cluster_id = 0;

auto scores = scorer.score_models(cluster_id, models);

// Scores are sorted by combined score (lower is better)
for (const auto& score : scores) {
    std::cout << score.model_name << ": " << score.score << "\n";
}
```

## Algorithm

The scoring formula combines error rate and cost:

```text
score = lambda_error * normalized_error + lambda_cost * normalized_cost
```

Where:
- `normalized_error` = error_rate / max_error_rate
- `normalized_cost` = cost_per_token / max_cost
- `lambda_error + lambda_cost = 1.0`

## Dependencies

- `Nordlys::Common` - Matrix and result types
