# Protein Diffusion Research Benchmark Report

Generated at: Sat Aug  9 01:47:07 2025

## Suite Information
- Duration: 2.01 seconds
- Configuration: {
  "dataset_size": 100,
  "sequence_lengths": [
    50,
    100,
    200,
    500
  ],
  "batch_sizes": [
    1,
    8,
    16,
    32
  ],
  "model_variants": [
    "small",
    "medium",
    "large"
  ],
  "temperature_values": [
    0.5,
    1.0,
    1.5
  ],
  "evaluate_quality": true,
  "evaluate_diversity": true,
  "evaluate_novelty": true,
  "evaluate_performance": true,
  "output_dir": "./research/results",
  "save_intermediate": true,
  "statistical_significance": 0.05,
  "num_runs": 3,
  "random_seeds": [
    42,
    123,
    456
  ]
}

## Performance Benchmarks

### Tokenization Performance
- length_50: 0.04ms
- length_100: 0.07ms
- length_200: 0.14ms
- length_500: 0.40ms

### Validation Performance
- Average validation time: 0.61ms

## Quality Analysis

### Sequence Statistics
- Mean length: 24.4
- Length range: 15 - 53
- Sequence diversity: 21.495
- Uniqueness ratio: 0.050

## Comparison Studies

### Tokenization Method Comparison
- selfies: avg compression 1.000

### Validation Strategy Comparison
- strict: 60.0% pass rate
- moderate: 100.0% pass rate
- permissive: 100.0% pass rate
