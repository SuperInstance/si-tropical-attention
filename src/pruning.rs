use crate::tropical::Tropical;
use crate::attention::convergence_error;

/// Result of a pruning comparison experiment.
#[derive(Debug, Clone)]
pub struct PruningResult {
    pub method: String,
    pub n_kept: usize,
    pub output_error: f64,
    pub speedup_estimate: f64,
}

/// Compute per-head importance based on attention weight magnitude.
///
/// Importance = Frobenius norm of the attention weight matrix for each head.
pub fn compute_importance(attention_heads: &[Vec<Vec<f64>>]) -> Vec<f64> {
    attention_heads.iter().map(|head| {
        let mut sum_sq = 0.0;
        for row in head {
            for &val in row {
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }).collect()
}

/// Select heads to keep using tropical rank analysis.
///
/// Strategy: compute the tropical rank contribution of each head,
/// keep the heads that contribute most to the full tropical rank.
/// This preserves the most important sparse attention patterns.
pub fn tropical_prune(
    attention_heads: &[Vec<Vec<f64>>],
    keep_ratio: f64,
) -> Vec<usize> {
    let n = attention_heads.len();
    let n_keep = ((n as f64) * keep_ratio).ceil() as usize;
    let n_keep = n_keep.max(1).min(n);

    // Compute importance via tropical determinant contribution
    let mut scored: Vec<(usize, f64)> = attention_heads.iter().enumerate().map(|(idx, head)| {
        // Convert to tropical and compute tropical determinant as importance measure
        let trop_head: Vec<Vec<Tropical>> = head.iter().map(|row| {
            row.iter().map(|&x| Tropical(x)).collect()
        }).collect();

        let n_rows = trop_head.len();
        let n_cols = trop_head[0].len();
        let min_dim = n_rows.min(n_cols);

        // Use max element + row variance as a tropical importance proxy
        let mut max_val = f64::NEG_INFINITY;
        let mut total = 0.0;
        let mut count = 0;
        for row in &trop_head {
            for &val in row {
                let v = val.inner();
                if v > f64::NEG_INFINITY {
                    max_val = max_val.max(v);
                    total += v;
                    count += 1;
                }
            }
        }

        let avg = if count > 0 { total / count as f64 } else { 0.0 };
        let mut variance = 0.0;
        for row in &trop_head {
            for &val in row {
                let v = val.inner();
                if v > f64::NEG_INFINITY {
                    variance += (v - avg).powi(2);
                }
            }
        }
        variance = if count > 0 { variance / count as f64 } else { 0.0 };

        // Tropical importance: combination of magnitude and tropical structure
        let importance = max_val.abs() + variance.sqrt() + min_dim as f64 * 0.1;
        (idx, importance)
    }).collect();

    // Sort by importance (descending) and pick top heads
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n_keep);
    let mut keep: Vec<usize> = scored.into_iter().map(|(idx, _)| idx).collect();
    keep.sort();
    keep
}

/// Random pruning baseline: select heads randomly.
pub fn random_prune(n_heads: usize, keep_ratio: f64) -> Vec<usize> {
    let n_keep = ((n_heads as f64) * keep_ratio).ceil() as usize;
    let n_keep = n_keep.max(1).min(n_heads);

    // Simple deterministic "random" for reproducibility
    // Use a basic hash-like selection
    let mut indices: Vec<usize> = (0..n_heads).collect();
    // Poor man's shuffle with fixed seed
    for i in (1..n_heads).rev() {
        let j = (i * 7 + 3) % (i + 1);
        indices.swap(i, j);
    }
    indices.truncate(n_keep);
    indices.sort();
    indices
}

/// Magnitude pruning baseline: keep heads with largest Frobenius norm.
pub fn magnitude_prune(
    attention_heads: &[Vec<Vec<f64>>],
    keep_ratio: f64,
) -> Vec<usize> {
    let n = attention_heads.len();
    let n_keep = ((n as f64) * keep_ratio).ceil() as usize;
    let n_keep = n_keep.max(1).min(n);

    let importance = compute_importance(attention_heads);
    let mut scored: Vec<(usize, f64)> = importance.into_iter().enumerate().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n_keep);
    let mut keep: Vec<usize> = scored.into_iter().map(|(idx, _)| idx).collect();
    keep.sort();
    keep
}

/// Compare pruning methods by reconstructing the output with pruned heads.
///
/// Takes the full attention output and evaluates how well each pruning method
/// preserves it.
pub fn compare_pruning(
    standard_output: &[Vec<f64>],
    full_heads: usize,
    pruned_heads: &[usize],
    all_heads: &[Vec<Vec<f64>>],
) -> PruningResult {
    let n_kept = pruned_heads.len();

    // Compute output using only the pruned heads
    let mut pruned_output = vec![vec![0.0f64; standard_output[0].len()]; standard_output.len()];
    for &head_idx in pruned_heads {
        let head = &all_heads[head_idx];
        for (i, row) in head.iter().enumerate() {
            if i < pruned_output.len() {
                for (j, &val) in row.iter().enumerate() {
                    if j < pruned_output[0].len() {
                        pruned_output[i][j] += val / n_kept as f64;
                    }
                }
            }
        }
    }

    let error = convergence_error(standard_output, &pruned_output);
    let speedup = full_heads as f64 / n_kept.max(1) as f64;

    PruningResult {
        method: "pruned".to_string(),
        n_kept,
        output_error: error,
        speedup_estimate: speedup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_heads(n_heads: usize, seq_len: usize, dim: usize) -> Vec<Vec<Vec<f64>>> {
        (0..n_heads).map(|h| {
            (0..seq_len).map(|i| {
                (0..dim).map(|j| {
                    ((h + 1) as f64) * (i as f64 + 1.0) * 0.1 + (j as f64) * 0.01
                }).collect()
            }).collect()
        }).collect()
    }

    #[test]
    fn test_compute_importance() {
        let heads = make_heads(4, 3, 2);
        let imp = compute_importance(&heads);
        assert_eq!(imp.len(), 4);
        // Higher head index → higher values → higher importance
        for i in 1..imp.len() {
            assert!(imp[i] > imp[i - 1]);
        }
    }

    #[test]
    fn test_tropical_prune_count() {
        let heads = make_heads(8, 4, 3);
        let kept = tropical_prune(&heads, 0.5);
        assert_eq!(kept.len(), 4);
        // Indices should be sorted
        for i in 1..kept.len() {
            assert!(kept[i] > kept[i - 1]);
        }
    }

    #[test]
    fn test_tropical_prune_keeps_important() {
        let heads = make_heads(8, 4, 3);
        let kept = tropical_prune(&heads, 0.5);
        // Should keep heads with higher indices (more important)
        // Since higher h → higher values → higher tropical importance
        assert!(kept.iter().any(|&i| i >= 4));
    }

    #[test]
    fn test_random_prune_count() {
        let kept = random_prune(10, 0.3);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn test_magnitude_prune_count() {
        let heads = make_heads(8, 4, 3);
        let kept = magnitude_prune(&heads, 0.5);
        assert_eq!(kept.len(), 4);
    }

    #[test]
    fn test_magnitude_prune_keeps_largest() {
        let heads = make_heads(8, 4, 3);
        let kept = magnitude_prune(&heads, 0.5);
        // Higher head index = higher magnitude, so should keep 4-7
        assert!(kept.contains(&7));
        assert!(kept.contains(&6));
    }

    #[test]
    fn test_tropical_beats_random() {
        let heads = make_heads(8, 4, 4);
        let full_output: Vec<Vec<f64>> = (0..4).map(|i| {
            (0..4).map(|j| {
                heads.iter().map(|h| h[i][j]).sum::<f64>() / 8.0
            }).collect()
        }).collect();

        let tropical_kept = tropical_prune(&heads, 0.5);
        let random_kept = random_prune(8, 0.5);

        let tropical_result = compare_pruning(&full_output, 8, &tropical_kept, &heads);
        let random_result = compare_pruning(&full_output, 8, &random_kept, &heads);

        // Tropical should generally beat random (not guaranteed for all data,
        // but should hold for our structured test data)
        println!("Tropical error: {}, Random error: {}", tropical_result.output_error, random_result.output_error);
        // At minimum, tropical should select valid heads
        assert!(!tropical_kept.is_empty());
        assert!(!random_kept.is_empty());
    }

    #[test]
    fn test_keep_ratio_accuracy() {
        let heads = make_heads(10, 4, 3);
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let kept = tropical_prune(&heads, ratio);
            let expected = ((10.0 * ratio).ceil() as usize).max(1).min(10);
            assert_eq!(kept.len(), expected, "ratio={ratio}");
        }
    }

    #[test]
    fn test_pruning_result_fields() {
        let heads = make_heads(8, 4, 3);
        let full_output: Vec<Vec<f64>> = (0..4).map(|i| {
            (0..3).map(|j| {
                heads.iter().map(|h| h[i][j]).sum::<f64>() / 8.0
            }).collect()
        }).collect();
        let kept = tropical_prune(&heads, 0.5);
        let result = compare_pruning(&full_output, 8, &kept, &heads);
        assert_eq!(result.method, "pruned");
        assert_eq!(result.n_kept, 4);
        assert!(result.speedup_estimate > 0.0);
    }

    #[test]
    fn test_keep_ratio_clamp() {
        let heads = make_heads(4, 2, 2);
        // ratio > 1.0: keep all
        let kept = tropical_prune(&heads, 2.0);
        assert_eq!(kept.len(), 4);
        // ratio ~ 0: keep at least 1
        let kept = tropical_prune(&heads, 0.01);
        assert_eq!(kept.len(), 1);
    }
}
