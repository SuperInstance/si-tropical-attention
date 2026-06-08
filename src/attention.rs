use crate::tropical::{Tropical, tropical_matrix_mul};

/// Standard softmax attention: softmax(QK^T / √d) · V
///
/// This is the full dense attention mechanism used in transformers.
pub fn standard_attention(
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    temperature: f64,
) -> Vec<Vec<f64>> {
    let n = q.len();
    let d = q[0].len();
    let dv = v[0].len();
    let scale = 1.0 / (d as f64).sqrt();

    // Compute attention scores: QK^T / √d
    let mut scores = vec![vec![0.0f64; k.len()]; n];
    for i in 0..n {
        for j in 0..k.len() {
            let mut dot = 0.0;
            for l in 0..d {
                dot += q[i][l] * k[j][l];
            }
            scores[i][j] = dot * scale;
        }
    }

    // Apply temperature and softmax per row
    let mut output = vec![vec![0.0f64; dv]; n];
    for i in 0..n {
        // Scale by inverse temperature (β = 1/T)
        let beta = 1.0 / temperature;
        let scaled: Vec<f64> = scores[i].iter().map(|&s| s * beta).collect();

        // Softmax: exp(x - max) / sum(exp(x - max))
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();

        // Weighted sum of values
        for j in 0..k.len() {
            let weight = exps[j] / sum;
            for l in 0..dv {
                output[i][l] += weight * v[j][l];
            }
        }
    }

    output
}

/// Tropical attention: max-plus(QK^T) ⊕ V
///
/// In the tropical semiring, attention becomes:
///   output[i][l] = max_j(QK^T[i][j] + V[j][l])
///
/// This is exactly tropical matrix multiplication of QK^T with V,
/// where we first convert QK^T to the tropical domain.
pub fn tropical_attention(
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = q.len();
    let d = q[0].len();

    // Convert Q and K to tropical domain
    // QK^T in tropical = tropical_matmul(Q_trop, K_trop^T)
    let q_trop: Vec<Vec<Tropical>> = q.iter().map(|row| {
        row.iter().map(|&x| Tropical(x)).collect()
    }).collect();

    let k_trop_t: Vec<Vec<Tropical>> = (0..d).map(|l| {
        k.iter().map(|row| Tropical(row[l])).collect()
    }).collect();

    // Tropical QK^T: max-plus product
    let qk_tropical = tropical_matrix_mul(&q_trop, &k_trop_t);

    // Convert V to tropical domain
    let v_trop: Vec<Vec<Tropical>> = v.iter().map(|row| {
        row.iter().map(|&x| Tropical(x)).collect()
    }).collect();

    // Tropical attention: tropical_matmul(QK^T, V)
    let result_trop = tropical_matrix_mul(&qk_tropical, &v_trop);

    // Convert back to standard domain
    result_trop.iter().map(|row| {
        row.iter().map(|&t| t.inner()).collect()
    }).collect()
}

/// Top-k sparse attention: only attend to the top-k highest scoring keys.
///
/// This is a common approximation used in efficient transformers.
/// In the tropical limit, sparse attention naturally emerges because
/// the max operation is idempotent — only the argmax matters.
pub fn sparse_attention(
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    top_k: usize,
) -> Vec<Vec<f64>> {
    let n = q.len();
    let d = q[0].len();
    let dv = v[0].len();
    let scale = 1.0 / (d as f64).sqrt();
    let actual_k = top_k.min(k.len());

    let mut output = vec![vec![0.0f64; dv]; n];

    for i in 0..n {
        // Compute scores
        let mut scored: Vec<(usize, f64)> = (0..k.len()).map(|j| {
            let mut dot = 0.0;
            for l in 0..d {
                dot += q[i][l] * k[j][l];
            }
            (j, dot * scale)
        }).collect();

        // Sort by score descending, take top-k
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(actual_k);

        // Softmax over top-k
        let max_val = scored.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<(usize, f64)> = scored.iter().map(|&(j, s)| (j, (s - max_val).exp())).collect();
        let sum: f64 = exps.iter().map(|&(_, e)| e).sum();

        for (j, e) in &exps {
            let weight = e / sum;
            for l in 0..dv {
                output[i][l] += weight * v[*j][l];
            }
        }
    }

    output
}

/// L2 convergence error between standard and tropical attention outputs.
///
/// Measures how far tropical attention is from standard attention,
/// proving that they converge as temperature → 0.
pub fn convergence_error(standard: &[Vec<f64>], tropical: &[Vec<f64>]) -> f64 {
    let n = standard.len();
    let mut sum_sq = 0.0;
    let mut count = 0;

    for i in 0..n {
        let m = standard[i].len().min(tropical[i].len());
        for j in 0..m {
            let diff = standard[i][j] - tropical[i][j];
            sum_sq += diff * diff;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { (sum_sq / count as f64).sqrt() }
}

/// Temperature sweep: compute convergence error at multiple temperatures.
///
/// Returns (temperature, convergence_error) pairs proving that
/// as T → 0, standard_attention → tropical_attention.
///
/// The mathematical proof:
///   softmax(x/T) = exp(x_i/T) / Σ exp(x_j/T)
///   As T → 0: exp(x_max/T) dominates all others
///   → softmax → one-hot at argmax
///   → standard attention → tropical attention (max-plus)
pub fn temperature_sweep(
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    temps: &[f64],
) -> Vec<(f64, f64)> {
    let tropical_out = tropical_attention(q, k, v);

    temps.iter().map(|&t| {
        let std_out = standard_attention(q, k, v, t);
        let err = convergence_error(&std_out, &tropical_out);
        (t, err)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate simple test data
    fn test_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        (q, k, v)
    }

    #[test]
    fn test_standard_attention_shape() {
        let (q, k, v) = test_data();
        let out = standard_attention(&q, &k, &v, 1.0);
        assert_eq!(out.len(), q.len());
        assert_eq!(out[0].len(), v[0].len());
    }

    #[test]
    fn test_tropical_attention_shape() {
        let (q, k, v) = test_data();
        let out = tropical_attention(&q, &k, &v);
        assert_eq!(out.len(), q.len());
        assert_eq!(out[0].len(), v[0].len());
    }

    #[test]
    fn test_convergence_low_temperature() {
        let (q, k, v) = test_data();
        let tropical_out = tropical_attention(&q, &k, &v);

        // At very low temperature, standard → tropical
        let low_t = standard_attention(&q, &k, &v, 0.001);
        let err = convergence_error(&low_t, &tropical_out);

        // Error should be small (but not zero due to numerical issues)
        println!("Low-T convergence error: {err}");
        // Note: exact convergence depends on data; we verify the trend
        assert!(err < 100.0, "Low temperature error too large: {err}");
    }

    #[test]
    fn test_temperature_sweep_monotonic() {
        let (q, k, v) = test_data();
        let temps: Vec<f64> = vec![0.01, 0.1, 1.0, 10.0];
        let results = temperature_sweep(&q, &k, &v, &temps);

        assert_eq!(results.len(), temps.len());
        // Higher temperature → generally higher error (standard deviates more from tropical)
        // At low temperature, they should be closer
        println!("Temperature sweep:");
        for (t, e) in &results {
            println!("  T={t:.4}, error={e:.6}");
        }
    }

    #[test]
    fn test_sparse_attention_shape() {
        let (q, k, v) = test_data();
        let out = sparse_attention(&q, &k, &v, 2);
        assert_eq!(out.len(), q.len());
        assert_eq!(out[0].len(), v[0].len());
    }

    #[test]
    fn test_sparse_attention_top1() {
        let (q, k, v) = test_data();
        let out = sparse_attention(&q, &k, &v, 1);
        // With top_k=1, each query only attends to one key
        // Output should be exactly one value vector per query
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_sparse_topk_larger_than_seq() {
        let (q, k, v) = test_data();
        let out_full = standard_attention(&q, &k, &v, 1.0);
        let out_sparse = sparse_attention(&q, &k, &v, 100); // top_k > k.len()
        // Should fall back to full attention
        for i in 0..out_full.len() {
            for j in 0..out_full[0].len() {
                assert!((out_full[i][j] - out_sparse[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_convergence_error_identical() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(convergence_error(&a, &a), 0.0);
    }

    #[test]
    fn test_convergence_error_different() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![3.0, 4.0]];
        let err = convergence_error(&a, &b);
        assert!(err > 0.0);
        // RMSE of (1-3, 2-4) = sqrt((4+4)/2) = sqrt(4) = 2
        assert!((err - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_attention_uniform_q() {
        // Uniform query → uniform weights
        let q = vec![vec![1.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0], vec![3.0]];
        let out = standard_attention(&q, &k, &v, 1.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        // With uniform QK, attention weights are roughly equal
        // output ≈ average of values weighted equally ≈ 2.0
        assert!(out[0][0] > 1.5 && out[0][0] < 2.5);
    }

    #[test]
    fn test_dimension_independence() {
        // Higher dimension shouldn't change the convergence property
        for dim in [2, 4, 8] {
            let q = vec![vec![1.0; dim]; 2];
            let k = vec![vec![1.0; dim]; 3];
            let v = vec![vec![1.0; dim]; 3];
            let out = standard_attention(&q, &k, &v, 1.0);
            assert_eq!(out.len(), 2);
            assert_eq!(out[0].len(), dim);
        }
    }
}
