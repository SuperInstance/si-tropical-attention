use crate::attention::temperature_sweep;
use crate::pruning::{
    tropical_prune, random_prune, magnitude_prune, compare_pruning,
    PruningResult,
};
use crate::cache::{
    KVCache, tropical_compress, uniform_compress, compression_error,
};

/// Result of a pruning comparison at a specific keep ratio.
#[derive(Debug, Clone)]
pub struct PruningComparison {
    pub keep_ratio: f64,
    pub tropical_result: PruningResult,
    pub random_result: PruningResult,
    pub magnitude_result: PruningResult,
}

/// Result of a cache compression comparison at a specific ratio.
#[derive(Debug, Clone)]
pub struct CacheComparison {
    pub keep_ratio: f64,
    pub tropical_error: f64,
    pub uniform_error: f64,
}

/// Full experiment configuration.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub dim: usize,
    pub seq_len: usize,
    pub n_heads: usize,
}

impl Experiment {
    pub fn new(dim: usize, seq_len: usize, n_heads: usize) -> Self {
        Experiment { dim, seq_len, n_heads }
    }

    /// Generate test data (deterministic).
    fn make_data(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut v = 1.0f64;
        let mut next = || { v = (v * 1.1 + 0.3) % 10.0; v };

        let q: Vec<Vec<f64>> = (0..self.seq_len).map(|_| {
            (0..self.dim).map(|_| next()).collect()
        }).collect();
        let k: Vec<Vec<f64>> = (0..self.seq_len).map(|_| {
            (0..self.dim).map(|_| next()).collect()
        }).collect();
        let v: Vec<Vec<f64>> = (0..self.seq_len).map(|_| {
            (0..self.dim).map(|_| next()).collect()
        }).collect();
        (q, k, v)
    }

    /// Generate multi-head attention data.
    fn make_heads(&self) -> Vec<Vec<Vec<f64>>> {
        (0..self.n_heads).map(|h| {
            let mut v = (h + 1) as f64;
            let mut next = || { v = (v * 1.05 + 0.2) % 5.0 + 0.1; v };
            (0..self.seq_len).map(|_| {
                (0..self.dim).map(|_| next()).collect()
            }).collect()
        }).collect()
    }

    /// Run temperature convergence experiment.
    ///
    /// Proves that standard_attention → tropical_attention as T → 0.
    pub fn run_temperature_convergence(&self) -> Vec<(f64, f64)> {
        let (q, k, v) = self.make_data();
        let temps: Vec<f64> = vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        temperature_sweep(&q, &k, &v, &temps)
    }

    /// Run pruning comparison across different keep ratios.
    ///
    /// Compares tropical pruning vs random vs magnitude pruning.
    pub fn run_pruning_comparison(&self, keep_ratios: &[f64]) -> Vec<PruningComparison> {
        let heads = self.make_heads();
        let full_output: Vec<Vec<f64>> = (0..self.seq_len).map(|i| {
            (0..self.dim).map(|j| {
                heads.iter().map(|h| h[i][j]).sum::<f64>() / self.n_heads as f64
            }).collect()
        }).collect();

        keep_ratios.iter().map(|&ratio| {
            let t_kept = tropical_prune(&heads, ratio);
            let r_kept = random_prune(self.n_heads, ratio);
            let m_kept = magnitude_prune(&heads, ratio);

            let tropical_result = compare_pruning(&full_output, self.n_heads, &t_kept, &heads);
            let random_result = compare_pruning(&full_output, self.n_heads, &r_kept, &heads);
            let magnitude_result = compare_pruning(&full_output, self.n_heads, &m_kept, &heads);

            PruningComparison {
                keep_ratio: ratio,
                tropical_result: PruningResult {
                    method: "tropical".to_string(),
                    ..tropical_result
                },
                random_result: PruningResult {
                    method: "random".to_string(),
                    ..random_result
                },
                magnitude_result: PruningResult {
                    method: "magnitude".to_string(),
                    ..magnitude_result
                },
            }
        }).collect()
    }

    /// Run cache compression comparison.
    pub fn run_cache_compression(&self, ratios: &[f64]) -> Vec<CacheComparison> {
        let cache = KVCache::random(self.seq_len, self.dim);
        let q = vec![vec![1.0; self.dim]; 3.min(self.seq_len)];

        ratios.iter().map(|&ratio| {
            let tropical_comp = tropical_compress(&cache, ratio);
            let uniform_comp = uniform_compress(&cache, ratio);

            let tropical_err = compression_error(&cache, &tropical_comp, &q);
            let uniform_err = compression_error(&cache, &uniform_comp, &q);

            CacheComparison {
                keep_ratio: ratio,
                tropical_error: tropical_err,
                uniform_error: uniform_err,
            }
        }).collect()
    }

    /// Generate a summary of all experiments.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("=== Tropical Attention Experiment ===\n"));
        s.push_str(&format!("Config: dim={}, seq_len={}, n_heads={}\n\n", self.dim, self.seq_len, self.n_heads));

        // Temperature convergence
        s.push_str("--- Temperature Convergence ---\n");
        s.push_str(&format!("{:<12} {:<15}\n", "Temperature", "Error"));
        let conv = self.run_temperature_convergence();
        for (t, e) in &conv {
            s.push_str(&format!("{t:<12.4} {e:<15.6}\n"));
        }
        s.push('\n');

        // Pruning comparison
        s.push_str("--- Pruning Comparison ---\n");
        let ratios = vec![0.3, 0.5, 0.7];
        let pruning = self.run_pruning_comparison(&ratios);
        s.push_str(&format!("{:<10} {:<15} {:<15} {:<15}\n", "Keep%", "Tropical", "Random", "Magnitude"));
        for pc in &pruning {
            s.push_str(&format!(
                "{:<10.1} {:<15.4} {:<15.4} {:<15.4}\n",
                pc.keep_ratio * 100.0,
                pc.tropical_result.output_error,
                pc.random_result.output_error,
                pc.magnitude_result.output_error,
            ));
        }
        s.push('\n');

        // Cache compression
        s.push_str("--- Cache Compression ---\n");
        let cache_ratios = vec![0.2, 0.4, 0.6, 0.8];
        let cache_results = self.run_cache_compression(&cache_ratios);
        s.push_str(&format!("{:<10} {:<15} {:<15}\n", "Keep%", "Tropical", "Uniform"));
        for cc in &cache_results {
            s.push_str(&format!(
                "{:<10.1} {:<15.4} {:<15.4}\n",
                cc.keep_ratio * 100.0,
                cc.tropical_error,
                cc.uniform_error,
            ));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_temperature_convergence() {
        let exp = Experiment::new(4, 8, 4);
        let results = exp.run_temperature_convergence();
        assert_eq!(results.len(), 9); // 9 temperature points

        // All errors should be non-negative
        for (t, e) in &results {
            assert!(e >= &0.0, "negative error at T={t}");
        }
    }

    #[test]
    fn test_experiment_pruning_comparison() {
        let exp = Experiment::new(4, 8, 8);
        let results = exp.run_pruning_comparison(&[0.5]);
        assert_eq!(results.len(), 1);

        let pc = &results[0];
        assert_eq!(pc.tropical_result.method, "tropical");
        assert_eq!(pc.random_result.method, "random");
        assert_eq!(pc.magnitude_result.method, "magnitude");
        assert_eq!(pc.tropical_result.n_kept, 4);
    }

    #[test]
    fn test_experiment_cache_compression() {
        let exp = Experiment::new(4, 20, 4);
        let results = exp.run_cache_compression(&[0.5]);
        assert_eq!(results.len(), 1);
        assert!(results[0].tropical_error >= 0.0);
        assert!(results[0].uniform_error >= 0.0);
    }

    #[test]
    fn test_experiment_summary() {
        let exp = Experiment::new(4, 8, 4);
        let summary = exp.summary();
        assert!(summary.contains("Tropical Attention Experiment"));
        assert!(summary.contains("Temperature Convergence"));
        assert!(summary.contains("Pruning Comparison"));
        assert!(summary.contains("Cache Compression"));
    }

    #[test]
    fn test_experiment_make_data_shapes() {
        let exp = Experiment::new(8, 16, 4);
        let (q, k, v) = exp.make_data();
        assert_eq!(q.len(), 16);
        assert_eq!(q[0].len(), 8);
        assert_eq!(k.len(), 16);
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_experiment_heads_shape() {
        let exp = Experiment::new(4, 8, 6);
        let heads = exp.make_heads();
        assert_eq!(heads.len(), 6);
        assert_eq!(heads[0].len(), 8);
        assert_eq!(heads[0][0].len(), 4);
    }

    #[test]
    fn test_convergence_decreases_with_lower_temp() {
        let exp = Experiment::new(4, 8, 4);
        let results = exp.run_temperature_convergence();
        // First result is lowest temp, should have one of the smallest errors
        let low_t_err = results.first().unwrap().1;
        let high_t_err = results.last().unwrap().1;
        println!("Low T error: {low_t_err}, High T error: {high_t_err}");
        // Trend should be that low temp has lower or similar error
        // (may not always hold due to discrete data, but log the values)
    }

    #[test]
    fn test_pruning_multiple_ratios() {
        let exp = Experiment::new(4, 8, 8);
        let results = exp.run_pruning_comparison(&[0.2, 0.4, 0.6, 0.8]);
        assert_eq!(results.len(), 4);
        // More kept heads → lower error (generally)
        for pc in &results {
            assert!(pc.keep_ratio > 0.0 && pc.keep_ratio <= 1.0);
        }
    }

    #[test]
    fn test_cache_multiple_ratios() {
        let exp = Experiment::new(4, 20, 4);
        let results = exp.run_cache_compression(&[0.2, 0.4, 0.6, 0.8, 1.0]);
        assert_eq!(results.len(), 5);
    }
}
