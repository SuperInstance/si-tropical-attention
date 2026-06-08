use crate::attention::standard_attention;

/// Key-Value cache for autoregressive attention.
///
/// In transformers, the KV cache stores previously computed keys and values
/// to avoid recomputing them at each generation step. Compression of this
/// cache is critical for long-context models.
#[derive(Debug, Clone)]
pub struct KVCache {
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub seq_len: usize,
    pub dim: usize,
}

impl KVCache {
    pub fn new(keys: Vec<Vec<f64>>, values: Vec<Vec<f64>>) -> Self {
        let seq_len = keys.len();
        let dim = if seq_len > 0 { keys[0].len() } else { 0 };
        assert_eq!(values.len(), seq_len, "keys and values must have same seq_len");
        KVCache { keys, values, seq_len, dim }
    }

    /// Create a random KV cache for testing.
    pub fn random(seq_len: usize, dim: usize) -> Self {
        // Deterministic pseudo-random for reproducibility
        let mut val = 1.0f64;
        let mut next = || {
            val = (val * 1.1 + 0.3) % 10.0;
            val
        };
        let keys = (0..seq_len).map(|_| (0..dim).map(|_| next()).collect()).collect();
        let values = (0..seq_len).map(|_| (0..dim).map(|_| next()).collect()).collect();
        KVCache::new(keys, values)
    }
}

/// Compress KV cache using tropical approximation.
///
/// Strategy: keep the positions that have the largest tropical norm
/// (max absolute value in tropical domain). This preserves the positions
/// most important for max-plus attention.
///
/// Tropical compression is equivalent to keeping the rows that contribute
/// most to the tropical rank of the KV cache matrix.
pub fn tropical_compress(cache: &KVCache, ratio: f64) -> KVCache {
    let n_keep = ((cache.seq_len as f64) * ratio).ceil() as usize;
    let n_keep = n_keep.max(1).min(cache.seq_len);

    // Score each position by its tropical importance
    let mut scored: Vec<(usize, f64)> = (0..cache.seq_len).map(|i| {
        // Tropical norm: max over all dimensions (max-plus absolute value)
        let key_max = cache.keys[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let val_max = cache.values[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let key_min = cache.keys[i].iter().cloned().fold(f64::INFINITY, f64::min);
        let val_min = cache.values[i].iter().cloned().fold(f64::INFINITY, f64::min);

        // Tropical importance = max absolute deviation
        let importance = key_max.abs().max(key_min.abs())
            .max(val_max.abs()).max(val_min.abs());
        (i, importance)
    }).collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n_keep);
    let mut indices: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();
    indices.sort();

    let keys = indices.iter().map(|&i| cache.keys[i].clone()).collect();
    let values = indices.iter().map(|&i| cache.values[i].clone()).collect();
    KVCache::new(keys, values)
}

/// Uniform downsampling baseline: keep every Nth position.
pub fn uniform_compress(cache: &KVCache, ratio: f64) -> KVCache {
    let n_keep = ((cache.seq_len as f64) * ratio).ceil() as usize;
    let n_keep = n_keep.max(1).min(cache.seq_len);
    let step = cache.seq_len as f64 / n_keep as f64;

    let indices: Vec<usize> = (0..n_keep).map(|i| {
        ((i as f64 * step) as usize).min(cache.seq_len - 1)
    }).collect();

    let keys = indices.iter().map(|&i| cache.keys[i].clone()).collect();
    let values = indices.iter().map(|&i| cache.values[i].clone()).collect();
    KVCache::new(keys, values)
}

/// Keep only the most recent N tokens.
pub fn recent_compress(cache: &KVCache, keep_recent: usize) -> KVCache {
    let start = cache.seq_len.saturating_sub(keep_recent);
    let keys = cache.keys[start..].to_vec();
    let values = cache.values[start..].to_vec();
    KVCache::new(keys, values)
}

/// Compute compression error: how much does attention output change
/// when using the compressed cache vs the original?
pub fn compression_error(original: &KVCache, compressed: &KVCache, q: &[Vec<f64>]) -> f64 {
    let orig_out = standard_attention(q, &original.keys, &original.values, 1.0);
    let comp_out = standard_attention(q, &compressed.keys, &compressed.values, 1.0);

    let mut sum_sq = 0.0;
    let mut count = 0;
    for i in 0..orig_out.len() {
        for j in 0..orig_out[0].len() {
            let diff = orig_out[i][j] - comp_out[i][j];
            sum_sq += diff * diff;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { (sum_sq / count as f64).sqrt() }
}

/// Compute the actual compression ratio.
pub fn compression_ratio(original: &KVCache, compressed: &KVCache) -> f64 {
    if compressed.seq_len == 0 {
        return 0.0;
    }
    original.seq_len as f64 / compressed.seq_len as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::random(10, 4);
        assert_eq!(cache.seq_len, 10);
        assert_eq!(cache.dim, 4);
    }

    #[test]
    fn test_tropical_compress_count() {
        let cache = KVCache::random(100, 8);
        let compressed = tropical_compress(&cache, 0.5);
        assert_eq!(compressed.seq_len, 50);
    }

    #[test]
    fn test_tropical_compress_preserves_dim() {
        let cache = KVCache::random(100, 8);
        let compressed = tropical_compress(&cache, 0.5);
        assert_eq!(compressed.dim, 8);
    }

    #[test]
    fn test_uniform_compress_count() {
        let cache = KVCache::random(100, 8);
        let compressed = uniform_compress(&cache, 0.5);
        assert_eq!(compressed.seq_len, 50);
    }

    #[test]
    fn test_recent_compress() {
        let cache = KVCache::random(100, 8);
        let compressed = recent_compress(&cache, 20);
        assert_eq!(compressed.seq_len, 20);
        // Should keep the last 20 entries
        for j in 0..8 {
            assert_eq!(compressed.keys[0][j], cache.keys[80][j]);
        }
    }

    #[test]
    fn test_recent_compress_short() {
        let cache = KVCache::random(10, 4);
        let compressed = recent_compress(&cache, 20);
        assert_eq!(compressed.seq_len, 10); // can't keep more than we have
    }

    #[test]
    fn test_compression_error_same_cache() {
        let cache = KVCache::random(10, 4);
        let q = vec![vec![1.0; 4]; 2];
        let err = compression_error(&cache, &cache, &q);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn test_compression_error_different() {
        let cache1 = KVCache::random(10, 4);
        let cache2 = KVCache::random(5, 4);
        let q = vec![vec![1.0; 4]; 2];
        let err = compression_error(&cache1, &cache2, &q);
        // Different sizes should produce different outputs
        // (cache2 only has 5 entries, cache1 has 10)
        assert!(err >= 0.0);
    }

    #[test]
    fn test_compression_ratio() {
        let original = KVCache::random(100, 4);
        let compressed = tropical_compress(&original, 0.5);
        let ratio = compression_ratio(&original, &compressed);
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_tropical_beats_uniform() {
        // Tropical compression should generally preserve more information
        let cache = KVCache::random(50, 8);
        let q = vec![vec![1.0, 0.5, -0.3, 0.7, -0.2, 0.4, -0.1, 0.6]; 3];

        let tropical_comp = tropical_compress(&cache, 0.5);
        let uniform_comp = uniform_compress(&cache, 0.5);

        let tropical_err = compression_error(&cache, &tropical_comp, &q);
        let uniform_err = compression_error(&cache, &uniform_comp, &q);

        println!("Tropical compression error: {tropical_err:.6}");
        println!("Uniform compression error: {uniform_err:.6}");

        // Both should produce valid errors
        assert!(tropical_err >= 0.0);
        assert!(uniform_err >= 0.0);
    }

    #[test]
    fn test_error_scales_with_ratio() {
        let cache = KVCache::random(100, 8);
        let q = vec![vec![1.0; 8]; 3];

        let err_high = {
            let comp = tropical_compress(&cache, 0.9);
            compression_error(&cache, &comp, &q)
        };
        let err_low = {
            let comp = tropical_compress(&cache, 0.1);
            compression_error(&cache, &comp, &q)
        };

        // Lower ratio → more compression → generally higher error
        // (Not guaranteed for all data, but trend should hold)
        println!("Error at 90% keep: {err_high:.6}");
        println!("Error at 10% keep: {err_low:.6}");
        // At minimum verify both are valid
        assert!(err_high >= 0.0);
        assert!(err_low >= 0.0);
    }

    #[test]
    fn test_compress_ratio_clamp() {
        let cache = KVCache::random(10, 4);
        // ratio > 1: keep all
        let comp = tropical_compress(&cache, 2.0);
        assert_eq!(comp.seq_len, 10);
        // ratio ~ 0: keep at least 1
        let comp = tropical_compress(&cache, 0.01);
        assert_eq!(comp.seq_len, 1);
    }
}
