pub mod tropical;
pub mod attention;
pub mod pruning;
pub mod cache;
pub mod experiment;

pub use tropical::Tropical;
pub use attention::{
    standard_attention, tropical_attention, sparse_attention,
    convergence_error, temperature_sweep,
};
pub use pruning::{PruningResult, compute_importance, tropical_prune, random_prune, magnitude_prune, compare_pruning};
pub use cache::{KVCache, tropical_compress, uniform_compress, recent_compress, compression_error, compression_ratio};
pub use experiment::Experiment;
