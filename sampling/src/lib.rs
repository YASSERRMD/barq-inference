//! Sampling algorithms for LLM token generation
//!
//! Comprehensive implementation of all major sampling methods:
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//! - Min-p sampling
//! - Mirostat sampling
//! - Typical sampling
//! - Repetition penalty
//! - XTC (eXtending The Context) sampling

pub mod chain;
pub mod min_p;
pub mod mirostat;
pub mod penalties;
pub mod sampler;
pub mod temperature;
pub mod top_k;
pub mod top_p;
pub mod typical;
pub mod xtc;

pub use chain::SamplerChain;
pub use min_p::MinP;
pub use mirostat::{Mirostat, MirostatType};
pub use penalties::{FrequencyPenalty, PresencePenalty, RepetitionPenalty};
pub use sampler::{Sampler, SamplerType};
pub use temperature::Temperature;
pub use top_k::TopK;
pub use top_p::TopP;
pub use typical::Typical;
pub use xtc::XtcSampler;
