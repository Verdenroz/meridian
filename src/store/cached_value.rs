use chrono::{DateTime, Duration, Utc};

/// A cached value with a staleness timestamp.
///
/// Every piece of polled data is wrapped in a `CachedValue`. If a poll fails,
/// the previous value is retained and served to the feature assembler so the
/// model always receives a complete feature vector.
#[derive(Clone)]
pub struct CachedValue<T: Clone> {
    pub value: T,
    #[allow(dead_code)]
    pub last_updated: DateTime<Utc>,
}

#[allow(dead_code)]
impl<T: Clone> CachedValue<T> {
    pub fn new(value: T) -> Self {
        Self { value, last_updated: Utc::now() }
    }

    /// Returns a new `CachedValue` with the given value and the current timestamp.
    /// Never mutates — always returns a fresh instance.
    pub fn update(&self, value: T) -> Self {
        Self { value, last_updated: Utc::now() }
    }

    pub fn is_stale(&self, max_age: Duration) -> bool {
        Utc::now() - self.last_updated > max_age
    }

    pub fn age(&self) -> Duration {
        Utc::now() - self.last_updated
    }
}
