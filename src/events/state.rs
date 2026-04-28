use super::{EventKind, EventRecord};
use chrono::{DateTime, Utc};

const N_EVENTS: usize = 18;

/// Per-asset record of the most recent event of each of the 18 types.
///
/// Used to produce Block 6 (10 decayed event signal features) every minute.
#[derive(Clone, Default)]
pub struct EventState {
    /// Most recent (magnitude, detected_at) for each EventKind, by index.
    slots: [(f32, Option<DateTime<Utc>>); N_EVENTS],
}

impl EventState {
    /// Returns a new `EventState` updated with the given event. Never mutates.
    pub fn with_event(&self, event: &EventRecord) -> Self {
        let mut slots = self.slots;
        let idx = event.kind.index();
        slots[idx] = (event.magnitude, Some(event.detected_at));
        Self { slots }
    }

    /// Produce the 10-dimensional Block 6 feature vector.
    ///
    /// Mapping (from spec Section 6.2 Block 6):
    ///   [0]  insider buy recency       exp(-Δt / τ_insider_buy)
    ///   [1]  insider buy magnitude     decayed log-dollar value
    ///   [2]  analyst upgrade recency   exp(-Δt / τ_analyst)
    ///   [3]  analyst downgrade recency exp(-Δt / τ_analyst)
    ///   [4]  SEC filing recency        exp(-Δt / τ_sec)
    ///   [5]  10-K/10-Q recency         exp(-Δt / τ_sec_annual)
    ///   [6]  news intensity (24h)      (stubbed: raw count proxy)
    ///   [7]  news sentiment (24h)      per-asset sentiment from llama-swap model
    ///   [8]  earnings proximity        decayed by days_to_earnings (external)
    ///   [9]  ex-dividend proximity     decayed by days_to_exdiv (external)
    pub fn to_block6(
        &self,
        now: DateTime<Utc>,
        earnings_proximity: f32,
        exdiv_proximity: f32,
        news_sentiment: f32,
    ) -> [f32; 10] {
        let decay = |kind: EventKind| -> f32 {
            let (mag, ts) = self.slots[kind.index()];
            match ts {
                None => 0.0,
                Some(t) => {
                    let elapsed_min = (now - t).num_minutes().max(0) as f64;
                    let tau = kind.decay_tau_minutes();
                    (mag as f64 * (-elapsed_min / tau).exp()) as f32
                }
            }
        };

        let buy_recency = decay(EventKind::InsiderBuy);
        let buy_mag = {
            let (mag, ts) = self.slots[EventKind::InsiderBuy.index()];
            match ts {
                None => 0.0,
                Some(t) => {
                    let elapsed_min = (now - t).num_minutes().max(0) as f64;
                    let tau = EventKind::InsiderBuy.decay_tau_minutes();
                    (mag as f64 * (-elapsed_min / tau).exp()) as f32
                }
            }
        };

        [
            buy_recency,
            buy_mag,
            decay(EventKind::AnalystUpgrade),
            decay(EventKind::AnalystDowngrade),
            // SEC filing recency: max of 8K/other (high-freq)
            decay(EventKind::SecFiling8K).max(decay(EventKind::SecFilingOther)),
            // 10-K/10-Q recency
            decay(EventKind::SecFiling10K).max(decay(EventKind::SecFiling10Q)),
            // news intensity proxy (magnitude = article count, log-scaled)
            decay(EventKind::NewsHighImpact),
            news_sentiment,
            earnings_proximity,
            exdiv_proximity,
        ]
    }
}
