//! Outbound alert notification (§1.11: every action notifies). One optional
//! webhook, configured by env var so the mechanism stays outside the model's
//! reach and absent config degrades to journal + log only — never to a
//! blocked alert. The JSON body is the alert row itself; an `X-Title` header
//! carries a one-line summary (ntfy renders it, generic receivers ignore it).

use serde::Serialize;
use std::time::Duration;

pub const WEBHOOK_ENV: &str = "MERIDIAN_ALERT_WEBHOOK";
const SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// Fire the webhook if configured. Failures are logged, never propagated —
/// notification is best-effort, the journal row is the record of truth.
pub async fn alert_webhook<T: Serialize>(title: &str, row: &T) {
    let Ok(url) = std::env::var(WEBHOOK_ENV) else {
        return;
    };
    if url.trim().is_empty() {
        return;
    }
    let result = async {
        let client = reqwest::Client::builder().timeout(SEND_TIMEOUT).build()?;
        client
            .post(url.trim())
            .header("X-Title", title)
            .json(row)
            .send()
            .await?
            .error_for_status()?;
        Ok::<_, anyhow::Error>(())
    }
    .await;
    match result {
        Ok(()) => tracing::info!(title, "alert webhook delivered"),
        Err(e) => tracing::warn!(err = %e, "alert webhook failed (journal row still recorded)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// End-to-end against a throwaway local listener: the webhook must POST
    /// the serialized row and include the X-Title header. The unset-env noop
    /// case is folded in (not a separate test) because both mutate WEBHOOK_ENV
    /// and the default test runner is parallel.
    #[tokio::test]
    async fn posts_json_with_title_header() {
        unsafe { std::env::remove_var(WEBHOOK_ENV) };
        alert_webhook("noop when unset", &serde_json::json!({})).await;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .unwrap();
            req
        });

        // SAFETY: test-only env mutation; tests in this module run serially.
        unsafe { std::env::set_var(WEBHOOK_ENV, format!("http://{addr}/alert")) };
        let row = serde_json::json!({"level": "break_glass", "session_date": "2026-08-03"});
        alert_webhook("Meridian break-glass", &row).await;
        unsafe { std::env::remove_var(WEBHOOK_ENV) };

        let req = server.join().unwrap();
        assert!(req.starts_with("POST /alert"));
        assert!(req.to_lowercase().contains("x-title: meridian break-glass"));
        assert!(req.contains(r#""level":"break_glass""#));
    }
}
