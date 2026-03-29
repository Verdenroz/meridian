use crate::features::FeatureRow;

/// Serialize a `FeatureRow` to a newline-terminated JSON object on stdout.
///
/// This is the contract boundary between the Rust DataBus and the Python pipeline.
/// Python reads line-by-line: `for line in sys.stdin: row = json.loads(line)`.
pub fn emit_row(row: &FeatureRow) {
    // serde_json::to_string is infallible for a well-formed FeatureRow
    let line = serde_json::to_string(row).expect("FeatureRow serialization failed");
    println!("{line}");
}
