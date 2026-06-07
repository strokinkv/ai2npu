use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct UnloadResponse {
    pub unloaded_model_count: usize,
}

pub fn unload_models(base_url: &str) -> Result<UnloadResponse> {
    let url = format!("{}/admin/models/unload", base_url.trim_end_matches('/'));
    let response = reqwest::blocking::Client::new()
        .post(&url)
        .send()
        .with_context(|| format!("failed to call {url}"))?
        .error_for_status()
        .with_context(|| format!("unload request failed: {url}"))?;
    let body = response
        .text()
        .context("failed to read unload response body")?;
    serde_json::from_str(&body).context("failed to parse unload response")
}
