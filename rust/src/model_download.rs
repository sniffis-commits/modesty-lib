#[cfg(feature = "download")]
/// Model download logic - downloads ONNX weights from GitHub placeholder.
/// Daemon installs without models; downloads on first run if enabled.

use crate::config::ScannerConfig;
use std::path::Path;
use std::fs;

/// List of model files to download
pub const MODEL_FILES: &[&str] = &[
    "nsfw_joint_model.onnx",
    "nsfw_model.onnx",
    "gender_model.onnx",
    "face_model.onnx",
];

/// Ensure all models are available locally. Downloads missing ones from GitHub.
pub fn ensure_models(config: &ScannerConfig) -> Result<(), Box<dyn std::error::Error>> {
    if !config.auto_download_models {
        return Ok(());
    }

    let model_dir = Path::new(&config.model_dir);
    fs::create_dir_all(model_dir)?;

    for model_file in MODEL_FILES {
        let local_path = model_dir.join(model_file);
        if local_path.exists() {
            continue; // already have it
        }

        let remote_url = format!("{}{}", config.model_download_base, model_file);
        println!("Downloading model {} from {}", model_file, remote_url);

        match download_file(&remote_url, &local_path) {
            Ok(_) => println!("Downloaded {}", model_file),
            Err(e) => {
                eprintln!("Failed to download {}: {} (continuing without it)", model_file, e);
            }
        }
    }

    Ok(())
}

/// Download a file from URL to local path.
fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let mut response = client.get(url).send()?;
    if !response.status().is_success() {
        return Err(format!("HTTP {} for {}", response.status(), url).into());
    }

    let mut file = fs::File::create(path)?;
    std::io::copy(&mut response, &mut file)?;

    Ok(())
}

/// Check which models are available locally.
pub fn check_models(config: &ScannerConfig) -> Vec<(String, bool)> {
    let model_dir = Path::new(&config.model_dir);
    MODEL_FILES
        .iter()
        .map(|f| {
            let path = model_dir.join(f);
            (f.to_string(), path.exists())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_files_not_empty() {
        assert!(!MODEL_FILES.is_empty());
    }
}