//! Configuration for nsfw-lib-rust - mirrors Python nsfw_scanner thresholds.
//! All thresholds configurable via environment variables with failsafe defaults.

use std::env;

/// Threshold mode: "default", "test2"
const DEFAULT_THRESHOLD_MODE: &str = "default";

/// Rating scale phrases (must match Python exactly)
pub const RATING_PHRASES: &[(i32, &str)] = &[
    (1, "no unsafe signal detected"),
    (2, "synthetic - not rated, scan it yourself"),
    (5, "women/girls present, no stronger unsafe signal detected"),
    (6, "romance - clothed, affectionate"),
    (7, "revealing, not fully explicit"),
    (8, "strongly suggestive"),
    (10, "explicit adult content"),
];

/// Rating verdicts (machine-readable)
pub const RATING_VERDICTS: &[(i32, &str)] = &[
    (1, "modesty-no-signal"),
    (2, "synthetic"),
    (5, "people-female"),
    (6, "romance"),
    (7, "revealing"),
    (8, "suggestive"),
    (10, "explicit"),
];

pub const NSFW_CLASSES: &[&str; 5] = &["drawings", "hentai", "neutral", "porn", "sexy"];

pub const MEAN_BGR: [f32; 3] = [104.0, 117.0, 123.0];

pub const JOINT_INPUT_SIZE: (u32, u32) = (299, 299);
pub const STUDENT_INPUT_SIZE: (u32, u32) = (224, 224);
pub const BINARY_INPUT_SIZE: (u32, u32) = (224, 224);

pub const STUDENT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STUDENT_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Model file names
pub const NSFW_JOINT_MODEL: &str = "nsfw_joint_model.onnx";
pub const NSFW_BINARY_MODEL: &str = "nsfw_model.onnx";
pub const NSFW_STUDENT_MODEL: &str = "nsfw_student.onnx";
/// FairFace ViT INT8 gender model - the default (mirrors Python).
pub const GENDER_MODEL_FAIRFACE: &str = "fairface_int8.onnx";
/// Legacy Levi/Adience gender model - fallback only.
pub const GENDER_MODEL: &str = "gender_model.onnx";
pub const FACE_MODEL: &str = "face_model.onnx";

/// GitHub placeholder for model downloads (repo does not exist yet)
pub const MODEL_DOWNLOAD_BASE: &str =
    "https://github.com/modesty-nsfw/models/releases/download/latest/";

/// Configuration struct
#[derive(Debug, Clone)]
pub struct ScannerConfig {
    pub threshold_explicit: f32,
    pub threshold_revealing: f32,
    pub threshold_middle: f32,
    pub joint_strong: f32,
    pub sexy_strong: f32,
    pub sexy_high: i32,
    pub female_threshold: f32,
    pub threshold_mode: String,
    pub model_dir: String,
    pub model_download_base: String,
    pub auto_download_models: bool,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ScannerConfig {
    pub fn from_env() -> Self {
        let threshold_mode = env::var("NSFW_THRESHOLD_MODE")
            .unwrap_or_else(|_| DEFAULT_THRESHOLD_MODE.to_string());

        let (threshold_explicit, threshold_revealing, threshold_middle, joint_strong, sexy_strong) =
            if threshold_mode == "test2" {
                (
                    env_var_float("NSFW_THRESHOLD_EXPLICIT", 0.6),
                    env_var_float("NSFW_THRESHOLD_REVEALING", 0.12),
                    env_var_float("NSFW_THRESHOLD_MIDDLE", 0.35),
                    env_var_float("NSFW_JOINT_STRONG", 0.75),
                    env_var_float("NSFW_SEXY_STRONG", 0.75),
                )
            } else {
                (
                    env_var_float("NSFW_THRESHOLD_EXPLICIT", 0.5),
                    env_var_float("NSFW_THRESHOLD_REVEALING", 0.2),
                    env_var_float("NSFW_THRESHOLD_MIDDLE", 0.35),
                    env_var_float("NSFW_JOINT_STRONG", 0.8),
                    env_var_float("NSFW_SEXY_STRONG", 0.8),
                )
            };

        Self {
            threshold_explicit,
            threshold_revealing,
            threshold_middle,
            joint_strong,
            sexy_strong,
            sexy_high: env_var_int("NSFW_SEXY_HIGH", 0),
            female_threshold: env_var_float("NSFW_FEMALE_THRESHOLD", 0.5),
            threshold_mode,
            model_dir: env::var("NSFW_MODEL_DIR")
                .unwrap_or_else(|_| default_model_dir()),
            model_download_base: env::var("NSFW_MODEL_DOWNLOAD_BASE")
                .unwrap_or_else(|_| MODEL_DOWNLOAD_BASE.to_string()),
            auto_download_models: env::var("NSFW_AUTO_DOWNLOAD_MODELS")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(true),
        }
    }

    pub fn rating_phrase(rating: i32) -> &'static str {
        RATING_PHRASES
            .iter()
            .find(|(r, _)| *r == rating)
            .map(|(_, phrase)| *phrase)
            .unwrap_or("unrated")
    }

    pub fn rating_verdict(rating: i32) -> &'static str {
        RATING_VERDICTS
            .iter()
            .find(|(r, _)| *r == rating)
            .map(|(_, verdict)| *verdict)
            .unwrap_or("unknown")
    }

    pub fn class_to_rating(class: &str, prob: f32, binary_score: Option<f32>) -> Option<i32> {
        match class {
            "porn" | "hentai" => {
                if let Some(bs) = binary_score {
                    if bs > Self::default().threshold_explicit {
                        Some(10)
                    } else if class == "hentai" && prob > Self::default().joint_strong {
                        Some(10)
                    } else if bs > Self::default().threshold_middle {
                        Some(7)
                    } else {
                        Some(6)
                    }
                } else if prob > Self::default().threshold_explicit
                    || (class == "porn" || class == "hentai")
                        && prob > Self::default().threshold_revealing
                {
                    Some(10)
                } else if class == "sexy" && prob > Self::default().threshold_revealing {
                    Some(6)
                } else {
                    None
                }
            }
            "sexy" => {
                if prob > Self::default().threshold_revealing {
                    if Self::default().sexy_high > 0 && prob > Self::default().sexy_strong {
                        Some(Self::default().sexy_high)
                    } else {
                        Some(6)
                    }
                } else {
                    None
                }
            }
            "neutral" | "drawings" => Some(1),
            _ => None,
        }
    }
}

fn env_var_float(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_var_int(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn default_model_dir() -> String {
    #[cfg(target_os = "windows")]
    {
        std::env::var("LOCALAPPDATA")
            .map(|d| format!("{}\\nsfw-lib-rust\\models", d))
            .unwrap_or_else(|_| "models".to_string())
    }
    #[cfg(not(target_os = "windows"))]
    {
        format!(
            "{}/.nsfw-lib-rust/models",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_phrase() {
        assert_eq!(ScannerConfig::rating_phrase(10), "explicit adult content");
        assert_eq!(ScannerConfig::rating_phrase(6), "romance - clothed, affectionate");
        assert_eq!(ScannerConfig::rating_phrase(99), "unrated");
    }

    #[test]
    fn test_class_to_rating() {
        // porn with binary agreement -> 10
        assert_eq!(
            ScannerConfig::class_to_rating("porn", 0.9, Some(0.6)),
            Some(10)
        );
        // porn without binary agreement -> 6
        assert_eq!(
            ScannerConfig::class_to_rating("porn", 0.9, Some(0.1)),
            Some(6)
        );
        // sexy -> 6
        assert_eq!(
            ScannerConfig::class_to_rating("sexy", 0.8, None),
            Some(6)
        );
        // neutral -> 1
        assert_eq!(
            ScannerConfig::class_to_rating("neutral", 0.9, None),
            Some(1)
        );
    }
}
