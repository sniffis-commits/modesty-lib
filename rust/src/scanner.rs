//! Image scanning logic - mirrors Python nsfw_scanner.analyze_pil_image exactly.
//! Rating branches are identical to Python:
//!   porn/hentai + binary agrees -> 10 (explicit)
//!   porn/hentai + binary disagrees -> 7 (revealing) or 6 (revealing/skin)
//!   sexy -> 6 (romance) or 8 (if sexy_high enabled)
//!   a female crop -> 5 (women/girls present)
//!   nothing found -> 1 (no unsafe signal)
//!
//! Gender model runs on quadrant crops (4-section), not face-crop only.
//!
//! When built without the `onnx` feature, all scans return rating: None
//! (not assessed). Enable with: cargo build --features onnx

use crate::config::ScannerConfig;
use crate::preprocessing::{
    array4_to_onnx_input, preprocess_gender, preprocess_nsfw, preprocess_nsfw_joint,
};
use image::{DynamicImage, RgbImage};
use std::path::{Path, PathBuf};

#[cfg(feature = "onnx")]
use ort::environment::Environment;
#[cfg(feature = "onnx")]
use ort::session::Session;

/// Result of scanning a single image.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanResult {
    pub rating: Option<i32>,
    pub verdict: Option<String>,
    pub phrase: Option<String>,
    pub nsfw_score: Option<f32>,
    pub female_score: Option<f32>,
    pub male_score: Option<f32>,
    pub gender: Option<String>,
    pub classes: Option<serde_json::Value>,
    pub decided_by: Option<String>,
}

/// Session holding loaded ONNX models.
pub struct ScannerSession {
    #[cfg(feature = "onnx")]
    pub env: Environment,
    #[cfg(feature = "onnx")]
    pub nsfw_session: Option<Session>,
    #[cfg(feature = "onnx")]
    pub binary_session: Option<Session>,
    #[cfg(feature = "onnx")]
    pub gender_session: Option<Session>,
    #[cfg(feature = "onnx")]
    pub face_session: Option<Session>,
    #[cfg(not(feature = "onnx"))]
    pub _placeholder: (),
    /// Output order of the gender model: (label at index 0, label at index 1).
    ///
    /// Mirrors Python's `session._nsfw_gender_output_order`. The FairFace ViT
    /// INT8 model emits `("female", "male")`, so female is index 0 and male
    /// is index 1. The legacy Levi/Adience model emits `("male", "female")`.
    pub gender_labels: [&'static str; 2],
    /// Name of the gender model file that was loaded (None when unavailable).
    pub gender_model_name: Option<String>,
}

impl ScannerSession {
    /// Index of the "female" class in the gender model's output vector.
    pub fn gender_female_index(&self) -> usize {
        self.gender_labels
            .iter()
            .position(|l| *l == "female")
            .unwrap_or(0)
    }

    /// Index of the "male" class in the gender model's output vector.
    pub fn gender_male_index(&self) -> usize {
        self.gender_labels
            .iter()
            .position(|l| *l == "male")
            .unwrap_or(1)
    }
}

/// Mirrors Python `_extract_positive_score(prediction, positive_index=1)`.
///
/// Pulls the positive-class probability out of the common ONNX output
/// shapes: a single value is taken as-is, otherwise the second entry is
/// the positive class.
pub fn extract_positive_score(flat: &[f32]) -> Option<f32> {
    if flat.is_empty() {
        return None;
    }
    if flat.len() == 1 {
        return Some(flat[0]);
    }
    Some(flat[1])
}

/// Mirrors Python `_run_gender_both()`: softmax the raw model output and
/// return `(female_score, male_score)` using the recorded output order.
pub fn gender_scores(
    labels: &[&str],
    flat: &[f32],
) -> Option<(f32, f32)> {
    if flat.len() < 2 {
        return None;
    }
    let female_index = labels.iter().position(|l| *l == "female").unwrap_or(0);
    let male_index = labels.iter().position(|l| *l == "male").unwrap_or(1);
    // Apply softmax to convert logits to probabilities.
    let max_val = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = flat.iter().map(|v| (v - max_val).exp()).sum();
    let female = (flat[female_index] - max_val).exp() / exp_sum;
    let male = (flat[male_index] - max_val).exp() / exp_sum;
    Some((female, male))
}

/// Create a scanner session from model directory.
pub fn create_session(model_dir: &str) -> Result<ScannerSession, Box<dyn std::error::Error>> {
    #[cfg(feature = "onnx")]
    {
        let env = ort::init().with_name("nsfw-scanner").build()?;

        let path = Path::new(model_dir);

        let nsfw_session = load_onnx_session(&env, path.join("nsfw_joint_model.onnx"));
        let binary_session = load_onnx_session(&env, path.join("nsfw_model.onnx"));
        let (gender_session, gender_labels, gender_model_name) =
            load_gender_session(&env, path);
        let face_session = load_onnx_session(&env, path.join("face_model.onnx"));

        Ok(ScannerSession {
            env,
            nsfw_session,
            binary_session,
            gender_session,
            face_session,
            gender_labels,
            gender_model_name,
        })
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = model_dir;
        Ok(ScannerSession {
            _placeholder: (),
            gender_labels: GENDER_LABELS_FAIRFACE,
            gender_model_name: None,
        })
    }
}

/// Output order of the gender model, mirroring Python's GENDER_LABELS.
/// FairFace ViT INT8 emits `["female", "male"]`.
pub const GENDER_LABELS_FAIRFACE: [&str; 2] = ["female", "male"];
/// Legacy Levi/Adience output order: `["male", "female"]`.
pub const GENDER_LABELS_LEVI: [&str; 2] = ["male", "female"];

/// Mirrors Python `load_gender_model()`.
///
/// Order of preference:
///   1. `NSFW_GENDER_MODEL` environment variable (explicit override)
///   2. `fairface_int8.onnx` (FairFace ViT INT8 - the default)
///   3. `gender_model.onnx` (legacy Levi/Adience model)
///
/// Returns the session, the output order and the model file name.
#[cfg(feature = "onnx")]
fn load_gender_session(
    ort_env: &Environment,
    model_dir: &Path,
) -> (
    Option<Session>,
    [&'static str; 2],
    Option<String>,
) {
    // 1. Explicit override (same as Python's NSFW_GENDER_MODEL env var).
    if let Ok(explicit) = std::env::var("NSFW_GENDER_MODEL") {
        if !explicit.is_empty() {
            let explicit_path = PathBuf::from(&explicit);
            if explicit_path.exists() {
                if let Some(session) = load_onnx_session(ort_env, explicit_path) {
                    return (
                        Some(session),
                        gender_labels_for(&explicit),
                        Some(explicit),
                    );
                }
            }
        }
    }

    // 2. FairFace ViT INT8 first, 3. legacy Levi/Adience as fallback.
    for candidate in [crate::config::GENDER_MODEL_FAIRFACE, crate::config::GENDER_MODEL] {
        let path = model_dir.join(candidate);
        if !path.exists() {
            continue;
        }
        if let Some(session) = load_onnx_session(ort_env, path.clone()) {
            return (
                Some(session),
                gender_labels_for(&candidate),
                Some(path.to_string_lossy().to_string()),
            );
        }
    }

    (None, GENDER_LABELS_FAIRFACE, None)
}

/// Mirrors Python's `load_gender_model()` output-order selection:
/// the FairFace ViT INT8 models emit `["female", "male"]`, everything else
/// keeps the legacy Levi/Adience order `["male", "female"]`.
pub fn gender_labels_for(model_name: &str) -> [&'static str; 2] {
    let name = model_name.to_lowercase();
    let file_name = name.rsplit(['/', '\\']).next().unwrap_or(&name);
    if file_name == "fairface_int8.onnx" || file_name == "gender_int8.onnx" {
        GENDER_LABELS_FAIRFACE
    } else {
        GENDER_LABELS_LEVI
    }
}

#[cfg(feature = "onnx")]
fn load_onnx_session(env: &Environment, path: PathBuf) -> Option<Session> {
    if !path.exists() {
        return None;
    }
    let mut builder = Session::builder(env).ok()?;
    builder.commit_from_file(path).ok()
}

#[cfg(not(feature = "onnx"))]
fn load_onnx_session(_env: &(), _path: PathBuf) -> Option<()> {
    None
}

/// Scan an image file path and return ScanResult.
pub fn scan_image_file(
    session: &mut ScannerSession,
    image_path: &str,
) -> Result<ScanResult, Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    scan_image(session, &img)
}

/// The joint model's verdict - mirrors the `if classes is not None:` block of
/// Python's `analyze_pil_image`.
#[derive(Debug, Clone)]
pub struct JointDecision {
    pub rating: Option<i32>,
    pub decided_by: Option<String>,
    /// `classes["porn"] + classes["hentai"]` - the "skin hypothesis".
    pub nsfw_score: Option<f32>,
}

/// The gender model's verdict - mirrors the gender block of Python's
/// `analyze_pil_image`.
#[derive(Debug, Clone)]
pub struct GenderDecision {
    pub female_score: Option<f32>,
    pub male_score: Option<f32>,
    /// "female" / "male" / "unknown".
    pub gender: String,
    /// True when at least one crop voted female - this is what rates 5.
    pub any_female: bool,
}

/// The complete rating decision - the parts of Python's `analyze_pil_image`
/// result the Rust API exposes.
#[derive(Debug, Clone)]
pub struct RatingDecision {
    pub rating: Option<i32>,
    pub decided_by: Option<String>,
    pub nsfw_score: Option<f32>,
    pub female_score: Option<f32>,
    pub male_score: Option<f32>,
    pub gender: String,
}

/// Mirrors the joint-model block of Python's `analyze_pil_image`.
///
/// `class_probs` holds the five class probabilities in the NSFW_CLASSES order
/// (drawings, hentai, neutral, porn, sexy), `binary_score` the calibrated
/// "is this explicit?" signal of the second model.
pub fn decide_joint(
    config: &ScannerConfig,
    class_probs: Option<[f32; 5]>,
    binary_score: Option<f32>,
) -> JointDecision {
    let mut rating: Option<i32> = None;
    let mut decided_by: Option<String> = None;
    let mut nsfw_score: Option<f32> = None;

    // Binary-only setup (mirrors Python analyze_pil_image's
    // `elif nsfw_session is not None:` branch): when the joint model is
    // absent, the calibrated binary model decides on its own thresholds.
    // Without this branch a binary-only install never rates anything:
    // every image comes back "modesty-not-assessed".
    if class_probs.is_none() {
        if let Some(binary) = binary_score {
            nsfw_score = Some(binary);
            if binary > config.threshold_explicit {
                rating = Some(10);
                decided_by = Some("modesty-explicit".to_string());
            } else if binary > config.threshold_revealing {
                rating = Some(7);
                decided_by = Some("modesty-revealing".to_string());
            }
        }
        return JointDecision {
            rating,
            decided_by,
            nsfw_score,
        };
    }

    if let Some(probs) = class_probs {
        nsfw_score = Some(probs[3] + probs[1]); // porn + hentai

        // best_class = max(classes, key=classes.get) - ties keep the
        // earliest class, exactly like Python's max().
        let mut best_index = 0usize;
        for index in 1..probs.len() {
            if probs[index] > probs[best_index] {
                best_index = index;
            }
        }
        let best_class = crate::config::NSFW_CLASSES[best_index];
        let best_prob = probs[best_index];
        let sexy = probs[4];

        match binary_score {
            // No second opinion available (the binary model is missing):
            // fall back to the conservative rule.
            None => {
                let score = nsfw_score.unwrap_or(0.0);
                if score > config.threshold_explicit
                    || ((best_class == "porn" || best_class == "hentai")
                        && score > config.threshold_revealing)
                {
                    rating = Some(10);
                    decided_by = Some(format!("modesty-class-{}", best_class));
                } else if best_class == "sexy" && sexy > config.threshold_revealing {
                    rating = Some(6);
                    decided_by = Some("modesty-class-sexy".to_string());
                } else {
                    rating = None;
                    decided_by = None;
                }
            }
            Some(binary) => {
                if binary > config.threshold_explicit {
                    rating = Some(10);
                    decided_by = Some("modesty-explicit-binary".to_string());
                } else if best_class == "hentai" && best_prob > config.joint_strong {
                    rating = Some(10);
                    decided_by = Some("modesty-class-hentai".to_string());
                } else if best_class == "porn" || best_class == "hentai" {
                    if best_prob > config.joint_strong {
                        if binary > config.threshold_explicit {
                            rating = Some(10);
                            decided_by = Some("modesty-explicit-binary-joint".to_string());
                        } else if binary > config.threshold_middle {
                            rating = Some(7);
                            decided_by = Some("modesty-middle-skin".to_string());
                        } else {
                            rating = Some(6);
                            decided_by = Some("modesty-class-skin".to_string());
                        }
                    } else {
                        rating = None;
                        decided_by = None;
                    }
                } else if best_class == "sexy" && sexy > config.threshold_revealing {
                    if config.sexy_high > 0 && sexy > config.sexy_strong {
                        rating = Some(config.sexy_high);
                        decided_by = Some("modesty-class-sexy-high".to_string());
                    } else {
                        rating = Some(6);
                        decided_by = Some("modesty-class-sexy".to_string());
                    }
                } else {
                    rating = None;
                    decided_by = None;
                }
            }
        }
    }

    JointDecision {
        rating,
        decided_by,
        nsfw_score,
    }
}

/// Mirrors the gender block of Python's `analyze_pil_image`.
///
/// `crop_scores` holds the (female, male) pairs of the four quadrant crops.
pub fn decide_gender(crop_scores: &[(f32, f32)]) -> GenderDecision {
    let (selected_crop, label, any_female) = gender_from_crops(crop_scores);
    let (female_score, male_score) = match selected_crop {
        Some((female, male)) => (Some(female), Some(male)),
        None => (None, None),
    };
    GenderDecision {
        female_score,
        male_score,
        gender: label.to_string(),
        any_female,
    }
}

/// Mirrors the closing block of Python's `analyze_pil_image`: when nothing
/// unsafe was found the image is still rated, just as safe.
pub fn finalize_rating(
    rating: Option<i32>,
    decided_by: Option<String>,
    has_scores: bool,
) -> (Option<i32>, Option<String>) {
    if rating.is_some() {
        return (rating, decided_by);
    }
    if has_scores {
        (Some(1), Some("modesty-no-signal".to_string()))
    } else {
        (None, Some("modesty-not-assessed".to_string()))
    }
}

/// Mirrors the whole decision of Python's `analyze_pil_image`.
///
/// `class_probs` holds the five class probabilities in the NSFW_CLASSES order
/// (drawings, hentai, neutral, porn, sexy), `binary_score` the calibrated
/// "is this explicit?" signal and `crop_scores` the (female, male) pairs of
/// the four quadrant crops.
pub fn decide_rating(
    config: &ScannerConfig,
    class_probs: Option<[f32; 5]>,
    binary_score: Option<f32>,
    crop_scores: &[(f32, f32)],
) -> RatingDecision {
    let joint = decide_joint(config, class_probs, binary_score);
    let mut rating = joint.rating;
    let mut decided_by = joint.decided_by;

    let mut female_score: Option<f32> = None;
    let mut male_score: Option<f32> = None;
    let mut gender_label = "unknown".to_string();

    if rating.is_none() && !crop_scores.is_empty() {
        let gender = decide_gender(crop_scores);
        female_score = gender.female_score;
        male_score = gender.male_score;
        gender_label = gender.gender;
        if gender.any_female {
            rating = Some(5);
            decided_by = Some("modesty-gender-female".to_string());
        }
    }

    let (rating, decided_by) =
        finalize_rating(rating, decided_by, joint.nsfw_score.is_some() || class_probs.is_some());

    RatingDecision {
        rating,
        decided_by,
        nsfw_score: joint.nsfw_score,
        female_score,
        male_score,
        gender: gender_label,
    }
}

/// Scan raw image bytes (RGB) and return ScanResult.
pub fn scan_image_bytes(
    session: &mut ScannerSession,
    width: i32,
    height: i32,
    bytes: &[u8],
) -> Result<ScanResult, Box<dyn std::error::Error>> {
    let img = image::load_from_memory(bytes)?;
    let img = img.resize(width as u32, height as u32, image::imageops::FilterType::Triangle);
    scan_image(session, &img)
}

/// Core scan logic - mirrors Python analyze_pil_image.
pub fn scan_image(session: &mut ScannerSession, img: &DynamicImage) -> Result<ScanResult, Box<dyn std::error::Error>> {
    let config = ScannerConfig::default();
    let rgb = img.to_rgb8();
    let rgb_img = &rgb;

    let mut classes: Option<serde_json::Value> = None;

    // --- classify_pil_image(): the five class probabilities of the joint model.
    let mut class_probs: Option<[f32; 5]> = None;
    #[cfg(feature = "onnx")]
    {
        if let Some(sess) = session.nsfw_session.as_mut() {
            let input_tensor = array4_to_onnx_input(preprocess_nsfw_joint(rgb_img));
            let outputs = sess.run(ort::inputs![input_tensor])?;
            let flat: Vec<f32> = outputs[0].try_extract_array::<f32>()?.iter().copied().collect();
            if flat.len() >= 5 {
                class_probs = Some([flat[0], flat[1], flat[2], flat[3], flat[4]]);
                classes = Some(serde_json::json!({
                    "drawings": flat[0],
                    "hentai": flat[1],
                    "neutral": flat[2],
                    "porn": flat[3],
                    "sexy": flat[4],
                }));
            }
        }
    }

    // --- _run_session(binary_session, preprocess_nsfw(img)): the calibrated
    // "is this explicit?" signal. It runs for every image, not only for the
    // porn/hentai branch.
    let mut binary_score: Option<f32> = None;
    #[cfg(feature = "onnx")]
    {
        if let Some(sess) = session.binary_session.as_mut() {
            let input_tensor = array4_to_onnx_input(preprocess_nsfw(rgb_img));
            let outputs = sess.run(ort::inputs![input_tensor])?;
            let flat: Vec<f32> = outputs[0].try_extract_array::<f32>()?.iter().copied().collect();
            binary_score = extract_positive_score(&flat);
        }
    }

    // --- The joint model's verdict.
    let joint = decide_joint(&config, class_probs, binary_score);
    let mut rating = joint.rating;
    let mut decided_by = joint.decided_by;

    // Gender model on quadrant crops (4-section, not face-crop only).
    // Mirrors the Python `analyze_pil_image` gender block exactly.
    #[allow(unused_assignments)]
    let mut female_score: Option<f32> = None;
    #[allow(unused_assignments)]
    let mut male_score: Option<f32> = None;
    #[allow(unused_assignments)]
    let mut gender_label = "unknown".to_string();

    #[cfg(feature = "onnx")]
    if rating.is_none() && session.gender_session.is_some() {
        // Gender always runs on 4-section quadrants (not face-crop only).
        let mut crop_scores: Vec<(f32, f32)> = Vec::with_capacity(4);

        for crop in crop_halves(rgb_img).iter() {
            let input_tensor = array4_to_onnx_input(preprocess_gender(crop));
            let scores = match session.gender_session.as_mut() {
                Some(sess) => {
                    let outputs = sess.run(ort::inputs![input_tensor])?;
                    let flat: Vec<f32> =
                        outputs[0].try_extract_array::<f32>()?.iter().copied().collect();
                    gender_scores(&session.gender_labels, &flat)
                }
                None => continue,
            };
            let Some((female, male)) = scores else {
                continue;
            };
            crop_scores.push((female, male));
        }

        let gender = decide_gender(&crop_scores);
        female_score = gender.female_score;
        male_score = gender.male_score;
        gender_label = gender.gender;
        if gender.any_female {
            rating = Some(5);
            decided_by = Some("modesty-gender-female".to_string());
        }
    }

    // Nothing unsafe found: the image is still rated, just as safe.
    let (rating, decided_by) = finalize_rating(
        rating,
        decided_by,
        joint.nsfw_score.is_some() || class_probs.is_some(),
    );

    let verdict = rating.map(|r| ScannerConfig::rating_verdict(r).to_string());
    let phrase = rating.map(|r| ScannerConfig::rating_phrase(r).to_string());

    Ok(ScanResult {
        rating,
        verdict,
        phrase,
        nsfw_score: joint.nsfw_score,
        female_score,
        male_score,
        gender: Some(gender_label.to_string()),
        classes,
        decided_by,
    })
}

/// The gender decision over the quadrant crops - mirrors the Python gender
/// block in `analyze_pil_image`.
///
/// The best female crop (highest female score among the crops that vote
/// female) wins over the best male crop. Returns the selected
/// `(female, male)` scores, the gender label and whether a female crop was
/// found (which is what makes the rating 5).
pub fn gender_from_crops(crop_scores: &[(f32, f32)]) -> (Option<(f32, f32)>, &'static str, bool) {
    let mut best_female_crop: Option<(f32, f32)> = None;
    let mut best_male_crop: Option<(f32, f32)> = None;

    for &(female, male) in crop_scores {
        if female > male {
            if best_female_crop.is_none() || female > best_female_crop.unwrap().0 {
                best_female_crop = Some((female, male));
            }
        } else if best_male_crop.is_none() || male > best_male_crop.unwrap().1 {
            best_male_crop = Some((female, male));
        }
    }

    let selected_crop = best_female_crop.or(best_male_crop);
    let label = match selected_crop {
        Some(_) => {
            if best_female_crop.is_some() {
                "female"
            } else {
                "male"
            }
        }
        None => "unknown",
    };

    (selected_crop, label, best_female_crop.is_some())
}

/// Crop an image into four quadrants (mirrors Python crop_halves).
/// Takes &RgbImage (has inherent width()/height()/get_pixel()).
fn crop_halves(img: &RgbImage) -> Vec<RgbImage> {
    let (w, h) = (img.width(), img.height());
    let half_w = w / 2;
    let half_h = h / 2;

    let mut result = Vec::with_capacity(4);
    for (y_start, y_end) in [(0, half_h), (half_h, h)] {
        for (x_start, x_end) in [(0, half_w), (half_w, w)] {
            let mut sub = RgbImage::new(x_end - x_start, y_end - y_start);
            for y in y_start..y_end {
                for x in x_start..x_end {
                    sub.put_pixel(x - x_start, y - y_start, *img.get_pixel(x, y));
                }
            }
            result.push(sub);
        }
    }
    result
}


#[cfg(all(test, feature = "onnx"))]
mod gender_classification_tests {
    use super::*;
    use std::path::PathBuf;

    /// The models that ship with the Python package.
    fn models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("nsfw-lib")
            .join("src")
            .join("nsfw_scanner")
            .join("models")
    }

    /// A sample image from the repository.
    fn sample(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(relative)
    }

    /// Mirrors the Python test
    /// `test_any_female_crop_wins_even_when_another_crop_has_a_higher_male_score`.
    ///
    /// The Python test patches `classify_pil_image` (a neutral 0.95 class
    /// map), `_run_session` (0.01) and `_run_gender_both` (the four crop
    /// scores) and then checks the result of `analyze_pil_image`. Here the
    /// same values are handed to `decide_rating` directly.
    #[test]
    fn test_any_female_crop_wins_even_when_another_crop_has_a_higher_male_score() {
        let config = ScannerConfig::default();
        // class_probs in the NSFW_CLASSES order:
        // drawings, hentai, neutral, porn, sexy
        let class_probs = Some([0.01, 0.01, 0.95, 0.02, 0.01]);
        let binary_score = Some(0.01);
        let crop_scores = [
            (0.65, 0.35),
            (0.10, 0.90),
            (0.20, 0.80),
            (0.25, 0.75),
        ];

        let result = decide_rating(&config, class_probs, binary_score, &crop_scores);

        assert_eq!(result.rating, Some(5));
        assert_eq!(result.gender, "female");
        assert_eq!(result.female_score, Some(0.65));
        assert_eq!(result.male_score, Some(0.35));
    }

    /// Mirrors the Python test
    /// `test_fairface_edge_samples_vote_by_class_order_per_crop`.
    #[test]
    fn test_fairface_edge_samples_vote_by_class_order_per_crop() {
        let model_path = models_dir().join(crate::config::GENDER_MODEL_FAIRFACE);
        if !model_path.exists() {
            eprintln!("skipping: {} is unavailable", model_path.display());
            return;
        }

        let mut session = create_session(models_dir().to_str().unwrap()).expect("session");
        assert_eq!(session.gender_labels, ["female", "male"]);

        let samples: [(&str, bool); 3] = [
            ("SAMPLES/real_live_data/111.png", false),
            ("SAMPLES/real_live_data/222.png", true),
            ("SAMPLES/_ANDROID_SAMPLE/nastypeople.jpeg", true),
        ];

        for (relative, expected_female) in samples {
            let path = sample(relative);
            let img = image::open(&path).unwrap().to_rgb8();

            let mut votes: Vec<bool> = Vec::with_capacity(4);
            for crop in crop_halves(&img) {
                let input_tensor = array4_to_onnx_input(preprocess_gender(&crop));
                let (female, male) = {
                    let sess = session.gender_session.as_mut().unwrap();
                    let outputs = sess.run(ort::inputs![input_tensor]).unwrap();
                    let flat: Vec<f32> = outputs[0]
                        .try_extract_array::<f32>()
                        .unwrap()
                        .iter()
                        .copied()
                        .collect();
                    gender_scores(&session.gender_labels, &flat).unwrap()
                };
                votes.push(female > male);
            }

            assert_eq!(
                votes.iter().any(|v| *v),
                expected_female,
                "sample {} voted wrong",
                relative
            );
        }
    }

    /// The Rust port must reach the same verdict as the Python scanner on
    /// the sample images. The reference values come from
    /// `eval/print_python_verdicts.py`.
    #[test]
    fn test_sample_images_match_the_python_scanner() {
        let joint_model = models_dir().join(crate::config::NSFW_JOINT_MODEL);
        let binary_model = models_dir().join(crate::config::NSFW_BINARY_MODEL);
        if !joint_model.exists() || !binary_model.exists() {
            eprintln!("skipping: the NSFW models are unavailable");
            return;
        }

        // (sample, rating, decided_by, gender, female_score, male_score)
        let samples: [(&str, Option<i32>, &str, &str, f32, f32); 3] = [
            (
                "SAMPLES/real_live_data/111.png",
                Some(1),
                "modesty-no-signal",
                "male",
                0.008248,
                0.991752,
            ),
            (
                "SAMPLES/real_live_data/222.png",
                Some(5),
                "modesty-gender-female",
                "female",
                0.710024,
                0.289976,
            ),
            (
                "SAMPLES/_ANDROID_SAMPLE/nastypeople.jpeg",
                Some(5),
                "modesty-gender-female",
                "female",
                0.511636,
                0.488364,
            ),
        ];

        let mut session = create_session(models_dir().to_str().unwrap()).expect("session");

        for (relative, rating, decided_by, gender, female, male) in samples {
            let path = sample(relative);
            let result = scan_image_file(&mut session, path.to_str().unwrap()).expect("scan");

            assert_eq!(result.rating, rating, "sample {} rated differently", relative);
            assert_eq!(
                result.decided_by.as_deref(),
                Some(decided_by),
                "sample {} decided differently",
                relative
            );
            assert_eq!(
                result.gender.as_deref(),
                Some(gender),
                "sample {} gendered differently",
                relative
            );
            // The resize filter's support bounds are rounded slightly
            // differently than PIL's and the ONNX Runtime version differs
            // (Rust uses 1.31, Python 1.23), so the scores are compared with
            // a small tolerance. The rating, the decision and the gender
            // label must match exactly.
            let tolerance = 0.06;
            let female_score = result.female_score.expect("female score");
            let male_score = result.male_score.expect("male score");
            assert!(
                (female_score - female).abs() < tolerance,
                "sample {} female score {} != {}",
                relative,
                female_score,
                female
            );
            assert!(
                (male_score - male).abs() < tolerance,
                "sample {} male score {} != {}",
                relative,
                male_score,
                male
            );
        }
    }
}

#[cfg(test)]
mod binary_only_tests {
    use super::*;

    fn config() -> ScannerConfig {
        ScannerConfig {
            threshold_explicit: 0.5,
            threshold_revealing: 0.2,
            ..Default::default()
        }
    }

    /// Mirrors Python analyze_pil_image's `elif nsfw_session is not None:`
    /// branch: the binary model decides on its own when the joint model is
    /// absent (class_probs is None).
    #[test]
    fn binary_only_rates_by_threshold() {
        let cfg = config();
        let explicit = decide_joint(&cfg, None, Some(0.9));
        assert_eq!(explicit.rating, Some(10));
        assert_eq!(explicit.decided_by.as_deref(), Some("nsfw-explicit"));
        assert_eq!(explicit.nsfw_score, Some(0.9));

        let revealing = decide_joint(&cfg, None, Some(0.3));
        assert_eq!(revealing.rating, Some(7));
        assert_eq!(revealing.decided_by.as_deref(), Some("nsfw-revealing"));

        // Below both thresholds: no rating yet, but nsfw_score is set so
        // finalize_rating produces rating 1 "no-unsafe-signal" (Python line 1261).
        let low = decide_joint(&cfg, None, Some(0.1));
        assert_eq!(low.rating, None);
        assert_eq!(low.decided_by, None);
        assert_eq!(low.nsfw_score, Some(0.1));
        let (rating, decided_by) = finalize_rating(low.rating, low.decided_by, low.nsfw_score.is_some());
        assert_eq!(rating, Some(1));
        assert_eq!(decided_by.as_deref(), Some("no-unsafe-signal"));

        // No binary model at all: still not assessed.
        let none = decide_joint(&cfg, None, None);
        assert_eq!(none.rating, None);
        assert_eq!(none.nsfw_score, None);
    }
}
