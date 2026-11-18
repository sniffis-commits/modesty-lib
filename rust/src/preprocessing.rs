//! Image preprocessing - mirrors Python nsfw_scanner exactly.
//!
//! Pipelines:
//!   - preprocess_nsfw: 256px resize, 224px center crop, BGR-mean, NHWC
//!     (Yahoo open_nsfw)
//!   - preprocess_gender: 224px resize, /255, mean/std 0.5, NCHW (FairFace ViT)
//!   - preprocess_nsfw_joint: 299px resize, values in [0,1], NHWC
//!   - preprocess_nsfw_student: 224px, ImageNet norm, NCHW

use image::{imageops::FilterType, RgbImage};
use ndarray::Array4;
use crate::config::{JOINT_INPUT_SIZE, MEAN_BGR, STUDENT_INPUT_SIZE, STUDENT_MEAN, STUDENT_STD};

/// Preprocess for the binary/224 model (Yahoo open_nsfw pipeline).
/// Returns NHWC float32 batch of shape (1, 224, 224, 3) in BGR order
/// with mean [104, 117, 123] subtracted.
pub fn preprocess_nsfw(img: &RgbImage) -> Array4<f32> {
    let img = image::imageops::resize(img, 256, 256, FilterType::Triangle);
    let cropped = image::imageops::crop_imm(&img, 16, 16, 224, 224).to_image();
    let (w, h) = (cropped.width(), cropped.height());

    // NHWC row-major: row, then column, then channel (BGR).
    let mut data = vec![0.0f32; h as usize * w as usize * 3];
    for y in 0..h {
        for x in 0..w {
            let pixel = cropped.get_pixel(x, y);
            let offset = (y as usize * w as usize + x as usize) * 3;
            data[offset] = pixel[2] as f32 - MEAN_BGR[0]; // B
            data[offset + 1] = pixel[1] as f32 - MEAN_BGR[1]; // G
            data[offset + 2] = pixel[0] as f32 - MEAN_BGR[2]; // R
        }
    }
    Array4::from_shape_vec((1, h as usize, w as usize, 3), data).unwrap()
}

/// Preprocess for gender model (224px, RGB, /255, mean/std 0.5, NCHW).
///
/// FairFace ViT pipeline: resize to 224px, divide by 255, then
/// (x - 0.5) / 0.5 per channel, NCHW layout.
///
/// --- OLD: Levi/Adience gender pipeline (commented out, keep for reference)
///     let img = image::imageops::resize(img, 224, 224, FilterType::Triangle);
///     let (w, h) = (img.width(), img.height());
///     let mut data = Vec::with_capacity(w as usize * h as usize * 3);
///     for y in 0..h {
///         for x in 0..w {
///             let pixel = img.get_pixel(x, y);
///             data.push(pixel[0] as f32 - MEAN_BGR[0]); // R
///             data.push(pixel[1] as f32 - MEAN_BGR[1]); // G
///             data.push(pixel[2] as f32 - MEAN_BGR[2]); // B
///         }
///     }
/// ---
pub fn preprocess_gender(img: &RgbImage) -> Array4<f32> {
    let img = image::imageops::resize(img, 224, 224, FilterType::Triangle);
    let (w, h) = (img.width(), img.height());

    // NCHW row-major: channel, then row, then column.
    let mut data = vec![0.0f32; 3 * h as usize * w as usize];
    for channel in 0..3 {
        let plane = &mut data[channel * h as usize * w as usize..];
        for y in 0..h {
            for x in 0..w {
                let pixel = img.get_pixel(x, y);
                let value = pixel[channel] as f32 / 255.0;
                plane[y as usize * w as usize + x as usize] = (value - 0.5) / 0.5;
            }
        }
    }
    Array4::from_shape_vec((1, 3, h as usize, w as usize), data).unwrap()
}

/// Preprocess for joint model (299x299, values in [0,1], NHWC).
pub fn preprocess_nsfw_joint(img: &RgbImage) -> Array4<f32> {
    let img = image::imageops::resize(img, JOINT_INPUT_SIZE.0, JOINT_INPUT_SIZE.1, FilterType::Triangle);
    let (w, h) = (img.width(), img.height());

    let mut data = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }
    }
    Array4::from_shape_vec((1, h as usize, w as usize, 3), data).unwrap()
}

/// Preprocess for student model (224x224, ImageNet norm, NCHW).
pub fn preprocess_nsfw_student(img: &RgbImage) -> Array4<f32> {
    let img = image::imageops::resize(img, STUDENT_INPUT_SIZE.0, STUDENT_INPUT_SIZE.1, FilterType::Triangle);
    let (w, h) = (img.width(), img.height());

    // NCHW row-major: channel, then row, then column.
    let mut data = vec![0.0f32; 3 * h as usize * w as usize];
    for channel in 0..3 {
        let plane = &mut data[channel * h as usize * w as usize..];
        for y in 0..h {
            for x in 0..w {
                let pixel = img.get_pixel(x, y);
                let value = pixel[channel] as f32 / 255.0;
                plane[y as usize * w as usize + x as usize] =
                    (value - STUDENT_MEAN[channel]) / STUDENT_STD[channel];
            }
        }
    }
    Array4::from_shape_vec((1, 3, h as usize, w as usize), data).unwrap()
}

/// Convert an Array4 (NCHW or NHWC) to the ONNX input tensor.
/// Only available when the `onnx` feature is enabled.
#[cfg(feature = "onnx")]
pub fn array4_to_onnx_input(arr: Array4<f32>) -> ort::value::Tensor<f32> {
    let shape: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
    let data: Vec<f32> = arr.into_raw_vec();
    ort::value::Tensor::from_array((shape, data)).expect("failed to build the ONNX tensor")
}

/// Stub when ONNX feature is disabled.
#[cfg(not(feature = "onnx"))]
pub fn array4_to_onnx_input(_arr: Array4<f32>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_shapes() {
        let img = RgbImage::new(300, 200);
        // The binary model is NHWC (1, 224, 224, 3) - see Python's
        // `preprocess_nsfw`.
        assert_eq!(preprocess_nsfw(&img).shape(), &[1, 224, 224, 3]);
        // The joint model is NHWC (1, 299, 299, 3).
        assert_eq!(preprocess_nsfw_joint(&img).shape(), &[1, 299, 299, 3]);
        // The gender model is NCHW (1, 3, 224, 224).
        assert_eq!(preprocess_gender(&img).shape(), &[1, 3, 224, 224]);
        // The student model is NCHW (1, 3, 224, 224).
        assert_eq!(preprocess_nsfw_student(&img).shape(), &[1, 3, 224, 224]);
    }

    /// The NCHW pipelines must lay the channels out plane by plane, the NHWC
    /// ones interleaved - exactly like numpy's transpose/reshape does.
    #[test]
    fn test_preprocess_layouts() {
        // A uniform image: every pixel is R=10, G=20, B=30. After the
        // resize every pixel keeps those values, so the channel order is
        // directly readable from the tensor.
        let mut img = RgbImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([10, 20, 30]);
        }

        // Gender: NCHW, so the three channels of the top-left pixel live in
        // three separate planes.
        let gender = preprocess_gender(&img);
        assert_eq!(gender.shape(), &[1, 3, 224, 224]);
        let gender_flat: Vec<f32> = gender.iter().copied().collect();
        let plane = 224 * 224;
        let red = gender_flat[0];
        let green = gender_flat[plane];
        let blue = gender_flat[2 * plane];
        assert!(red < green && green < blue, "NCHW channels are out of order");

        // Joint: NHWC, so the three channels of the top-left pixel are
        // adjacent.
        let joint = preprocess_nsfw_joint(&img);
        assert_eq!(joint.shape(), &[1, 299, 299, 3]);
        let joint_flat: Vec<f32> = joint.iter().copied().collect();
        let red = joint_flat[0];
        let green = joint_flat[1];
        let blue = joint_flat[2];
        assert!(red < green && green < blue, "NHWC channels are out of order");
    }
}
