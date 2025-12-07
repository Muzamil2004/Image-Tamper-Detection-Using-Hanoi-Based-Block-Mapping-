Image Tamper Detection and Recovery Using Hanoi-Based Block Mapping
Overview

This project implements a robust image tamper detection and recovery system using Hanoi-based block mapping and block-wise watermarking. The technique divides an input image into small blocks, embeds watermark bits derived from the block’s frequency content (via DCT), and maps the blocks using a sequence inspired by the Tower of Hanoi algorithm. Any unauthorized modifications disrupt this block mapping, allowing precise detection and localization of tampered regions. The system also supports tamper recovery using either the original image or neighboring block information combined with iterative diffusion, making it highly effective in restoring tampered areas.

Key Features

Hanoi-Based Block Mapping: Ensures predictable block arrangement for tamper detection.

Watermarking Using DCT: Embeds robust 8-bit watermarks in 2x2 blocks using the block’s DC coefficient.

Tamper Detection: Multi-step detection process including watermark parity checks, variance-based detection, and connectivity filtering.

Tamper Recovery:

Original-based recovery: If the original image is available.

Neighbor-based recovery: Weighted averaging of neighboring untampered blocks.

Diffusion-based smoothing: Iterative diffusion to blend tampered regions seamlessly.

No External Toolbox Required: Fully implemented using custom DCT, IDCT, and binary conversion functions.

Visualization: Highlights tampered regions and overlays them on the original image for easy analysis.

Requirements

MATLAB R2018a or later (compatible with standard MATLAB without additional toolboxes).

Images in grayscale format. Recommended size: 256×256 pixels for optimal performance.

Usage

Clone the repository:

git clone https://github.com/Muzamil2004/Image-Tamper-Detection-Using-Hanoi-Based-Block-Mapping-
cd Image-Tamper-Detection


Place your input image in the repository folder and update the filename in the script:

a1 = imread('your_image.png');


Set the recovery mode:

recovery_mode = 1; % Use original image for recovery  
recovery_mode = 2; % Use neighbor-based recovery


Run the tamper_detection_recovery.m script in MATLAB.

Outputs

image1.png – Pre-processed image with cleared LSBs.

image2.png – Watermarked image.

image4.png – Preliminary tamper flags (Step 1).

image5.png – Visualized Step 1 detection.

image6.png / image6b.png – Step 2 detection flags and visualization.

image7.png / image7b.png – Step 3 refined detection.

image8.png – Final tamper detection flags.

tamper_map.png – Pixel-level tampered region map.

tampered_highlighted.png – Tampered regions highlighted on the received image.

image9.png – Recovered image after tamper restoration.

Algorithm Summary

Divide image into 2x2 blocks.

Generate watermark from block DCT coefficients.

Embed watermark in the 2 LSBs of each block.

Shuffle blocks using Hanoi mapping for tamper detection robustness.

Detect tampered blocks using watermark parity checks and variance analysis.

Refine detection via connectivity and neighborhood filtering.

Recover tampered blocks using original image or neighbor-weighted averaging.

Apply iterative diffusion to smooth recovered regions.

Applications

Digital forensics and legal image verification.

Medical imaging security.

Secure image transmission and storage.

Academic research in watermarking and tamper detection.

Future Improvements

Extend to color images.

Optimize for real-time processing.

Integrate deep learning-based detection for enhanced robustness.

License

This project is licensed under the MIT License – see the LICENSE
 file for details.
