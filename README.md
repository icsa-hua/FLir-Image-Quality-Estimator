# FLIR Image Quality Estimator (FLIQE)
A tool for evaluating and analyzing image quality from FLIR thermal cameras.

![Distorted Frames with Scores](pics/distorted_frames_with_scores.png)

Download pre-trained models:
- [FLIQE Encoder](https://github.com/icsa-hua/FLir-Image-Quality-Estimator/releases/download/uploading_models/resnet50_128_out.pth)
- [FLIQE Encoder + Binary Head](https://github.com/icsa-hua/FLir-Image-Quality-Estimator/releases/download/uploading_models/binary_head.pth)

## How to use
FLIQE implements two classes for image quality estimation:

- **FLIQE class**: takes an image and outputs a quality score.
  ```python
  from flir_iqa import FLIQE

  fliqe = FLIQE(
      quality_model_path='models/encoder_with_binary_head.pth'
  )
  quality_score = fliqe.estimate_image_quality(image)
  print("Image quality score:", quality_score)
  ```

- **OnlineFLIQE class**: allows creating a session and computes a smoothed quality score over a sequence of images.
  ```python
  from flir_iqa import OnlineFLIQE

  online_fliqe = OnlineFLIQE(
      quality_model_path='models/encoder_with_binary_head.pth',
      smoothing_window_size=300
  )
  online_fliqe.create_session('testing_video')
  quality_score = online_fliqe.estimate_smoothed_quality(image, session_id='testing_video')
  print("Smoothed image quality score:", quality_score)
  ```

## How it works
FLIQE employs a sophisticated machine learning approach to assess the quality of thermal images from FLIR cameras. The system works through several key stages:

### 1. Feature Extraction with Pre-trained Networks
The system leverages powerful pre-trained computer vision model (ResNet50) as feature extractor. This network, originally trained on large-scale image datasets, provide robust visual representations that capture important structural and textural patterns in thermal images.

### 2. FLIQE Encoder With Supervised Contrastive Learning
The FLIQE Encoder was trained using supervised contrastive learning on the FLIR Thermal Images Dataset, learning an embedding space where images with similar quality characteristics are positioned close together, while those with differing quality issues are separated. To effectively train the quality assessment system, FLIQE applies comprehensive distortion simulation, including:
- **Optical distortions**: Lens blur and motion blur that simulate camera movement or focus issues
- **Environmental artifacts**: Gaussian noise representing sensor limitations and thermal interference
- **Exposure problems**: Overexposure and underexposure conditions that affect thermal sensitivity
- **Compression artifacts**: Quality degradation from image compression during storage or transmission
- **Thermal-specific issues**: Ghosting effects and aliasing that are particularly relevant to thermal imaging \
One t-SNE representation of the learned embedding space is shown below:
![t-SNE Visualization of Distorted Images](pics/tsne_distorted_images.png)

### 3. FLIQE Binary Head 
The FLIQE Binary Head is a lightweight MLP classifier that takes the embeddings produced by the FLIQE Encoder and predicts the quality level of input thermal images. It is trained on a private dataset of FLIR thermal images annotated as either distorted (1) or not (0), using the simulated distortions described above. The performance of the FLIQE Binary Head is summarized in the table below:

| Metric | Score |
|--------|-------|
| Accuracy | 0.9275 |
| Precision | 0.9423 |
| Recall | 0.8917 |
| F1-score | 0.9163 |