# Synthetic Dataset Report

## Dataset Overview

This synthetic dataset was created to support the development of a Space Debris Detection and Tracking System.

The dataset simulates optical observations of space debris using computer-generated imagery and automatically generated YOLO annotations.

## Dataset Statistics

* Total Images: 100
* Image Resolution: 640 × 640
* Classes: 1 (Debris)
* Annotation Format: YOLO
* Objects Per Image: 1–6

## Object Types

### Circular Debris

Small bright objects represented as blurred circular blobs.

### Streak Debris

Linear bright objects representing moving debris during camera exposure.

## Background Characteristics

* Black space background
* Random star field
* Sensor noise simulation
* Variable object brightness

## Assumptions

* Debris appears brighter than background stars.
* Debris can be approximated as dots or streaks.
* Synthetic observations approximate optical telescope imagery.
* All debris belongs to a single detection class.

## Limitations

* Does not model realistic orbital dynamics.
* Does not model atmospheric effects.
* Does not include spacecraft attitude changes.
* Does not include complex optical distortions.

## Future Improvements

* Motion-based video generation
* Real orbital trajectory simulation
* Multiple debris classes
* Integration with real space-object datasets
* Domain adaptation using real imagery
