## Rescources

- [2023 Istanbul ITUNOM Design](https://www.youtube.com/watch?v=hlhPw4j1K4E&list=PLelb3ZzP70dQa_RKuvNC5fHli9hd39_ku)
- [2025 Teams](https://suas-competition.org/teams)
- 2025 Rules
  - [Mapping](https://robonation.gitbook.io/suas-resources/section-3-mission-demonstration/3.5-mapping)
  - [Object Detection and Air drop](https://robonation.gitbook.io/suas-resources/section-3-mission-demonstration/3.6-object-detection-and-air-drop)

# Problem - Object Detection & Air Drop

- Detect, then classify targets.
  - (Person/Mannequin, Car (>1:8 Scale Model), Motorcycle (>1:8 Scale Model), Airplane (>3m Wing Span Scale Model), Bus (>1:8 Scale Model), Boat (>1:8 Scale Model), Stop Sign (Flat, Upwards Facing), Snowboard, Umbrella, Sports Ball (Regulation Size Soccer Ball, Basketball, Volleyball, or Football), Baseball Bat, Bed/Mattress (> Twin Size), Tennis Racket, Suitcase, Skis)
- Minimum 75ft altitude
- Air Drop Payload Delivered to Unique Object
- Air Drop Payload Lands within 25' of Object
-

# Solution

1. Supervised learning method, likely with YOLO
2. Organize a dataset with relevant objects

# Datasets

[Our Combined Dataset from 2025 SUAS](https://app.roboflow.com/suas-gyf9o/2025_suas-fkngn/1)

- [DOTADataset](https://www.kaggle.com/datasets/chandlertimm/dota-data)
  ![alt text](DOTADataset_examples.png)
- [VisDrone Dataset](https://docs.ultralytics.com/datasets/detect/visdrone/)
- [Roboflow](https://universe.roboflow.com/)

# Tools

- [Format Converter](https://github.com/ISSResearch/Dataset-Converters)
- [Upload dataset](https://docs.roboflow.com/developer/command-line-interface/upload-a-dataset)
- [Supervision](https://github.com/roboflow/supervision)
- [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX)
- [sahi](https://github.com/obss/sahi)
- [fiftyone](https://github.com/voxel51/fiftyone)

# Hardware

- [SIYI A8](https://shop.siyi.biz/products/siyi-a8-mini-gimbal-camera)
- [Jetson Orion Nano Super](https://docs.rs-online.com/4051/A700000009607470.pdf)
