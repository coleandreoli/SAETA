# SEATA Autonomous Drone Competition - Object Detection & Air Drop - Detection with YOLO

## Overview

This repository contains the object detection and air drop solution for the SUAS (Small Unmanned Aerial Systems) Competition organized by RoboNation. The project focuses on developing computer vision systems to detect targets from aerial imagery and autonomously deliver payloads to specific objects.

### Mission Objectives

The competition requires autonomous drones to:

- **Search and Detect**: Identify specific objects (mannequins, tents) from aerial views at altitudes between 150-400ft AGL
- **Classify Targets**: Accurately classify detected objects into predefined categories
- **Deliver Payloads**: Execute precision air drops to deliver items (water bottles to mannequins, strobing beacons to tents) within specified distance thresholds
- **Navigate Autonomously**: Operate within defined search boundaries and altitude restrictions

## 2026 Competition

[Official link](https://robonation.gitbook.io/suas-resources/2026-team-handbook)

- 3.6 Search, Detect, and Deliver
    - One mannequin and one tent will be scattered amongst other debris around the Search Boundary. Teams must detect the mannequin to deliver the water bottle, and detect the tent to deliver the strobing beacon. The mannequin may be positioned in any orientation (lying down, sitting up, face-down, etc.) and will be surrounded/covered by surrounding features such as bushes/trees/vehicles/etc.
    - Object Survives (Within Vicinity of Search Boundary) = 20 Points
    - Object Lands within 50' of a Target = 50 Points
    - Object Delivered to the Correct Target (Water Bottle to Mannequin + Beacon to Tent) = 30 Points
    - The UAS must remain within this polygon and the altitude restrictions of [150ft AGL, 400ft AGL]
- 3.5 Risk Mapping (TODO)

## 2025 Competition

- Detect, then classify targets.
    - (Person/Mannequin, Car (>1:8 Scale Model), Motorcycle (>1:8 Scale Model), Airplane (>3m Wing Span Scale Model), Bus (>1:8 Scale Model), Boat (>1:8 Scale Model), Stop Sign (Flat, Upwards Facing), Snowboard, Umbrella, Sports Ball (Regulation Size Soccer Ball, Basketball, Volleyball, or Football), Baseball Bat, Bed/Mattress (> Twin Size), Tennis Racket, Suitcase, Skis)
- Minimum 75ft altitude
- Air Drop Payload Delivered to Unique Object
- Air Drop Payload Lands within 25' of Object

# Constraints info from discord Q&A

[Discord Q&A section](https://discordapp.com/channels/969707043218935848/1120835450928627862)

- Q: Can Search/Detect CV processing be done on a ground station, or must it be onboard the UAV?
    - A: You can do any processing (outside of safety critical failsafes) you want on the ground or onboard - up to the teams.
- Q: If a mannequin is present, its orientation will not be given to teams ahead of time.
    - A: Standard adult-sized white mannequin wearing clothing that will be a surprise
- Q: Hi, my team was hoping to get some clarification on a tent that is being used for the beacon drop. I imagine it is very early on and likely not fully decided yet, but if there were some more clarity on whether it is a camping tent, EZ-Up, or any other specs we could get, my team would greatly appreciate it.
    - A: No update :(

# Solution

1. Supervised learning method, likely with YOLO or RF-DETR
2. Organize a dataset with relevant objects

# Datasets

- [Google Drive](https://drive.google.com/drive/u/3/folders/1dSfketzhvdU6ND2Lg0wPZzmkcD2sxP5k)
- [2026-suas-v6](https://app.roboflow.com/suas-gyf9o/2026-suas-v6/1)
- [SAETA Roboflow Dataset Collection](https://app.roboflow.com/suas-gyf9o)
- [Our Combined Dataset from 2025 SUAS](https://app.roboflow.com/suas-gyf9o/2025_suas-fkngn/1)
- [FiftyOne Dataset - VisDrone2019 - DET](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET)
- [FiftyOne Dataset - VisDrone - mot](https://huggingface.co/datasets/Voxel51/visdrone-mot)
- [VisDrone Dataset](https://docs.ultralytics.com/datasets/detect/visdrone/)
- [DOTADataset](https://www.kaggle.com/datasets/chandlertimm/dota-data)
- [Roboflow Universe](https://universe.roboflow.com/)

# Tools

- [fiftyone](https://github.com/voxel51/fiftyone)
- [Format Converter](https://github.com/ISSResearch/Dataset-Converters)
- [Roboflow Upload dataset](https://docs.roboflow.com/developer/command-line-interface/upload-a-dataset)
- [Supervision](https://github.com/roboflow/supervision)
- [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX)
- [sahi](https://github.com/obss/sahi)

# Hardware

- [SIYI A8 - Camera](https://shop.siyi.biz/products/siyi-a8-mini-gimbal-camera)
- [Jetson Orion Nano Super](https://docs.rs-online.com/4051/A700000009607470.pdf)

## Rescources

- [2023 Istanbul ITUNOM Design](https://www.youtube.com/watch?v=hlhPw4j1K4E&list=PLelb3ZzP70dQa_RKuvNC5fHli9hd39_ku)
- [2025 Teams](https://suas-competition.org/teams)
- 2025 Rules
    - [Mapping](https://robonation.gitbook.io/suas-resources/section-3-mission-demonstration/3.5-mapping)
    - [Object Detection and Air drop](https://robonation.gitbook.io/suas-resources/section-3-mission-demonstration/3.6-object-detection-and-air-drop)
