# EcoSense — Wildlife Audio Monitoring System

A biodiversity monitoring system that listens to forests so researchers don't have to.

***

## What's the Problem?

1. Wildlife monitoring is broken. Traditional surveys are expensive, happen once in a while, and cover tiny areas.
2. Camera traps miss insects, frogs, and anything that moves at night. 
3. Satellite images show trees — not the animals living in them. And there's no system that monitors ecosystems                 continuously, at scale, without disturbing the habitat.

We built EcoSense to fix that.

***

## What We Built

EcoSense uses small microphones attached to solar-powered IoT devices deployed in forests, wetlands, or any habitat. 
These devices run 24/7, record sounds, classify animals by their calls using an on-device ML model (TensorFlow Lite), and sync the data to a server when connected.

No human needs to be there. No animal gets disturbed. It just listens.

***

## How It Works

**Hardware Layer**
- IoT audio units with microphones
- Solar powered, low energy consumption
- Continuous 24/7 recording
- Data stored locally on device (secure, no data loss if offline)

**Intelligence Layer**
- TensorFlow Lite model runs directly on the device
- Classifies bats, frogs, birds, insects, and more from their sounds
- Works fully offline — syncs results when internet is available

**Dashboard Layer**
- Live species maps
- Population trend charts
- Ecosystem health scores
- Alerts and long-term data analysis

***

## Tech Stack

- TensorFlow Lite (on-device inference)
- IoT microcontrollers (ESP32 / Raspberry Pi)
-  Librosa (Audio Processing)
- Python (backend + data processing)
- React (dashboard frontend)
- Keras (Model API)
- FastAPI (REST API Server)
- BirtNet (Pre-trained model)

  
***

## System Architecture

The system is divided into five layers that work together:

**1. IoT Device (Field Unit)**
Hardware placed in the environment — includes a microphone and a small
processing unit that captures audio continuously from the surroundings.

**2. Data Processing Layer**
Captured audio is cleaned and prepared for analysis. This step filters out
unnecessary noise and makes the data ready for the ML model.

**3. Machine Learning Model**
It trained the model to identifies the species

***4. Storage System***
All processed results and raw data are stored for future use, long-term
tracking, and historical comparison.

**5. Visualization / Dashboard**
A simple interface where users can view species presence, detection
frequency, and ecosystem trends over time.

 ## Usage

Once the system is set up, it runs on its own with minimal input needed.

- The IoT device automatically starts capturing audio from the surrounding environment
- Audio is processed and species are identified in the background without manual input
- Results can be viewed through the dashboard , no need to go anywhere
- Data collected over time can be used to study which species are active and when

The system is designed to run continuously — no constant supervision required.

***

## Add Demo Screenshot and Video
**how its interface look**

**Image_1**

![image alt](https://github.com/WTC-Group-4/wtc-round-2-group-4-modelminds/blob/f5c19dcc18dde334e67d8a583770cb617eba7d48/screeshot_1.png)

**Image_2**

![image alt](https://github.com/WTC-Group-4/wtc-round-2-group-4-modelminds/blob/f5c19dcc18dde334e67d8a583770cb617eba7d48/screeshot_2.png)

**Image_3**

![image alt](https://github.com/WTC-Group-4/wtc-round-2-group-4-modelminds/blob/f5c19dcc18dde334e67d8a583770cb617eba7d48/Screenshot_3.png)

**DEMO VIDEO**
https://github.com/user-attachments/assets/bb23407a-807f-4fa3-aa5d-48b9aeb5327b


## Future Improvements


**there are some feature which we would like to improve in future for betterment:**

- Expand the ML model to recognize a wider range of species
- Improve accuracy in noisy environments (heavy rain, wind, etc.)
- Add real-time alerts for rare species detection
- Shift to full edge computing so devices work without internet
- Make hardware more energy efficient using solar power
- Integrate with official wildlife and conservation databases for broader impact


  
  








