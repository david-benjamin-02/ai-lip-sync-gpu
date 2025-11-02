
---

````md
# AI Lip-Sync GPU

A multilingual AI-powered lip-syncing application that translates, generates speech, and synchronizes lips in real time using Deep Learning (Wav2Lip, Whisper, Transformers) and Google Cloud APIs.

---

## Features
- Automatic speech recognition with OpenAI Whisper
- Real-time translation to target languages
- Lip-syncing using Wav2Lip GAN
- Text-to-speech conversion for translated output
- Google Cloud API integration for translation and speech synthesis
- GPU-accelerated pipeline for faster video generation

---

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-API-blue)

---

## Project Structure
```bash
ai-lipsync-gpu/
│
├── app.py                  # Main Gradio web app
├── requirements.txt        # Main environment requirements
│
├── Wav2Lip/                # Lip-sync model directory
│   ├── inference.py
│   ├── audio.py
│   ├── requirements.txt    # Wav2Lip dependencies
│   ├── checkpoints/
│   │   └── wav2lip_gan.pth
│   └── venv_wav/           # Wav2Lip virtual environment
│
└── myenv/                  # Main project virtual environment
````

---

## Full Installation & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/david-benjamin-02/ai-lipsync-gpu.git
cd ai-lipsync-gpu
```

### 2. Create and Activate Main Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate   # On Windows
```

### 3. Install Main Requirements

Create a file named `requirements.txt` with the following:

```txt
gradio==5.1.0
torch==2.2.2
torchaudio==2.2.2
transformers==4.39.3
openai-whisper==20231117
pydub==0.25.1
soundfile==0.12.1
librosa==0.9.2
opencv-python==4.9.0.80
numpy==1.23.5
ffmpeg-python==0.2.0
tqdm==4.66.1
gTTS==2.3.2
sentencepiece==0.1.99
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Verify CUDA & GPU Setup

```bash
python
>>> import torch
>>> print(torch.cuda.is_available())
>>> print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

Expected output:

```bash
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

If CUDA is not available, reinstall CUDA-compatible PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Setup the Wav2Lip Environment

```bash
cd Wav2Lip
python -m venv venv_wav
venv_wav\Scripts\activate
```

Create `Wav2Lip/requirements.txt`:

```txt
torch==2.2.2
torchvision==0.17.2
numpy==1.23.5
opencv-python==4.9.0.80
dlib==19.24.2
ffmpeg-python==0.2.0
librosa==0.9.2
tqdm==4.66.1
numba==0.56.4
```

Then install:

```bash
pip install -r requirements.txt
```

### 6. Download Pretrained Model

```bash
mkdir checkpoints
```

Download [wav2lip_gan.pth (Google Drive)](https://drive.google.com/file/d/1BXd6BC5_Jg_pAke9U74FY-jBmMUIHZpY/view?usp=drive_link)
and place it in:

```
Wav2Lip/checkpoints/wav2lip_gan.pth
```

### 7. Run Wav2Lip Test (Optional)

```bash
cd Wav2Lip
venv_wav\Scripts\activate
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face "sample.mp4" \
  --audio "sample.wav" \
  --outfile "result.mp4"
```

### 8. Run the Full Project

```bash
cd ..
myenv\Scripts\activate
python app.py
```

Expected output:

```
Using device: CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)
* Running on local URL: http://127.0.0.1:7860
```

### 9. Open the Web App

Open in your browser:

```
http://127.0.0.1:7860
```

Then:

1. Upload a video
2. Select the target language
3. Wait for processing (ASR → Translation → TTS → Lip-Sync)
4. Download the final video

---

## Troubleshooting

| Issue                                           | Solution                                                        |
| ----------------------------------------------- | --------------------------------------------------------------- |
| `ModuleNotFoundError: cv2`                      | Activate the correct environment and install `opencv-python`    |
| `TypeError: mel() takes 0 positional arguments` | Use `librosa==0.9.2`                                            |
| Processing too long                             | Ensure GPU is available (`torch.cuda.is_available()` → True)    |
| Audio & video out of sync                       | Match input and generated audio lengths before the Wav2Lip step |
| `np.complex` error                              | Downgrade NumPy to `1.23.5`                                     |

---

## Demo Files

* Demo Video: `<video controls src="https://github.com/david-benjamin-02/ai-lip-sync-gpu/blob/main/video-accessiblity_project_demo.mp4" title="Title"></video>`
* Input Video: `<video controls src="https://github.com/david-benjamin-02/ai-lip-sync-gpu/blob/main/spanish.mp4" title="Title"></video>`
* Output Video: `<video controls src="https://github.com/david-benjamin-02/ai-lip-sync-gpu/blob/main/final_output.mp4" title="Title"></video>`

---

© 2025 David Benjamin B — All Rights Reserved

```

---
