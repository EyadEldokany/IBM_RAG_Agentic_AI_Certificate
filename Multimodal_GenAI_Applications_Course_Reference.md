# Multimodal Generative AI Applications — Course Reference

*A consolidated study reference covering the foundations of multimodal AI, computer vision, text/speech processing and text-to-speech, speech-to-text, image generation, image captioning, text-to-video and image-to-video, multimodal chatbots and QA systems, real-world strengths and limitations, integration challenges, and hands-on project patterns (AI storyteller, image captioning, AI Nutrition Coach, fashion/retail assistant).*

> This is a companion reference to the earlier course files (GenAI/LangChain/Flask, RAG/LlamaIndex/Gradio, Vector Databases/Chroma DB, FAISS/HNSW/Advanced Retrievers). Concepts such as embeddings, cosine similarity, Flask apps, and prompt templates are covered there and reused here in a multimodal context.

---

## Table of Contents

1. [Course Overview](#1-course-overview)
2. [What Is Multimodal Generative AI?](#2-what-is-multimodal-generative-ai)
3. [Introduction to Multimodal AI — Architecture & Landscape](#3-introduction-to-multimodal-ai--architecture--landscape)
4. [Computer Vision Basics](#4-computer-vision-basics)
5. [Text Processing, Speech Processing & Text-to-Speech](#5-text-processing-speech-processing--text-to-speech)
6. [Speech-to-Text (STT / ASR)](#6-speech-to-text-stt--asr)
7. [Text-to-Speech (TTS) Technologies](#7-text-to-speech-tts-technologies)
8. [Project Pattern: AI Storyteller (LLM + gTTS)](#8-project-pattern-ai-storyteller-llm--gtts)
9. [Image Generation (DALL-E / GPT Image)](#9-image-generation-dall-e--gpt-image)
10. [Image Captioning with a Multimodal LLM](#10-image-captioning-with-a-multimodal-llm)
11. [Text-to-Video & Image-to-Video](#11-text-to-video--image-to-video)
12. [Multimodal Chatbots & QA Systems](#12-multimodal-chatbots--qa-systems)
13. [Advanced Multimodal Applications — Code Patterns](#13-advanced-multimodal-applications--code-patterns)
14. [Strengths, Limitations & Real-World Applications of Vision Models](#14-strengths-limitations--real-world-applications-of-vision-models)
15. [Challenges in Multimodal AI Integration](#15-challenges-in-multimodal-ai-integration)
16. [Cheat Sheets Recap](#16-cheat-sheets-recap)

---

## 1. Course Overview

The course teaches how to build applications that **see, listen, speak, and create**, combining text, speech, images, and video.

- **Tools/platforms covered**: IBM Granite, Meta Llama, OpenAI Whisper, DALL-E / GPT Image, Sora, Hugging Face, Gradio, Flask, LangChain, IBM watsonx.ai, gTTS.
- **Target roles**: software developers, full-stack developers, AI/ML engineers, AI researchers, data scientists.
- **Skills gained**: multimodal integration techniques, AI model deployment, text-to-image/video, image captioning, speech processing, cross-modal retrieval, chatbot development.
- **Prerequisites**: generative AI basics (including RAG), Python, basic web development.

**Course structure (three modules)**
1. **Foundations of multimodal AI** — how data types are integrated; hands-on AI storyteller, TTS systems, AI meeting assistant (Whisper + Mixtral + gTTS).
2. **Integrating visual and video modalities** — DALL-E/GPT Image, image captioning, text-to-video (Sora and others).
3. **Advanced multimodal applications** — personal shopping assistants, visual chatbots, calorie estimation from food photos, multimodal retrieval/search.

---

## 2. What Is Multimodal Generative AI?

### 2.1 Multimodal AI
**Multimodal** = the ability to process and integrate information from multiple *modes* of data: **text, images, audio, video**. Traditional AI specializes in one mode (NLP *or* computer vision); multimodal AI combines insights across modes for richer reasoning and output. Analogy: reading a recipe (text) while looking at photos (images) and listening to an explanation (audio) — the brain synthesizes everything.

### 2.2 Generative AI
**Generative** = models *create new content* (text, images, audio, video) by learning patterns from large datasets, rather than only classifying or predicting. They don't retrieve pre-written content; they generate new content from learned structure.

### 2.3 How the Landscape Changed
- For years, AI was siloed: **CNNs** for vision, **Transformers** for language — excellent specialists, no generalists, no easy information exchange between modalities.
- **CLIP** (OpenAI, 2021) showed one model can understand images *and* text by learning a **shared "language" of concepts**, paving the way for models that caption images, answer visual questions, interpret audio, etc.
- Industry examples: **Meta Llama 3.2 Vision Instruct** (text + images), **OpenAI o1** (images, documents, and text side by side).

### 2.4 Why It Matters
- **Richer insight**: e.g., combine product images with customer reviews to explain how color/design influences sentiment (text-only or vision-only models can't).
- **Unified assistant**: one system to read a manual *and* interpret a technical diagram.
- **Beyond business**: educators (richer materials), journalists (multimedia analysis), medical professionals (patient scans + electronic health records).
- **More natural interaction**: like collaborating with a colleague who can read your reports *and* interpret your photos.

---

## 3. Introduction to Multimodal AI — Architecture & Landscape

### 3.1 Definition & Evolution
Multimodal AI processes and understands **multiple data types simultaneously** (text, images, audio, video), mimicking how humans integrate what they see, hear, and read.

- **Before**: specialized single-task models (e.g., **BERT** for text, **ResNet** for image recognition) that couldn't exchange information.
- **Now**: integrated models — **CLIP**, **IBM Granite 3.2 Vision**, **Meta Llama 3.2 / Llama 4**, **OpenAI GPT-4.1 / GPT-4o**, **Google Gemini / Gemma**, **Anthropic Claude 3.7 Sonnet**.
- **Direction**: unified, general-purpose models with seamless multimodal understanding.

### 3.2 Core Functionalities
- **Text processing** (NLU), **computer vision** (images/video), **speech processing** (audio → text/meaning), **text-to-speech**, and **multimodal fusion** (combining everything into a unified understanding).

### 3.3 How a Multimodal System Works
1. **Input processing** — each modality handled by its own specialized **encoder**.
2. **Feature extraction** — key features pulled from each modality.
3. **Alignment** — ensure different data types are properly synchronized/matched.
4. **Multimodal fusion** — combine information to enhance context understanding.
5. **Output generation** — a unified response taking all modalities into account.

*(Simplified view — modern architectures are often more complex.)*

### 3.4 Applications
Virtual assistants (voice + visual context), healthcare (medical images + patient records), education (interactive text/image/audio learning), autonomous vehicles (visual, audio, sensor data), content creation (rich multimedia), and **accessibility**.

### 3.5 Key Industry Players & Models
| Company | Models / contributions |
|---|---|
| **IBM** | Granite 3.2 Vision (document understanding focus) |
| **OpenAI** | GPT-4o (text + images), CLIP, DALL-E |
| **Meta** | Llama 3.2, Llama 4 (open-source multimodal family) |
| **Google** | Gemini, Gemma |
| **Anthropic** | Claude 3.7 Sonnet |

### 3.6 Challenges (overview)
Data alignment, modality imbalance (data availability/quality differs), heavy compute requirements, interpretability, inherited bias, and privacy of sensitive multimodal data. *(Detailed in Section 15.)*

### 3.7 Future Trends
Unified models, **edge computing** (on-device processing → privacy + lower latency), **self-supervised learning** (less labeled data), personalization, and ethics (fairness, transparency, responsible use).

### 3.8 How to Get Started
Learn AI/ML fundamentals → explore open-source tools (**Hugging Face, TensorFlow, PyTorch**) → start with simple projects (image captioning, text-to-speech) → join communities/hackathons → follow industry leaders → experiment with pre-built multimodal APIs.

---

## 4. Computer Vision Basics

**Computer vision (CV)** lets machines "see" and understand visual information from images and video; in multimodal AI it's one of the key modalities combined with text and audio.

### 4.1 Evolution
- 1960s: teaching computers to recognize simple geometric shapes.
- 2010s breakthrough: **Convolutional Neural Networks (CNNs)** — mimic the human visual cortex, automatically learning features from simple edges/textures up to complex objects/scenes.
- Today: recognize thousands of objects, detect/track in real-time video, generate image descriptions, create images from text, analyze medical scans, enable autonomous vehicles.

### 4.2 How Computer Vision Works (5 steps)
1. **Image acquisition** — capture via cameras or load existing images.
2. **Preprocessing** — resizing, normalization, cleaning.
3. **Feature extraction** — identify important visual elements.
4. **Pattern recognition** — ML classifies/understands what's in the image.
5. **Interpretation** — make sense of recognized patterns and generate useful outputs.

### 4.3 Applications in Multimodal AI
Image captioning, **visual question answering (VQA)**, document analysis (text + visual elements), video understanding (visual + audio), augmented reality.

### 4.4 Real-World Impact
Smartphones (face unlock, photo organization), healthcare (medical image analysis), retail (automated checkout, inventory), security (surveillance), transportation (autonomous vehicles).

### 4.5 Challenges & Future
- **Challenges**: robustness (lighting/angles/environments), interpretability, ethics (privacy, bias), efficiency (compute cost).
- **Future**: self-supervised learning, efficient architectures for mobile devices, better integration with other modalities, improved 3D scene and spatial understanding.

---

## 5. Text Processing, Speech Processing & Text-to-Speech

### 5.1 Text Processing (Understanding Written Language)
1. **NLP** — sentence structure/grammar, context and meaning, entity identification (people/places/things), key information extraction, human-like response generation.
2. **Text classification** — spam detection, sentiment analysis, topic categorization, content moderation.
3. **Information extraction** — named entity recognition, relationship extraction, key-phrase identification, summarization.

### 5.2 Speech Processing (Understanding Spoken Language)
1. **Speech recognition** — audio → text; handles accents/dialects, background noise, multiple languages, real-time transcription.
2. **Speaker recognition** — voice authentication, speaker identification ("who said what"), emotion detection in voice.
3. **Speech understanding** — command recognition, intent detection, context understanding, dialogue management.

### 5.3 Text-to-Speech (Giving Voice to AI)
- **Modern TTS**: neural models with human-like voices, emotional expression, natural prosody/intonation, multiple voice options, real-time synthesis.
- **Applications**: virtual assistants, audiobook narration, accessibility tools, interactive voice response, educational content.

### 5.4 Integration in Multimodal AI
- **Conversational AI** — understand text and speech input, respond via text or speech, maintain context across modalities.
- **Content creation** — written content, audio narration, multimedia presentations, interactive experiences.
- **Accessibility** — TTS for visual impairments, speech transcription for hearing impairments, multimodal learning materials, universal design.

### 5.5 Real-World Use
Healthcare (medical transcription), education (language learning), customer service (voice assistants), entertainment (interactive games/characters), business (meeting transcription, document analysis).

### 5.6 Challenges & Future
- **Technical**: accents/dialects, context and nuance, cross-modality consistency, real-time requirements.
- **Ethical**: voice-data privacy, language-model bias, accessibility for all users, responsible AI.
- **Future**: more expressive TTS, deeper language understanding, better multilingual capability, tighter modality integration.

---

## 6. Speech-to-Text (STT / ASR)

STT is also called **Automatic Speech Recognition (ASR)** — combining audio processing with natural-language understanding; it recognizes speech patterns and **phonemes** (the smallest units of sound).

### 6.1 Evolution
Template matching/rule-based (tiny vocabularies) → **Hidden Markov Models** and statistical approaches → deep learning/neural networks (end-to-end architectures) → **self-supervised learning + transformers** learning from vast unlabeled audio.

### 6.2 The STT Pipeline
1. **Audio capture & cleaning** — raw input normalized; noise reduction / voice activity detection.
2. **Feature extraction** — waveform → compact representation, typically a **spectrogram** or **MFCCs** (Mel-frequency cepstral coefficients).
3. **Acoustic model** — maps short audio frames (a few ms each) to basic sound units (phonemes).
4. **Phonetic/grapheme recognition + decoder** — turns phonemes/subword units into words; a **language model** improves accuracy using context.
5. **Output text** — sometimes with punctuation/formatting for readability.

### 6.3 End-to-End Example — Wav2Vec2
Facebook AI Research's **Wav2Vec2** directly maps audio to text (no separate acoustic/language/decoding stages): load audio → preprocess to the right format → pass through the model → decode output to a transcription. Pre-trained on thousands of hours of audio, then fine-tuned for speech recognition.

### 6.4 Applications
Accessibility (captions), virtual assistants, healthcare (medical transcription), education (note-taking, language learning), business (meeting transcription, customer service), legal (court reporting, depositions).

### 6.5 Challenges
Background noise, speaker variability (voices/accents), real-time processing, **domain adaptation** (medicine/law vocabulary), low-resource languages, and capturing semantic meaning beyond words.

### 6.6 Future Trends
Self-supervised learning, multilingual models, contextual understanding, personalization, **edge/on-device processing** (privacy + latency).

### 6.7 Getting Started
Explore **OpenAI Whisper** (open-source), cloud services (Google Speech-to-Text, Azure Speech), simple transcription projects, audio-processing basics, linguistics (phonetics/phonology), communities and hackathons.

---

## 7. Text-to-Speech (TTS) Technologies

TTS combines **linguistic analysis with speech synthesis**.

### 7.1 Evolution
1. **Rule-based formant synthesis** — robotic-sounding speech.
2. **Concatenative synthesis** — assembling pre-recorded speech segments; more natural.
3. **Deep learning** — Google's **WaveNet** and **Tacotron** generate speech waveforms directly from text; highly natural/expressive.
4. **End-to-end architectures** — text → speech without intermediate representations; efficient, real-time capable.
5. **Latest** — emotionally expressive, context-aware TTS.

### 7.2 The Modern TTS Pipeline
1. **Text preprocessing** — normalization: expand abbreviations, numbers → words, **grapheme-to-phoneme** conversion for pronunciation.
2. **Linguistic feature extraction** — syntax, semantics, prosody.
3. **Acoustic model** — predicts pitch, duration, energy, typically as **mel-spectrograms**.
4. **Neural vocoder** — converts mel-spectrograms into the actual audio waveform.

### 7.3 End-to-End Example — VITS
**VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) combines **VAEs, normalizing flows, and GANs** in a single unified framework to map text directly to audio waveforms. The example uses Facebook AI Research's pre-trained VITS: tokenize the text → feed to the model → receive the synthesized waveform.

### 7.4 Applications
Accessibility (screen readers, audiobooks), virtual assistants (Siri, Alexa, Google Assistant), education (language learning, audio content), entertainment (gaming, interactive media), healthcare (clear information delivery), navigation (GPS/transport systems).

### 7.5 Challenges
Natural **prosody** (rhythm and stress), conveying emotional context, **multi-speaker synthesis**, real-time latency, multilingual support.

### 7.6 Future
Personalized voices created in seconds, nuanced emotion, real-time speech translation preserving vocal identity, context-aware adaptation, **zero-shot learning** for instantly adopting new voice styles.

### 7.7 Getting Started
Explore open-source tools, test cloud TTS solutions, build small projects (e.g., TTS for a blog), experiment with voice personalities (tone/age/accent), gather user feedback, iterate.

---

## 8. Project Pattern: AI Storyteller (LLM + gTTS)

**Goal**: generate an educational story on any topic with an LLM, then convert it to speech — an end-to-end text → audio pipeline played inside a Jupyter notebook.

- **LLM**: Mistral (the lab uses Mistral Medium via watsonx.ai) — sparse Mixture-of-Experts architecture, strong at creative writing/analysis/storytelling.
- **gTTS (Google Text-to-Speech)**: Python library/CLI over Google Translate's TTS API — multiple languages/accents, natural sound, easy integration, MP3 output or streaming, speed control.

**Setup**
```bash
pip install gTTS==2.5.4
pip install ibm-watsonx-ai==1.1.20
```

**Credentials & model**
```python
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
project_id = "skills-network"

client = APIClient(credentials)
client.foundation_models.TextModels.show()      # list available models

model_id = 'mistralai/mistral-medium-2505'
params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 1000,
}
model = ModelInference(model_id=model_id, credentials=credentials,
                       project_id=project_id, params=params)
```

**Story generation (carefully structured prompt)**
```python
def generate_story(topic):
    prompt = f"""Write an engaging and educational story about {topic} for beginners.
            Use simple and clear language to explain basic concepts.
            Include interesting facts and keep it friendly and encouraging.
            The story should be around 200-300 words and end with a brief summary of what we learned.
            Make it perfect for someone just starting to learn about this topic."""
    return model.generate_text(prompt=prompt)

story = generate_story("the life cycle of butterflies")
```

**Text-to-speech, playback, and saving**
```python
from gtts import gTTS
from IPython.display import Audio
import io

tts = gTTS(story)
audio_bytes = io.BytesIO()
tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)
Audio(audio_bytes.read(), autoplay=False)   # in-notebook audio player

tts.save("generated_story.mp3")             # optional: save as MP3
```

**Prompt-design takeaways**: specify audience (beginners), language style (simple/clear), content (interesting facts), length (200–300 words), and structure (end with a summary).

---

## 9. Image Generation (DALL-E / GPT Image)

### 9.1 Multimodal Directions (as listed in the lab)
Text→image (e.g., DALL·E), text→audio, image→text (captioning), audio→text (transcription), video analysis (visual + audio).

### 9.2 DALL-E 2 vs. DALL-E 3
| | **DALL·E 2** (2022) | **DALL·E 3** (2023) |
|---|---|---|
| Endpoints | Generations, **edits**, **variations** | Generations only |
| Best for | More options and control; more requests at once | Higher quality, larger sizes, better prompt understanding |
| Sizes | 256×256, 512×512, 1024×1024 | 1024×1024, 1024×1792, 1792×1024 |
| Quality | `standard` only | `standard` (default) or `hd` for enhanced detail |
| Images per request | Up to 10 (`n` parameter) | 1 at a time (parallel requests for more) |
| Notable | Inpainting-style edits, variations | Improved text rendering inside images, artistic styles, safety filtering, ChatGPT integration |

### 9.3 The Images API — Three Endpoints
- **Generations** — create images from scratch from a text prompt.
- **Edits** (DALL·E 2) — replace areas of an existing image using a **mask** (inpainting); transparent mask areas = regions to edit; the prompt should describe the **full new image**, not just the erased area. Image and mask must be square PNGs, <4 MB, same dimensions.
- **Variations** (DALL·E 2) — generate variations of an existing image (square PNG <4 MB).

Output is returned as a **URL (default, expires after ~1 hour)** or **Base64** via `response_format`.

### 9.4 Basic Generation (DALL·E series)
```python
from openai import OpenAI
from IPython import display

client = OpenAI()
response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1,
)
url = response.data[0].url
display.Image(url=url, width=512)
```

### 9.5 GPT Image (Cheat Sheet Version)
GPT Image supports generations, edits, and variations, and allows **up to 8 images per request**.

```python
# Basic generation
response = client.images.generate(
    model="gpt-image-2",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="low",
    n=1,
)

# Base64 instead of URL (no download needed)
import base64, io
from PIL import Image
response = client.images.generate(
    model="gpt-image-2", prompt="a white siamese cat",
    size="1024x1024", response_format="b64_json", n=1,
)
image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
image.show()

# Multiple images at once (up to 8)
response = client.images.generate(
    model="gpt-image-2", prompt="a white siamese cat",
    size="1024x1024", quality="low", n=4,
)
for i, image_data in enumerate(response.data):
    image_bytes = base64.b64decode(image_data.b64_json)
    display.Image(data=image_bytes, width=256)

# Quality options: low / medium / high
response = client.images.generate(
    model="gpt-image-2", prompt="a mountain landscape",
    size="1024x1024", quality="high", n=1,
)

# Size options: 1024x1024, 1024x1536, 1536x1024, auto
response = client.images.generate(
    model="gpt-image-2", prompt="a white siamese cat",
    size="1024x1536", quality="low", n=1,
)

# Save a generated image to disk
image_bytes = base64.b64decode(response.data[0].b64_json)
with open("generated_cat.jpg", "wb") as f:
    f.write(image_bytes)
```

### 9.6 Effective Prompting for Image Generation
- **Be specific and detailed.**
- **Include artistic style references.**
- **Specify lighting, perspective, and composition.**
- **Add context/setting information.**

```python
prompt = "a cat"                                  # basic

prompt = ("a fluffy white siamese cat with blue eyes sitting on a window sill, "
          "golden hour lighting, soft shadows, shallow depth of field, "
          "professional photography style")       # detailed

prompt = ("a white siamese cat in the style of a Renaissance oil painting, "
          "dramatic lighting, rich colors, detailed fur texture")   # artistic style
```

### 9.7 Installation
```bash
pip install openai==1.64.0
```

---

## 10. Image Captioning with a Multimodal LLM

**Image captioning** = automatically generating textual descriptions of images, using computer vision + NLP. Example use case: classifying thousands of vacation photos by year and destination in minutes instead of hours of manual work.

### 10.1 The Three Stages
1. **Input processing** — the system receives an image and (optionally) a text prompt that guides what to focus on; the image is pre-processed (normalized, resized, optimized).
2. **Image validation & encoding** — verify the processed image meets technical requirements, contains detectable features, and suits the model; then **encode to base64** so a text-based API can carry the visual data (captures objects, scenes, relationships, styles).
3. **Multimodal LLM processing**:
   - **Visual encoder** extracts meaningful visual features.
   - The text prompt is converted to numerical vectors via **text embedding**.
   - A **multimodal fusion layer** merges visual features and text embeddings into a unified representation.
   - A **language-generation component** crafts the caption responsive to the prompt.

*(Classic implementations combine a **CNN** image encoder with an **RNN** or transformer-based decoder.)*

### 10.2 Implementation with Llama 4 Maverick on watsonx.ai
Model: `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`.

```python
import base64, requests
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
client = APIClient(credentials)

model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
project_id = "skills-network"
params = TextChatParameters(temperature=0.2, top_p=0.6, max_tokens=2000)

model = ModelInference(model_id=model_id, credentials=credentials,
                       project_id=project_id, params=params)

# Encode multiple images from URLs → base64
def encode_images_to_base64(image_urls):
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_images.append(base64.b64encode(response.content).decode("utf-8"))
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images

# Send text + image together
def generate_model_response(encoded_image, user_query,
        assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": assistant_prompt + user_query},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + encoded_image}},
        ]
    }]
    response = model.chat(messages=messages)
    return response['choices'][0]['message']['content']

# Caption every image
user_query = "Describe the photo"
for i, image in enumerate(encoded_images):
    print(f"Description for image {i + 1}: {generate_model_response(image, user_query)}")
```

**Key idea**: the model processes the image with vision components and the text prompt simultaneously, using **attention mechanisms** to relate visual features to language concepts, then generates descriptive text.

**Alternative vision model** (used elsewhere in the course): `ibm/granite-vision-3-2-2b` with `TextChatParameters(temperature=0.2, top_p=0.5)`.

---

## 11. Text-to-Video & Image-to-Video

### 11.1 Text-to-Video — How It Works
1. **Text encoding** — a language model turns the prompt into a high-dimensional semantic vector.
2. **Latent space generation** — the vector guides a **diffusion model** that starts from random noise and iteratively refines it into latent representations of video frames.
3. **Temporal consistency** — ensures smooth frame-to-frame transitions via **3D U-Nets** (spatiotemporal extensions of U-Nets) and/or **Transformers** (self-attention across frames).
4. **Video decoding** — a decoder network (often a CNN) converts refined latents into actual frames.
5. **Frame interpolation** *(optional)* — generates intermediate frames for higher frame rates/smoothness.

### 11.2 Text-to-Video Models (as listed in the reading)
| Model | Released | Highlights | Access |
|---|---|---|---|
| **OpenAI Sora** | Dec 2024 | Up to 60 s videos; complex scenes, camera motion; diffusion-based 3D patch generation | ChatGPT Plus/Pro |
| **Google Veo 2** | Apr 2025 | 8 s, 720p; cinematic quality; strong motion/physics modeling | Gemini Advanced, Google One AI Premium |
| **Runway Gen-4** | Apr 2025 | Consistent characters, storytelling control, real-world aesthetics | Paid/enterprise |
| **MiniMax Hailuo T2V-01-Director** | Jan 2025 | High control over scene motion and randomness | Hailuo AI platform |
| **Step-Video-T2V** | Feb 2025 | 30B parameters, Video-VAE, 204-frame videos | Open-source (GitHub) |
| **AMD Hummingbird** | Mar 2025 | Lightweight, 31× speedup, only 4 GPUs to train | Open-source |

### 11.3 Image-to-Video — How It Works
1. **Feature extraction** — edges, textures, semantic content via CNNs/feature extractors.
2. **Motion prediction** — **optical flow estimation** (pixel-level motion) or **latent flow models** (motion in compressed latent space).
3. **Frame generation** — synthesize frames by warping the original image, using **GANs** or **VAEs**.
4. **Video assembly** — compile frames, with post-processing (stabilization, color correction).

### 11.4 Image-to-Video Models
| Model | Released | Highlights | Access |
|---|---|---|---|
| **OpenAI Sora** | Dec 2024 | Animates static images with realistic motion/scene transitions | ChatGPT Plus/Pro |
| **Google Whisk Animate** | Apr 2025 | Stills → 8 s, 720p video with animation and scene expansion | Google One AI Premium |
| **I2V3D** | Mar 2025 | 3D camera movement and object rotation; geometry-aware | Open-source (code to be released) |
| **MiniMax Hailuo I2V-01-Director** | Jan 2025 | High control over motion from a single image | Hailuo AI platform |

### 11.5 OpenAI Sora — Demo Walkthrough
**Sora** = a multimodal, **diffusion-based transformer** generating high-quality video from text or images. Uses: creative storytelling (VR/AR/games), content creation with custom characters/scenes without animation teams, video editing (upscaling, interpolation, gap-filling), and **synthetic data** generation (tracking, segmentation, action recognition).

**Other tools**: **Runway Gen-3 Alpha** (cinematic output, fine-grained motion control), **Pika Labs** (stylized animated clips, storyboarding), **Google Lumiere** (space-time diffusion, smooth coherent motion).

**Workflow**: write a descriptive prompt → send to Sora → receive a video clip.

**Prompt = 3 essential elements**
1. **Scene context** — environment, location, weather.
2. **Visual details** — lighting, color tones, camera angles.
3. **Motion** — slow motion, drone shots, panning, etc. *(Sora responds well to cinematic/film vocabulary.)*

**Example prompt**: *"A black cat walks confidently across a sunlit rooftop at golden hour in a bustling city. Warm, low-angle sunlight casts long shadows as a drone camera slowly circles the cat, revealing the skyline. Rich, cinematic color tones with a hint of slow motion highlight the cat's graceful movements."*

**Interface steps**: open sora.openai.com → log in → Explore (inspiration), Library, **Composer** (prompt box) → set Type = Video, aspect ratio, resolution, duration, number of variations (options vary by subscription) → Create Video (queued; ~30 s to a few minutes) → videos saved under **My Videos**.

**Editing toolbar**
- **Storyboard / Recut** — trim or extend sections on a new timeline.
- **Remix** — describe changes in natural language (e.g., "make the cat a fluffy orange cat").
- **Blend** — transform the video with content from another video.
- **Loop** — create a seamless repeating section.

### 11.6 Applications
Marketing/advertising (promo videos, multilingual/localized ads at scale), education/e-learning (tutorials, animated explainers), entertainment/media (storyboards, VFX, background scenes, concert visuals), social media (short-form content, memes, GIFs), corporate training (onboarding, policy explainers).

### 11.7 Challenges
- **Computational complexity** — high training/inference cost; longer/higher-res video needs more power.
- **Temporal & spatial coherence** — flickering, unnatural transitions, inconsistent lighting.
- **Data limitations** — need large, diverse, high-quality annotated video data.
- **Ethical/legal** — deepfakes, misinformation, copyright and consent uncertainty.
- **Interpretability & control** — limited user control over specifics.

### 11.8 Future Directions
More efficient architectures (edge/real-time), finer prompt/UI control and feedback loops, richer multimodal inputs (text + image + audio + video), watermarking/content-verification and regulation, personalized/adaptive content.

---

## 12. Multimodal Chatbots & QA Systems

### 12.1 What They Are
Advanced AI applications that **process, understand, and generate responses from multiple input types** — text, images, audio, sometimes video — unlike text-only chatbots. Example: showing your phone a photo of fridge ingredients and asking for a recipe, with follow-up questions.

### 12.2 Architecture
1. **Accept multiple input modalities** — text (queries/commands), images (photos/diagrams), audio (voice/ambient), video.
2. **Process each modality separately**, then **fuse across modalities** into a unified understanding (e.g., answering a question about an image = text + visual analysis).
3. **Generate a response** — text answers, suggestions, action prompts, or even image generation.

### 12.3 Basic Implementation (Llama 3.2 Vision / Llama 4 via watsonx.ai)
1. **Set up environment & libraries.**
2. **Initialize the model** — credentials (URL + API key), `model_id`, `project_id`, `TextChatParameters` (e.g., temperature to control creativity vs. determinism), then a `ModelInference` object.
3. **Prepare images** — AI can't "see" like humans, so convert to numeric/text representations:
   - `prepareImage` — read a local file in binary → **Base64** string.
   - `prepareImageFromURL` — download via HTTP request → Base64.
4. **Create the multimodal query function** — build the message structure with both text and image elements, call `model.chat(...)`, extract the text from the (possibly complex) return object.
5. **Use the QA system** — pass the encoded image, a specific question, and an optional **system prompt** that assigns a role/perspective (e.g., "expert nutritionist" vs. "fashion consultant" — the same image yields very different analyses).

```python
def send_multimodal_query(model, encoded_image, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ]
    }]
    response = model.chat(messages=messages)
    return response['choices'][0]['message']['content']

def visual_question_answering(model, encoded_image, question):
    prompt = f"Please answer the following question about the image: {question}"
    return send_multimodal_query(model, encoded_image, prompt)

def generate_image_caption(model, encoded_image):
    prompt = "Please provide a detailed description of this image."
    return send_multimodal_query(model, encoded_image, prompt)
```

### 12.4 Foundations Cheat-Sheet Helpers
```python
import base64
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    return image.resize(target_size)

def create_prompt(image_description, user_query):
    return f"""
    Analyze the following image and answer the question.
    Image Description: {image_description}
    User Query: {user_query}
    Please provide a detailed response that:
    1. Describes the image content
    2. Answers the specific question
    3. Provides relevant context
    """
```

Model setup used in the cheat sheet:
```python
params = TextChatParameters(temperature=0.2, top_p=0.5, max_tokens=2000)
model = ModelInference(
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    credentials=credentials, project_id="skills-network", params=params
)
```

---

## 13. Advanced Multimodal Applications — Code Patterns

### 13.1 Environment Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install ibm-watsonx-ai==1.1.20 image==1.5.33 flask requests==2.32.0
pip install torch torchvision scikit-learn pillow gradio
```

### 13.2 Basic Image Querying & Object Detection by Prompting
Using `generate_model_response(encoded_image, user_query)` (Section 10.2):
```python
# Describe
generate_model_response(encoded_images[0], "Describe the photo")

# Count objects
generate_model_response(encoded_images[1], "How many cars are in this image?")

# Examine details
generate_model_response(encoded_images[1], "What color is the woman's jacket in this image?")

# Read text from images (OCR-style)
generate_model_response(encoded_images[3], "How much sodium is in this product?")   # nutrition label
```
*Key insight*: a general vision LLM can do counting, attribute detection, and text reading simply by asking specific questions — no separate detector needed.

### 13.3 Flask Web App for Vision AI
```python
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")
        if uploaded_file:
            encoded_image = input_image_setup(uploaded_file)
            response = generate_model_response(encoded_image, user_query, assistant_prompt)
            return render_template("index.html", user_query=user_query, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
```
```python
import base64

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        return base64.b64encode(bytes_data).decode("utf-8")
    else:
        raise FileNotFoundError("No file uploaded")
```

### 13.4 Nutrition Analysis Prompt (Structured Output)
```python
def generate_nutrition_response(encoded_image, user_query):
    assistant_prompt = """
    You are an expert nutritionist. Your task is to analyze the food items displayed
    in the image and provide a detailed nutritional assessment using the following format:
    1. **Identification**: List each identified food item clearly, one per line.
    2. **Portion Size & Calorie Estimation**: For each item, specify portion size and
       estimated calories (e.g. * **Salmon**: 6 ounces, 210 calories).
    3. **Total Calories**: Provide the total.
    4. **Nutrient Breakdown**: Protein, Carbohydrates, Fats, Vitamins, Minerals (bullet points).
    5. **Health Evaluation**: One paragraph on the healthiness of the meal.
    6. **Disclaimer**: Include the exact approximate-values/consult-a-professional text.
    """
    return generate_model_response(encoded_image, user_query, assistant_prompt)
```
**Prompt-engineering takeaways**: assign an expert role, define a numbered output structure with examples, and include a mandatory disclaimer.

### 13.5 Fashion / Retail Assistant — Multimodal Retrieval + LLM
**Idea**: embed images as vectors, find the closest catalog item by **cosine similarity**, and feed the matched product details into the LLM prompt.

**Image → vector embedding (ResNet50)**
```python
import torch, requests, base64
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from io import BytesIO
import numpy as np

class ImageProcessor:
    def __init__(self, image_size=(224, 224),
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def encode_image(self, image_input, is_url=True):
        try:
            if is_url:
                image = Image.open(BytesIO(requests.get(image_input).content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(input_tensor)
            return {"base64": base64_string, "vector": features.cpu().numpy().flatten()}
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}
```

**Similarity matching**
```python
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_match(user_vector, dataset):
    try:
        dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
        similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
        closest_index = np.argmax(similarities)
        return dataset.iloc[closest_index], similarities[0][closest_index]
    except Exception as e:
        print(f"Error finding closest match: {e}")
        return None, None
```

**Threshold-aware prompting** — if `similarity_score >= threshold` (default 0.8), the prompt presents an **ITEM DETAILS** section (exact match); otherwise it presents a **SIMILAR ITEMS** section noting these are *similar but not exact*. Both prompts ask the model to objectively identify clothing elements (colors, patterns, materials), categorize style (business/casual, etc.), and append the product list (name, price, link), in formal catalog language.

### 13.6 Hands-On Lab: AI Nutrition Coach (Flask + Llama 4 Maverick)
A 30-minute project building an **image-based Flask web app** that estimates calories from a food photo and returns dietary advice.

- **Learning objectives**: build a Flask app with AI capabilities; use `ibm-watsonx-ai` to talk to **Llama 4 Maverick 17B 128E Instruct FP8** for food recognition; integrate visual recognition with nutritional data for calorie estimation.
- **Project structure**: `cal_coach_app/` with `app.py` (main logic), `templates/index.html`, `static/style.css`; run with `python app.py` on **port 5000** inside an activated virtual environment (`source my_env/bin/activate`).
- **Helper functions in `app.py`**: model initialization, `input_image_setup` (upload → Base64), `format_response` (clean the model's output for the UI), `generate_model_response` (send text + image to the model).
- **Disclaimer from the lab**: generative AI outputs may be inaccurate/inconsistent — use discretion.

---

## 14. Strengths, Limitations & Real-World Applications of Vision Models

### 14.1 Strengths
- **Cross-modal understanding** — links images and text; **zero-shot learning** (e.g., Qwen2.5-VL recognizing unseen objects via textual descriptions); **contextual understanding** of nuance a unimodal model would miss.
- **Flexibility & generalization** — strong transfer learning/domain adaptation with minimal fine-tuning; reduced task-specific data needs.
- **Enhanced performance** — modalities provide complementary signals; integrating inputs improves robustness to noise or missing information.

### 14.2 Limitations & Challenges
- **Computational complexity** — heavy compute for training/inference; energy consumption/environmental impact.
- **Alignment issues** — cross-modal alignment is hard; cultural/contextual bias.
- **Reliability** — **hallucinations** (plausible but factually wrong outputs), uneven performance across tasks/domains, **adversarial vulnerability** exploiting cross-modal interaction.

### 14.3 Real-World Applications
| Domain | Examples |
|---|---|
| **Healthcare** | Medical image interpretation with patient history/clinical notes; accessibility tools translating visual info for visually impaired users |
| **Retail & e-commerce** | Visual search (search by image), virtual try-on |
| **Education & training** | Adaptive multimodal learning materials; simulation environments (medicine, engineering, emergency response) |
| **Content creation & curation** | Automated visual + textual content; cross-modality content moderation |
| **Autonomous systems** | Robots following natural-language instructions in a visual environment; vehicle navigation (visual + map + directions) |

### 14.4 Case Studies
1. **Accessibility** — in July 2023, **Envision** integrated OpenAI's **GPT-4** into its assistant **Ask Envision** to give blind users detailed visual descriptions of surroundings (e.g., via Google Glass), improving independence.
2. **Autonomous driving** — in October 2024, **Waymo** explored Google's **Gemini** to train vehicles; the resulting **EMMA** (End-to-End Multimodal Model for Autonomous Driving) processes sensor data to generate future trajectories, leveraging multimodal world knowledge/reasoning.
3. **Healthcare** — **PathChat** (Dec 2023), a vision-language AI assistant for pathology: a vision encoder pre-trained on millions of histology images and image-caption pairs, combined with an LLM and fine-tuned on diverse visual-language instructions; showed high diagnostic accuracy with potential in education, research, and clinical decision support.

---

## 15. Challenges in Multimodal AI Integration

### 15.1 Technical Challenges
| Challenge | Problem | Ways to overcome |
|---|---|---|
| **Combining different data types** | Text and images are very different; even CLIP (trained on millions of image-text pairs) struggles on data unlike its training set | More robust **data augmentation**; **multi-task learning** frameworks that learn from multiple data types simultaneously |
| **Fusing modalities** | No perfect fusion strategy; many systems use **late fusion** (separate encoders, merge near the end), which limits deep cross-modal interaction | **Early fusion** (combine raw data at input level); **cross-attention** to dynamically align/integrate modalities throughout the model |
| **Consistency & hallucinations** | Models may misread objects or mix text and visuals | **Grounding** (e.g., object-detection models to verify visual claims); **cross-modal consistency checks** before final generation |

### 15.2 Ethical Concerns
| Concern | Problem | Ways to overcome |
|---|---|---|
| **Bias in data** | Internet-trained models reflect stereotypes; better at recognizing Western objects | Diverse cultural/racial/geographic datasets; bias detection & mitigation frameworks |
| **Deepfakes & misinformation** | Realistic fake images/voices/videos enable impersonation and false info | **Watermarking** of AI content; AI detectors for synthetic media |
| **Privacy risks** | Vision/audio AI may identify people or capture private info; surveillance concerns | Strict data governance (anonymization, encryption); **differential privacy** |

### 15.3 Implementation Issues
| Issue | Problem | Ways to overcome |
|---|---|---|
| **High cost & resources** | Heavy compute; hard for small teams | **Knowledge distillation**, parameter sharing, **pruning**; scalable cloud platforms |
| **Difficult to deploy** | Slow response times; costly real-time integration across text/image/audio | **Model compression**; **modular architectures** that process each modality separately but synchronize outputs |
| **Imbalanced data** | Abundant English text vs. scarce data for other languages/non-Western cultures → less fair/reliable globally | Prioritize underrepresented groups in data collection; **data augmentation** to balance datasets |

### 15.4 Transparency & Explainability
- **Problem**: multimodal systems often act as **black boxes**, causing mistrust and hindering adoption in critical sectors (healthcare, finance).
- **Solutions**: **Explainable AI (XAI)**; transparent design practices (documenting data sources, architectures, decision processes); regulatory alignment (e.g., GDPR's "right to explanation").

---

## 16. Cheat Sheets Recap

### 16.1 Cheat Sheet: Foundations of Multimodal AI
Image captioning, image processing/base64 encoding (`encode_image`, `process_image`), multimodal model setup with watsonx.ai (Llama 4 Maverick, `TextChatParameters`), the multimodal query function (`send_multimodal_query`), text processing/prompt creation (`create_prompt`), and visual question answering (`visual_question_answering`) — Sections 10 and 12.

### 16.2 Cheat Sheet: Integrating Visual and Video Modalities
Base64 image responses, credentials setup, GPT Image generation (`gpt-image-2`; up to 8 images per request), effective prompting tips, file-download helper, image captioning loop, image display (`IPython.display.Image`), image encoding from URLs, message formatting, model invocation, vision-model initialization (`ibm/granite-vision-3-2-2b`), multiple-image generation (`n` parameter), OpenAI client initialization, object detection by prompting, `pip install` list, quality options (low/medium/high), saving generated images, and size options (1024x1024, 1024x1536, 1536x1024, auto) — Sections 9, 10, and 13.

### 16.3 Cheat Sheet: Advanced Multimodal Applications
Basic image querying and object detection by prompting, `create_vision_message`, environment setup, **fashion-analysis prompting** with similarity thresholds, **Flask integration** for vision AI web apps, image encoding from URLs and uploads, **nutrition-analysis prompt** template, **cosine-similarity matching**, **ResNet50 image embeddings**, and vision-model initialization — Section 13.

### 16.4 Readings Recap
- *What Is Multimodal Generative AI and Why Does It Matter?* — Section 2.
- *What Is Computer Vision?* — Section 4.
- *Text Processing, Speech Processing, and Text-to-Speech* — Section 5.
- *Introduction to Text-to-Video and Image-to-Video Technologies* — Section 11.
- *Strengths, Limitations, and Practical Applications of Multimodal Vision Models* — Section 14.
- *Challenges in Multimodal AI Integration* — Section 15.

---

## Appendix: Source Material Map

This reference was compiled from:
- **Video lessons**: Welcome to multimodal applications with GenAI (course overview); Introduction to Multimodal AI; Text-to-Speech Technologies; Speech-to-Text Technologies; Understanding Image Captioning with Meta's Llama; Text-to-Video Generation with OpenAI's Sora (demo); Multimodal Chatbots and QA Systems.
- **Readings/cheat sheets**: *What Is Multimodal Generative AI and Why Does It Matter?*; *What Is Computer Vision?*; *Text Processing, Speech Processing, and Text-to-Speech*; *Introduction to Text-to-Video and Image-to-Video Technologies*; *Strengths, Limitations, and Practical Applications of Multimodal Vision Models in Real World Scenarios*; *Challenges in Multimodal AI Integration*; *Cheat Sheet: Foundations of Multimodal AI*; *Cheat Sheet: Integrating Visual and Video Modalities*; *Cheat Sheet: Advanced Multimodal Applications*.
- **Notebooks/labs**: `Use_Mixtral_and_gTTS_to_create_your_personal_storyteller.ipynb`; `GPT_Image_generation_Guide_for_Beginners.ipynb`; `lab-instructions (8).md` (AI Nutrition Coach Flask app).
