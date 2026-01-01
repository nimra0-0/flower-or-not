---
title: Flower or Not
emoji: 🌸
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.2.0"
app_file: app.py
pinned: false
python_version: "3.12"
---

# 🪴 Flower or Not ??

Ever wondered if a neural network can tell whether your doodle is a flower… or just chaotic squiggles? 🌸✏️
This repository contains the code (and a trained model) that does exactly that.

The goal is simple: train and export an image classifier that can distinguish between:

- 🌸 **Flower doodles**
- ❌ **Not-flower doodles**

![Demo](demo.gif)

The model is trained on **hand-drawn sketches** from Google’s **QuickDraw dataset** and exported for easy reuse in future apps or demos(stay tuned — a fun interactive app is coming soon 👀).

For details on training and experimentation, see(The notebook is still in draft so proceed with caution): [Kaggle Notebook](https://www.kaggle.com/code/nimra00/is-it-a-flower-doodle-edition)

---

## 🚀 Live Demo

👉 **Try it on Hugging Face Spaces:**  
**🔗 https://huggingface.co/spaces/N0-0/flower-or-not**

_(Draw a doodle or upload an image and see if the model thinks it’s a flower 🌸)_

---

## ✨ Features

- 🖌️ **Doodle-first classification** (QuickDraw-style sketches)
- 🧠 **FastAI-powered model** trained on Google’s QuickDraw dataset
- 🎨 **Gradio web app** with:
  - Image upload
  - Canvas-based doodle drawing
- 🌍 **Deployed on Hugging Face Spaces** for instant access
- 🔁 Exported `.pkl` model for easy reuse in other projects

---

## 🏗️ Model Training

- Dataset: **Google QuickDraw**
- Task: Binary image classification (`flower` vs `not flower`)
- Framework: **FastAI**
- Input: Hand-drawn sketches (grayscale / resized)
- Output: Class probabilities

> 📓 Training notebooks and experiments can be found here:  
> **🔗 <TK – Coming Soon>**

---

## 🖥️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/nimra0-0/flower-or-not.git

```

### 2️⃣ Create & activate environment (recommended)

Activate Virtual Env (I am using UV):

```
uv venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```
uv sync
```

### 4️⃣ Run the Gradio app

```
uv run python app.py
```

Then open your browser at:

```
http://localhost:7860
```

or whatever the terminal points to

---

### Background

**Flower or Not?** started as a small, simple yet playful CV project inspired by **fastai Lesson 1 and 2**. I built it as a way to revisit and solidify the fundamentals of image classification, but now i have even more ideas on how to make it grow into a lush virual garden. so stay tuned for more...

---

## 📜 License

MIT — feel free to use this project for learning or inspiration.
If you build something cool from it, I’d love to hear about it 🌸
