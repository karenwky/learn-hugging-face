---
title: Food Not Food Text Classifier
emoji: 🍔🚫🍰
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
license: mit
---

# 🍔🚫🍰 Food or Not Food Text Classifier

<a target="_blank" href="https://colab.research.google.com/github/karenwky/learn-hugging-face/blob/main/01_text-classification/learn_hugging_face_text_classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A text classifier to determine whether a sentence pertains to food or not. 

Fine-tuned from [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) on a [dataset](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions) of LLM-generated image captions categorizing food and non-food topics.