# Import required packages
import torch
import gradio as gr

from typing import Dict
from transformers import pipeline

# Define a function to use the model
def food_not_food_classifier(text: str) -> Dict[str, float]:

  # Set up food not food text classifier
  food_not_food_classifier_pipeline = pipeline(task="text-classification",
                                      model="karenwky/learn_hf_food_not_food_text_classifier_distilbert-base-uncased",
                                      batch_size=32,
                                      device="cuda" if torch.cuda.is_available() else "cpu",
                                      top_k=None)

  # Get the output from the pipeline
  outputs = food_not_food_classifier_pipeline(text)[0]

  # Format output for Gradio
  output_dict = {}
  for item in outputs:
    output_dict[item["label"]] = item["score"]

  return output_dict

# Create a Gradio interface with detilas about the app
description = """
A text classifier to determine whether a sentence pertains to food or not. 

Fine-tuned from [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) on a [dataset](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions) of LLM-generated image captions categorizing food and non-food topics.

Follow-along [notebook](https://colab.research.google.com/github/karenwky/learn-hugging-face/blob/main/learn_hugging_face_text_classification.ipynb). 
"""

demo = gr.Interface(
    fn=food_not_food_classifier,
    inputs="text",
    outputs=gr.Label(num_top_classes=2),
    title="🍔🚫🍰 Food or Not Food Text Classifier",
    description=description,
    examples=[["Today is a sunny day."],
              ["Pineapple fried rice."]]
)

# Launch the interface
if __name__ == "__main__":
  demo.launch()
