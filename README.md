# Detect AI Generated Text using GenAI

## About
This project focuses on Detect AI Generated Text using GenAI. The model, **CNN** , is fined tuned, trained and tested on text classification dataset. This results in a more efficient models with reduced computational costs while maintaining high performance. CNN model has achieved **state-of-the-art results** around 90% accuracy in detecting both AI generated and Human written texts

## Installation and Setup

### Prerequisites
To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git https://github.com/SourabhKD/DetectAI.git
   ```
2. Navigate to the project directory(for each model comes with their folder, lets say CNN):
   ```sh
   cd AI-Generated-Text-Detection
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
## How to Start

1. **Move to the Working Directory**
   ```sh
   cd AI-Generated-Text-Detection
   ``` 
2. **Add the Trained Model**
   - comes with the fine tuned model file no need for training!.
   - You can use your own Dataset and Train the Model or even improve the accuracy!.
   - Comes With the Training File Just run the file with your own dataset

| Dataset | Download Link |(optional)
|---------|--------------|
| **kaggle** | [Download](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)|

4. **Run the Application**
   ```sh
   python app.py
   ```
   - After execution, a local URL link will appear in the terminal.
   - Open the link in a browser to access the interface.
   - use chatgpt or similar AI tool to generate ai text, then copy paste in the text area.
   - use any kind of human written text, paste in text area.
   - The website will dispay the result with certain confidence score whether the provided text is HUMAN OR AI written.

## Model Performance

| Model Name               | Dataset | Training acurracy | Detection accuracy |
|--------------------------|---------|-------------------|--------------------|
| CNN                      | kaggle  |       95%         |         90%        |  

---
## Future Improvments
Build A ensemble model consisting of multiple models piplined together.
This improves over all Accuracy and Detection rate.
