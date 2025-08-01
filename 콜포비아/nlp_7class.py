import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
PATH = r'C:\Users\USER\Desktop\beomi.pth'

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = torch.load(PATH, map_location=torch.device('cpu'))

emotion_labels = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
emotion_korean = {
    'angry': '분노',
    'disgust': '혐오',
    'fear': '불안',
    'happiness': '행복',
    'neutral': '중립',
    'sadness': '슬픔',
    'surprise': '놀람'
}

def predict_emotion(sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    results = {emotion_korean[emotion_labels[i]]: f"{probabilities[i] * 100:.2f}%" for i in range(len(emotion_labels))}

    return results

sample_sentence = input()
predicted_emotion = predict_emotion(sample_sentence)
print(f"입력 문장: {sample_sentence}")
print(f"예측 감정: {predicted_emotion}")
