import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

data = {
    'text': [
        "Had a wonderful day with my friends at the beach today!",
        "Just finished reading a great book, highly recommend it to everyone.",
        "Going to the gym to start my morning healthy routine.",
        "I love coding and building new software projects.",
        "I had an amazing dinner with my family today!",
        "Celebrating my birthday with awesome people and good food.",
        "Listening to some joyful music and cooking my favorite meal.",
        "The weather is absolutely beautiful today, perfect for a walk.",
        "Had a highly productive day at work today, feeling accomplished.",
        "Spent the weekend playing football with my teammates.",
        
        "I feel so empty inside, nothing makes me happy anymore. Just want to sleep forever.",
        "I have been feeling hopeless for weeks and crying every night. I'm so tired.",
        "I feel lonely and sad every single day, nobody cares about me.",
        "Everything feels dark and I don't see any hope for my future.",
        "Feeling deeply depressed and lonely tonight, crying alone in my room.",
        "I want to disappear from this world, the emotional pain is too much to bear.",
        "No matter how hard I try, I feel like a complete failure in life.",
        "I have lost all interest in my hobbies, everything feels useless.",
        "Woke up feeling deeply sad and heavy today, couldn't even leave my bed.",
        "Nobody understands how lonely it feels to be surrounded by people.",
        
        "My heart is racing and I can't breathe properly. What if something terrible happens?",
        "I am constantly worrying about the future and can't focus on anything. So anxious.",
        "Woke up feeling panicky and shaking for no clear reason. I can't calm down.",
        "I am having a massive panic attack and my hands are trembling with fear.",
        "So stressed and scared about my exams, my mind is completely blank and panicking.",
        "I can't stop overthinking every single detail, my brain feels like it's exploding.",
        "Feeling extremely nervous and restless, like a disaster is about to happen.",
        "My chest feels tight and my heart rate is going crazy because of fear.",
        "I am absolutely terrified of making mistakes, the pressure is making me panic.",
        "Can't sleep because my mind is constantly running with scary anxious thoughts."
    ],
    'label': [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'#\w+', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Depression', 'Anxiety'], zero_division=0))

def predict_mental_health(post_text):
    cleaned = clean_text(post_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    mapping = {0: "Normal 😊", 1: "Depression Warning 😔", 2: "Anxiety Warning 😰"}
    return mapping[prediction]

print("\n--- Real-time Mental Health Sentiment Analyzer ---")
while True:
    user_post = input("\nEnter a social media post (or type 'exit' to stop): ")
    if user_post.lower() == 'exit':
        print("Thank you for using the Analyzer! Take care. 😊")
        break
    
    if user_post.strip() == "":
        print("Please enter some text!")
        continue
        
    result = predict_mental_health(user_post)
    print(f"Analysis Result: {result}")