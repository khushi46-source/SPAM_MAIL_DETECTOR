import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("mail_data.csv")

# Replace missing values
data = data.fillna('')

# Convert labels into numbers
# Convert labels
data['Category'] = data['Category'].map({
    'spam': 0,
    'ham': 1
})

X = data['Message']
Y = data['Category']

# Convert text into numerical values
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Check accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Test custom messages
while True:
    message = input("\nEnter a message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        break

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    if prediction[0] == 0:
        print("Spam Mail")
    else:
        print("Not Spam Mail")
