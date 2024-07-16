import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from transformers import pipeline, DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np

class IssueAnalyzer:
    def __init__(self, url="https://api.github.com/repos/rails/rails/issues"):
        self.url = url
    
    def getIssues(self):
        data_file = "data.csv"
        params = {
            "state": "all",
            "per_page":100
        }

        headers = {
            "Accept": "application/vnd.github.v3+json"
        }

        issues = []
        for page in range(1, 6):
            response = requests.get(
                self.url,
                params= {
                    **params,
                    "page": page
                },
                headers=headers
            )

            if response.status_code != 200:
                raise Exception(f"Error : {response.content}")
            issues.extend(response.json())

        df = pd.DataFrame(issues)
        if not os.path.exists(data_file):
            df.to_csv(data_file, index=False)
        else:
            print("File already exists.")
    
    def analyze(self):
        df = pd.read_csv("data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['month'] = df['created_at'].dt.to_period('M')

        issue_in_time = df.groupby('month').size()
        plt.figure(figsize=(12, 6))
        issue_in_time.plot()
        plt.title("Number of Issues in Specific Time")
        plt.xlabel('Month')
        plt.ylabel("# of Issues")
        plt.show()
    
    def highestPeriods(self):
        df = pd.read_csv("data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['month'] = df['created_at'].dt.to_period('M')
        issues = df.groupby('month').size()
        high_periods = issues[issues > issues.mean() + 2 * issues.std()]
        print(f"Highest Time Period : {high_periods}")

        plt.figure(figsize=(10,5))
        plt.plot(issues.index.to_timestamp(), issues.values, label="Daily Issues")
        plt.scatter(high_periods.index.to_timestamp(), high_periods.values, color='red', label='High Activity Days')

        plt.title('Daily Issues')
        plt.xlabel('Date')
        plt.ylabel('Number of Issues')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def getActiveUser(self):
        df = pd.read_csv("data.csv")
        active_user = df['user'].value_counts().idxmax()
        json_str = json.dumps(active_user, indent=4)
        print(json_str)

    def getPopularCategory(self):
        df = pd.read_csv("data.csv")
        labels = df['labels'].dropna().apply(eval)
        labels = labels.apply(
            lambda x: [label['name'] for label in x]
        )
        label_counts = pd.Series([label for sublist in labels for label in sublist]).value_counts()
        print(f"Popular Category : {label_counts}")

    def classifyIssue(self):
        df = pd.read_csv("data.csv")
        df['body'] = df['body'].astype(str)
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)

        def classify_text(text):
            if pd.isna(text):
                return None
            return classifier(text)[0]['label']

        df['classification'] = df['body'].apply(classify_text)
        df.to_csv("issues_classified.csv", index=False)

        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")

    def generateRandomTrueLabels(self):
        df = pd.read_csv("issues_classified.csv")
        possible_values = ['LABEL_0', 'LABEL_1']
        np.random.seed(42)
        df['random_labels'] = np.random.choice(possible_values, size=len(df))
        df.to_csv("issues_classified.csv", index=False)

    def evaluateResults(self):
        self.generateRandomTrueLabels()
        df = pd.read_csv("issues_classified.csv")
        y_true = df['random_labels']
        y_pred = df['classification']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


analyzer = IssueAnalyzer()
analyzer.getIssues()
analyzer.analyze()
analyzer.highestPeriods()
analyzer.getActiveUser()
analyzer.getPopularCategory()
analyzer.classifyIssue()
analyzer.evaluateResults()
analyzer.classifyIssue()
