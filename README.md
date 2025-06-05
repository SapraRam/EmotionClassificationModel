# 🎭 Emotion AI Classifier

A real-time web application that classifies text into six emotions using **TF-IDF** and **Logistic Regression**. Built with **Streamlit**, the app provides predictions, confidence scores, and performance metrics.

---

## 📚 Dataset Used

The dataset includes 3 CSV files:

- `training.csv`
- `test.csv`
- `validation.csv`

Each file contains:

- `text`: the user’s message
- `label`: an integer from 0 to 5 representing emotion

**Emotion Mapping**:

| Label | Emotion  |
|-------|----------|
| 0     | 😢 Sadness |
| 1     | 😊 Joy     |
| 2     | ❤️ Love    |
| 3     | 😠 Anger   |
| 4     | 😨 Fear    |
| 5     | 😲 Surprise |

---

## 🧠 Approach Summary

1. **Preprocessing**:  
   - Lowercasing  
   - Punctuation removal  
   - Extra whitespace cleanup

2. **Vectorization**:  
   - TF-IDF with 5000 features  
   - Removes English stopwords

3. **Model**:  
   - `LogisticRegression(max_iter=1000)` from scikit-learn  
   - Trained on the `training.csv` set  
   - Evaluated on the `test.csv` set

4. **Evaluation Metrics**:  
   - Accuracy score  
   - Classification report  
   - Confusion matrix (visualized via Plotly)

5. **Streamlit UI**:
   - Styled layout
   - Real-time predictions
   - Example sentence testing
   - Confidence bar display

---

## 🔧 Dependencies

Install using pip:

```bash
pip install -r requirements.txt
```
## Run
 
