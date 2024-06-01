import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('questions.csv')
new_df = df.sample(1000, random_state=2)

# Feature Engineering
new_df['q1_len'] = new_df['question1'].str.len()
new_df['q2_len'] = new_df['question2'].str.len()
new_df['q1_num_words'] = new_df['question1'].apply(lambda row: len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row: len(row.split(" ")))

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
new_df['word_common'] = new_df.apply(common_words, axis=1)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))
new_df['word_total'] = new_df.apply(total_words, axis=1)

new_df['word_share'] = round(new_df['word_common'] / new_df['word_total'], 2)

def last_word_eq(row):
    return row['question1'].split()[-1].lower() == row['question2'].split()[-1].lower()

def first_word_eq(row):
    return row['question1'].split()[0].lower() == row['question2'].split()[0].lower()

def total_unique_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 | w2)

def total_char_length(row):
    return len(row['question1']) + len(row['question2'])

def common_word_ratio(row):
    total_words_in_both = len(row['question1'].split()) + len(row['question2'].split())
    return common_words(row) / total_words_in_both

new_df['cwc_min'] = new_df.apply(lambda row: common_words(row) / min(len(row['question1'].split()), len(row['question2'].split())), axis=1)
new_df['cwc_max'] = new_df.apply(lambda row: common_words(row) / max(len(row['question1'].split()), len(row['question2'].split())), axis=1)
new_df['last_word_eq'] = new_df.apply(lambda row: 1 if last_word_eq(row) else 0, axis=1)
new_df['first_word_eq'] = new_df.apply(lambda row: 1 if first_word_eq(row) else 0, axis=1)
new_df['mean_len'] = new_df.apply(lambda row: (len(row['question1'].split()) + len(row['question2'].split())) / 2, axis=1)
new_df['abs_len_diff'] = new_df.apply(lambda row: abs(len(row['question1'].split()) - len(row['question2'].split())), axis=1)
new_df['total_unique_words'] = new_df.apply(total_unique_words, axis=1)
new_df['total_char_length'] = new_df.apply(total_char_length, axis=1)
new_df['common_word_ratio'] = new_df.apply(common_word_ratio, axis=1)

ques_df = new_df[['question1', 'question2']]
final_df = new_df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])

questions = list(ques_df['question1']) + list(ques_df['question2'])
cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)

final_df = pd.concat([final_df, temp_df], axis=1)
# Train the model
X_train, X_test, y_train, y_test = train_test_split(final_df.iloc[:, 1:].values, final_df.iloc[:, 0].values, test_size=0.2, random_state=1)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Streamlit web app
st.title("Duplicate Question Detector")

# User input
st.header("User Input:")
q1 = st.text_input("Enter Question 1:")
q2 = st.text_input("Enter Question 2:")

# Prediction
if q1 and q2:
    # Feature Engineering for user input
    user_input = pd.DataFrame([[q1, q2]], columns=['question1', 'question2'])
    
    user_input['q1_len'] = user_input['question1'].str.len()
    user_input['q2_len'] = user_input['question2'].str.len()
    user_input['q1_num_words'] = user_input['question1'].apply(lambda row: len(row.split(" ")))
    user_input['q2_num_words'] = user_input['question2'].apply(lambda row: len(row.split(" ")))
    user_input['cwc_min'] = user_input.apply(lambda row: common_words(row) / min(len(row['question1'].split()), len(row['question2'].split())), axis=1)
    user_input['cwc_max'] = user_input.apply(lambda row: common_words(row) / max(len(row['question1'].split()), len(row['question2'].split())), axis=1)
    user_input['word_common'] = user_input.apply(common_words, axis=1)
    user_input['word_total'] = user_input.apply(total_words, axis=1)
    user_input['word_share'] = round(user_input['word_common'] / user_input['word_total'], 2)
    user_input['last_word_eq'] = user_input.apply(lambda row: 1 if last_word_eq(row) else 0, axis=1)
    user_input['first_word_eq'] = user_input.apply(lambda row: 1 if first_word_eq(row) else 0, axis=1)
    user_input['mean_len'] = user_input.apply(lambda row: (len(row['question1'].split()) + len(row['question2'].split())) / 2, axis=1)
    user_input['abs_len_diff'] = user_input.apply(lambda row: abs(len(row['question1'].split()) - len(row['question2'].split())), axis=1)
    user_input['total_unique_words'] = user_input.apply(total_unique_words, axis=1)
    user_input['total_char_length'] = user_input.apply(total_char_length, axis=1)
    user_input['common_word_ratio'] = user_input.apply(common_word_ratio, axis=1)

    user_questions = list(user_input['question1']) + list(user_input['question2'])
    user_q1_arr, user_q2_arr = np.vsplit(cv.transform(user_questions).toarray(), 2)
    user_temp_df1 = pd.DataFrame(user_q1_arr, index=user_input.index)
    user_temp_df2 = pd.DataFrame(user_q2_arr, index=user_input.index)
    user_temp_df = pd.concat([user_temp_df1, user_temp_df2], axis=1)
    user_final_df = pd.concat([user_input.drop(columns=['question1', 'question2']), user_temp_df], axis=1)

    user_final_df.columns = user_final_df.columns.astype(str)
    rf_prediction = rf.predict_proba(user_final_df)

    # Display result
    if rf_prediction[0][1] >= 0.5:
        st.error("These questions are duplicate.")
    else:
        st.success("These questions are not duplicate.")
else:
    st.info("Please enter both questions to make a prediction.")