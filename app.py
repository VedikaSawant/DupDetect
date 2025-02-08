import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
df1 = pd.read_csv('sample_ques1.csv')
# Read the subsequent CSV files, skipping the header row
df2 = pd.read_csv('sample_ques2.csv', skiprows=1, header=None)
df3 = pd.read_csv('sample_ques3.csv', skiprows=1, header=None)

# Assign column names to the subsequent DataFrames
df2.columns = df1.columns
df3.columns = df1.columns

df = pd.concat([df1, df2, df3], ignore_index=True)
new_df = df.sample(1000, random_state=2)

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))

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

# Feature Engineering
def add_features(input_df):
    input_df['q1_len'] = input_df['question1'].str.len()
    input_df['q2_len'] = input_df['question2'].str.len()
    input_df['q1_num_words'] = input_df['question1'].apply(lambda row: len(row.split(" ")))
    input_df['q2_num_words'] = input_df['question2'].apply(lambda row: len(row.split(" ")))
    input_df['word_common'] = input_df.apply(common_words, axis=1)
    input_df['word_total'] = input_df.apply(total_words, axis=1)
    input_df['word_share'] = round(input_df['word_common'] / input_df['word_total'], 2)
    input_df['cwc_min'] = input_df.apply(lambda row: common_words(row) / min(len(row['question1'].split()), len(row['question2'].split())), axis=1)
    input_df['cwc_max'] = input_df.apply(lambda row: common_words(row) / max(len(row['question1'].split()), len(row['question2'].split())), axis=1)
    input_df['last_word_eq'] = input_df.apply(lambda row: 1 if last_word_eq(row) else 0, axis=1)
    input_df['first_word_eq'] = input_df.apply(lambda row: 1 if first_word_eq(row) else 0, axis=1)
    input_df['mean_len'] = input_df.apply(lambda row: (len(row['question1'].split()) + len(row['question2'].split())) / 2, axis=1)
    input_df['abs_len_diff'] = input_df.apply(lambda row: abs(len(row['question1'].split()) - len(row['question2'].split())), axis=1)
    input_df['total_unique_words'] = input_df.apply(total_unique_words, axis=1)
    input_df['total_char_length'] = input_df.apply(total_char_length, axis=1)
    input_df['common_word_ratio'] = input_df.apply(common_word_ratio, axis=1)
    input_df['levenshtein_distance'] = input_df.apply(lambda row: levenshtein_distance(row['question1'], row['question2']), axis=1)
    return input_df

new_df = add_features(new_df)

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
st.title("Question Similarity Detector")

# User input
st.header("User Input:")
q1 = st.text_input("Enter Question 1:", placeholder="eg. What is the capital of India?")
q2 = st.text_input("Enter Question 2:", placeholder="eg. Where is the capital of India?")

# Prediction
if st.button('Predict') and (q1 and q2):
    # Feature Engineering for user input
    user_input = pd.DataFrame([[q1, q2]], columns=['question1', 'question2'])
    
    user_input = add_features(user_input)

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
        st.error("These questions have the same meaning.")
    else:
        st.success("These questions do not have the same meaning.")
elif not q1 or not q2:
    st.info("Please enter both questions to make a prediction.")
