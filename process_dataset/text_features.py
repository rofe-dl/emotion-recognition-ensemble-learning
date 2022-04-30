import csv
import os
import pandas as pd
import unicodedata
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle

def get_data():

    df = pd.read_csv('data/text_data_clean.csv')
    df.drop(df.loc[(df['Emotion'] == 'xxx') | (df['Emotion'] == 'dis') | (df['Emotion'] == 'oth') | (df['Emotion'] == 'fea') | (df['Emotion'] == 'sur') | (df['Emotion'] == 'fru')].index, inplace = True)
    df.loc[(df['Emotion'] == 'exc'), 'Emotion'] = 'hap'

    df = shuffle(df, random_state=42)

    x = df["Cleaned"]
    y = df["Emotion"]

    return x, y


def get_train_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    return x_train, x_test, y_train, y_test

def _extract_text():

    fields = ["Sentence", "Emotion", "Session", "Session"]
    with open("data/text_data.csv", "a") as f:
        write = csv.writer(f)
        write.writerow(fields)

    for i in range(1, 6):
        directory = f"data/IEMOCAP_dataset/Session{i}/dialog/transcriptions"
        directory1 = f"data/IEMOCAP_dataset/Session{i}/dialog/EmoEvaluation"

        emotions = []

        final = []
        transcripts = []
        ses_list_final = []
        emo_ses_list = []

        transcriptions = os.listdir(directory)
        transcriptions.sort()

        emoEvaluation = os.listdir(directory1)
        emoEvaluation.sort()


        for filename1 in emoEvaluation[3:]:
            file_path = directory1 + "/" + filename1
            file = open(file_path, "r")
            lines = file.readlines()

            for line in lines:
                if line[0] == "[":
                    emotions.append(line.split()[4])
                    emo_ses_list.append(line.split()[3])

        for filename in transcriptions:
            m_list = []
            f_list = []
            ses_list_f = []
            ses_list_m = []

            file_path = directory + "/" + filename
            file = open(file_path, "r")
            transcript_lines = file.readlines()

            for lines in transcript_lines:
                lines = lines.split(":")

                if len(lines[0]) > 10:

                    if lines[0][-24] == "F" and lines[0][-23] != "X":
                        f_list.append(lines[1][1:-1])
                        ses_list_f.append(lines[0])

                    elif lines[0][-24] == "M" and lines[0][-23] != "X":
                        m_list.append(lines[1][1:-1])
                        ses_list_m.append(lines[0])

            transcripts += f_list + m_list
            ses_list_final += ses_list_f + ses_list_m

        for i in range(len(transcripts)):
            final.append([transcripts[i], emotions[i],
                        ses_list_final[i], emo_ses_list[i]])

        rows = final

        with open("data/text_data.csv", "a") as f:
            write = csv.writer(f)
            write.writerows(rows)

def _unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def _normalizeString(s):
    s = _unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def make_text_features():
    _extract_text()

    data = pd.read_csv("data/text_data.csv")
    transcripts = data['Sentence'].tolist()

    for i in range(len(transcripts)):

        transcripts[i] = transcripts[i].replace("'", "")
        transcripts[i] = _normalizeString(transcripts[i])


    data["Cleaned"] = transcripts

    data.to_csv('data/text_data_clean.csv', index=False)

if __name__ == '__main__':
    make_text_features()