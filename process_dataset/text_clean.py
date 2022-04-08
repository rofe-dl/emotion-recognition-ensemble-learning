import unicodedata
import re
import csv
from pandas import read_csv
import re 

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

data = read_csv("data/text_data.csv")
transcripts = data['Sentence'].tolist()

for i in range(len(transcripts)):

  transcripts[i] = transcripts[i].replace("'", "")
  transcripts[i] = normalizeString(transcripts[i])


data["Cleaned"] = transcripts

data.to_csv('data/text_data_clean.csv', index=False)