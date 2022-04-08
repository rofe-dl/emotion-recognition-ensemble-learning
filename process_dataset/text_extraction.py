import csv
import os

def main():
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
                # print(lines)
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


        fields = ["Sentence", "Emotion", "Session", "Session"]
        rows = final

        with open("data/text_data.csv", "a") as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)

if __name__ == '__main__':
    main()