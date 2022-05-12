# User Evaluation Data analysis 자동화

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import makeheatmap

annotators_datasets = list() # Annotators 전체 데이터 담을 List

def get_data_csvfile():
    # csv 파일 데이터 추출
    csv_file_root_path = os.getcwd() + "\\Datasets\\"
    for dir_path in os.listdir(csv_file_root_path):  # Annotator Data 디렉토리

        csv_files = list()  # Annotator별 Data csv 파일

        for index in range(10):
            csv_file = open(csv_file_root_path + dir_path + "\\" + "Data" + "\\Data" + str(index + 1) + ".csv", "r",
                            encoding='utf-8')
            reader = csv.reader(csv_file)
            data = list()

            # Data(x).csv 파일 읽어오기
            for line in reader:
                data.append(line)

            # 10개의 Data csv 파일 저장
            csv_files.append(data)

        annotators_datasets.append(csv_files)  # Annotator별 10개의 Data csv 파일 저장

def get_summarize_low_data():
    emotion_labels = ["Happy", "Sad", "Surprise", "Angry", "Disgust", "Fear", "Neutral"]

    total_correct_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 라벨당 맞은 갯수 리스트
    total_incorrect_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 라벨당 틀린 갯수 리스트
    sum_sol_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 갯수 solution

    total_sol_label_list = list()  # solution 답안
    correct_label_accuracy_list = list()  # 각 라벨 맞은 갯수 리스트
    incorrect_label_accuracy_list = list()  # 각 라벨 오답 갯수 리스트

    # low Data 출력 [내가 선택한 emotions]
    annotator_count = 1
    for annotator in annotators_datasets:
        total_annotator_emotions_list = [0, 0, 0, 0, 0, 0, 0]

        for session in annotator:
            for emotion in range(len(session)):
                # csv 파일 헤더와 count 푸터 제거
                if emotion > 0 and emotion < len(session) - 1:
                    total_annotator_emotions_list[int(session[emotion][1]) - 1] += 1

        plt.title('Annotate Emotions [Annotator ' + str(annotator_count) + "]")
        plt.xlabel('Emotion label', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylabel('Annotate motion(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylim([0, 100])
        colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

        x = np.arange(len(emotion_labels))

        plt.bar(x, [(total_emotions / 4060) * 100 for total_emotions in total_annotator_emotions_list],
                color=colors, width=0.5)
        plt.xticks(x, emotion_labels)
        plt.show()
        annotator_count += 1

    annotator_count = 0
    for annotator in annotators_datasets:
        correct_annotator_emotions_list = [0, 0, 0, 0, 0, 0, 0]
        incorrect_label_list = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]  # 틀린 emotion 갯수 저장 list

        for session in annotator:
            for emotion in range(len(session)):
                # csv 파일 헤더와 count 푸터 제거
                if emotion > 0 and emotion < len(session) - 1:
                    anim_correct_answer = session[emotion][0].split("_")[-1]  # emotion 정답

                    if anim_correct_answer == session[emotion][1]:
                        correct_annotator_emotions_list[int(session[emotion][1]) - 1] += 1
                    else:
                        incorrect_label_list[int(anim_correct_answer) - 1][int(session[emotion][1]) - 1] += 1

        # 전체 데이터에서 맞은 감정 정확도
        plt.title('Correct Emotion [Annotator ' + str(annotator_count + 1) + "]")
        plt.xlabel('Emotion', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylim([0, 100])
        colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

        x = np.arange(len(emotion_labels))

        plt.bar(x, [(correct_accuary / 580) * 100 for correct_accuary in correct_annotator_emotions_list],
                color=colors,
                width=0.5)
        plt.xticks(x, emotion_labels)
        plt.show()
        annotator_count += 1




def get_accuracy_annotators():
    annotators_accuracy_list = list() # 각 맞춘 개수 list

    for annotator in annotators_datasets:
        accuracy_list = list()
        total_correct_count = 0 # 10개 파일에서 총 맞은 개수
        for data in annotator:
            accuracy_list.append(data[-1][-1]) # 맨 마지막 줄 마지막 열 출력 => count(정답 맞춘 개수)
            total_correct_count += int(data[-1][-1])

        annotator_list = [total_correct_count, accuracy_list] # 총 맞은 개수 , 10개 파일마다 맞은 개수
        annotators_accuracy_list.append(annotator_list)

    # 각 평가자별 정확도 표시
    plt.title('Total Accuracy')
    plt.xlabel('Annotator', labelpad = 5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylim([0, 100])
    colors = ['C2', 'dodgerblue', '#e35f62']

    x = np.arange(len(annotators_datasets))

    plt.bar(x, [(total_accuracy[0] / 4060) * 100 for total_accuracy in annotators_accuracy_list], color=colors, width=0.5)
    plt.xticks(x, ["Annotator" + str(index+1) for index in range(len(annotators_accuracy_list))])
    plt.show()

    # 각 평가자 피험자별로 정확도 표시
    x = np.arange(10)

    annotator_count = 0
    for annotator in annotators_accuracy_list:
        plt.title('Accuracy of each session')
        plt.xlabel('Session', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylim([0, 100])
        colors = ['C2', 'dodgerblue', '#e35f62']

        plt.bar(x, [(int(accuracy) / 406) * 100 for accuracy in annotator[-1]], color = colors[annotator_count], width=0.5, label='Annotator' + str(annotator_count + 1))
        plt.xticks(x, ["S" + str(index + 1) for index in range(10)])
        plt.legend()
        plt.show()
        annotator_count += 1

    # 피험자별로 정확도 표시
    x = np.arange(len(annotators_accuracy_list))

    for participant in range(len(annotators_datasets[0])):
        plt.title('Accuracy of each session [S' + str(participant + 1) + ']')
        plt.xlabel('Annotator', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylim([0, 100])
        colors = ['C2', 'dodgerblue', '#e35f62']

        plt.bar(x, [(int(annotator_accuracy[1][participant]) / 406) * 100 for annotator_accuracy in annotators_accuracy_list], color=colors,
                width=0.5)
        plt.xticks(x, ["Annotator" + str(index+1) for index in range(len(annotators_accuracy_list))])
        plt.show()


def get_accuracy_labels():
    emotion_labels = ["Happy", "Sad", "Surprise", "Angry", "Disgust", "Fear", "Neutral"]

    total_correct_label_list = [0, 0, 0, 0, 0, 0, 0] # 총 라벨당 맞은 갯수 리스트
    total_incorrect_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 라벨당 틀린 갯수 리스트
    sum_sol_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 갯수 solution

    total_sol_label_list = list() # solution 답안
    correct_label_accuracy_list = list() # 각 라벨 맞은 갯수 리스트
    incorrect_label_accuracy_list = list()  # 각 라벨 오답 갯수 리스트


    for participant in range(len(annotators_datasets[0])): # 피험자 (1번  - 10번)
        correct_emotion_dict = {"Happy": 0, "Sad": 0, "Surprise": 0, "Angry": 0, "Disgust": 0, "Fear": 0, "Neutral": 0} # 세명 모두 맞은 emotion 저장 dictionary
        incorrect_label_list = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]] # 세명 모두 틀린 emotion 갯수 저장 list
        sol_label_list = [0, 0, 0, 0, 0, 0, 0]


        emotion_label_list = list(correct_emotion_dict.keys())

        for data in range(len(annotators_datasets[0][participant])): # 406개 데이터

            #csv 파일 헤더와 count 푸터 제거
            if data > 0 and data < len(annotators_datasets[0][participant]) - 1:

                anim_correct_answer = annotators_datasets[0][participant][data][0].split("_")[-1]  # emotion 정답
                sol_label_list[int(anim_correct_answer) - 1] += 1  # 정답 개수 세기

                if (annotators_datasets[0][participant][data][1] == annotators_datasets[1][participant][data][1]) and (annotators_datasets[1][participant][data][1] == annotators_datasets[2][participant][data][1]):

                    if anim_correct_answer == annotators_datasets[0][participant][data][1]:
                        correct_emotion_dict[emotion_label_list[int(anim_correct_answer) - 1]] += 1
                    else:
                        incorrect_label_list[int(anim_correct_answer) - 1][int(annotators_datasets[0][participant][data][1]) - 1] += 1 # 오답 체크

        correct_label_accuracy_list.append(list(correct_emotion_dict.values()))
        incorrect_label_accuracy_list.append(incorrect_label_list)
        total_sol_label_list.append(sol_label_list)

    for accuracy_index in range(len(correct_label_accuracy_list)):

        for emotion_index in range(len(emotion_labels)):
            total_correct_label_list[emotion_index] += correct_label_accuracy_list[accuracy_index][emotion_index]

            for incorrect_index in range(len(incorrect_label_accuracy_list[accuracy_index][emotion_index])):
                total_incorrect_label_list[incorrect_index] += incorrect_label_accuracy_list[accuracy_index][emotion_index][incorrect_index]

            sum_sol_label_list[emotion_index] += total_sol_label_list[accuracy_index][emotion_index]

    total_correct_count = 0
    total_incorrect_count = 0
    for emotion_index in range(len(emotion_labels)):
        total_correct_count += total_correct_label_list[emotion_index]
        total_incorrect_count += total_incorrect_label_list[emotion_index]

    # 전체 데이터에서 세명 다 맞은 감정 정확도
    plt.title('Correct Emotion (All data)')
    plt.xlabel('Emotion', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylim([0, 100])
    colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

    x = np.arange(len(emotion_labels))

    plt.bar(x, [(total_correct_accuracy / 580 ) * 100 for total_correct_accuracy in total_correct_label_list], color=colors,
            width=0.5)
    plt.xticks(x, emotion_labels)
    plt.show()

    # 전체 데이터에서 세명 다 틀린 감정 정확도
    plt.title('Incorrect Emotion (All data)')
    plt.xlabel('Emotion', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylim([0, 100])
    colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

    x = np.arange(len(emotion_labels))

    plt.bar(x, [(total_incorrect_accuracy / 580) * 100 for total_incorrect_accuracy in
                total_incorrect_label_list], color=colors,
            width=0.5)
    plt.xticks(x, emotion_labels)
    plt.show()

    # 전체 데이터에서 세명 같은 답 히트맵
    same_dataset = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            if i == j:
                same_dataset[i][j] = total_correct_label_list[i]
            else:
                for incorrect_index in range(len(incorrect_label_accuracy_list)):
                    same_dataset[i][j] += incorrect_label_accuracy_list[incorrect_index][i][j]

    fig, ax = plt.subplots()

    im, cbar = makeheatmap.heatmap(same_dataset, emotion_labels, emotion_labels, ax=ax,
                       cmap="Reds", vmin=0, vmax=580, cbarlabel="Annotate Emotion") # Heatmap
    texts = makeheatmap.annotate_heatmap(im, valfmt="{x:d}")

    ax.set_title("Annotate Emotion (All data)")
    ax.set_xlabel("Annotated Emotion Label")
    ax.set_ylabel("Answer Emotion Label")

    fig.tight_layout()
    plt.show()

    # 각 피험자별 세명 같은 답 히트맵
    for participant in range(len(annotators_datasets[0])):
        participant_dataset = np.array([[0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0]])

        for i in range(len(emotion_labels)):
            for j in range(len(emotion_labels)):
                if i == j:
                    participant_dataset[i][j] = correct_label_accuracy_list[participant][i]
                else:
                    participant_dataset[i][j] = incorrect_label_accuracy_list[participant][i][j]


        fig, ax = plt.subplots()

        im, cbar = makeheatmap.heatmap(participant_dataset, emotion_labels, emotion_labels, ax=ax,
                                       cmap="Reds", vmin=0, vmax=70, cbarlabel="Annotate Emotion")  # Heatmap
        texts = makeheatmap.annotate_heatmap(im, valfmt="{x:d}")

        ax.set_title("Annotate Emotion (Session" + str(participant + 1) + " data)")
        ax.set_xlabel("Annotated Emotion Label")
        ax.set_ylabel("Answer Emotion Label")

        fig.tight_layout()
        plt.show()

def get_accuracy_labels_more_than_tp():
    emotion_labels = ["Happy", "Sad", "Surprise", "Angry", "Disgust", "Fear", "Neutral"]

    total_correct_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 라벨당 맞은 갯수 리스트
    total_incorrect_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 라벨당 틀린 갯수 리스트
    sum_sol_label_list = [0, 0, 0, 0, 0, 0, 0]  # 총 갯수 solution

    total_sol_label_list = list()  # solution 답안
    correct_label_accuracy_list = list()  # 각 라벨 맞은 갯수 리스트
    incorrect_label_accuracy_list = list()  # 각 라벨 오답 갯수 리스트

    for participant in range(len(annotators_datasets[0])):  # 피험자 (1번  - 10번)

        correct_emotion_dict = {"Happy": 0, "Sad": 0, "Surprise": 0, "Angry": 0, "Disgust": 0, "Fear": 0,
                                "Neutral": 0}  # 두명 이상 맞은 emotion 저장 dictionary
        incorrect_label_list = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]  # 두명 이상 틀린 emotion 갯수 저장 list
        sol_label_list = [0, 0, 0, 0, 0, 0, 0]

        emotion_label_list = list(correct_emotion_dict.keys())

        for data in range(len(annotators_datasets[0][participant])):  # 406개 데이터

            # csv 파일 헤더와 count 푸터 제거
            if data > 0 and data < len(annotators_datasets[0][participant]) - 1:

                anim_correct_answer = annotators_datasets[0][participant][data][0].split("_")[-1]  # emotion 정답
                sol_label_list[int(anim_correct_answer) - 1] += 1  # 정답 개수 세기

                temp_annotator_data_list = [annotators_datasets[0][participant][data][1], annotators_datasets[1][participant][data][1], annotators_datasets[2][participant][data][1]]
                temp_annotator_data_set = set(temp_annotator_data_list)

                if len(temp_annotator_data_set) < 3:
                    temp_annotator_data_set = list(temp_annotator_data_set)

                    for temp_annotator_data in temp_annotator_data_set: # 중복 제거
                        temp_annotator_data_list.remove(temp_annotator_data)

                    if anim_correct_answer == temp_annotator_data_list[0]:
                        correct_emotion_dict[emotion_label_list[int(anim_correct_answer) - 1]] += 1
                    else:
                        incorrect_label_list[int(anim_correct_answer) - 1][
                            int(annotators_datasets[0][participant][data][1]) - 1] += 1  # 오답 체크

        correct_label_accuracy_list.append(list(correct_emotion_dict.values()))
        incorrect_label_accuracy_list.append(incorrect_label_list)
        total_sol_label_list.append(sol_label_list)

    for accuracy_index in range(len(correct_label_accuracy_list)):

        for emotion_index in range(len(emotion_labels)):
            total_correct_label_list[emotion_index] += correct_label_accuracy_list[accuracy_index][emotion_index]

            for incorrect_index in range(len(incorrect_label_accuracy_list[accuracy_index][emotion_index])):
                total_incorrect_label_list[incorrect_index] += \
                incorrect_label_accuracy_list[accuracy_index][emotion_index][incorrect_index]

            sum_sol_label_list[emotion_index] += total_sol_label_list[accuracy_index][emotion_index]

    total_correct_count = 0
    total_incorrect_count = 0
    for emotion_index in range(len(emotion_labels)):
        total_correct_count += total_correct_label_list[emotion_index]
        total_incorrect_count += total_incorrect_label_list[emotion_index]


    # 전체 데이터에서 두명 이상 맞은 감정 정확도
    plt.title('Correct Emotion(N≥2) (All data)')
    plt.xlabel('Emotion', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylim([0, 100])
    colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

    x = np.arange(len(emotion_labels))


    plt.bar(x, [(total_correct_accuracy / 580) * 100 for total_correct_accuracy in
                total_correct_label_list], color=colors,
            width=0.5)
    plt.xticks(x, emotion_labels)
    plt.show()

    # 전체 데이터에서 두명 이상 틀린 감정 정확도
    plt.title('Incorrect Emotion(N≥2) (All data)')
    plt.xlabel('Emotion', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
    plt.ylim([0, 100])
    colors = ['C2', 'dodgerblue', '#f1c40f', '#e35f62', '#e67e22', '#9b59b6', '#34495e']

    x = np.arange(len(emotion_labels))

    plt.bar(x, [(total_incorrect_accuracy / 580) * 100 for total_incorrect_accuracy in
                total_incorrect_label_list], color=colors,
            width=0.5)
    plt.xticks(x, emotion_labels)
    plt.show()

    # 전체 데이터에서 두명 이상 같은 답 히트맵
    same_dataset = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            if i == j:
                same_dataset[i][j] = total_correct_label_list[i]
            else:
                for incorrect_index in range(len(incorrect_label_accuracy_list)):
                    same_dataset[i][j] += incorrect_label_accuracy_list[incorrect_index][i][j]

    fig, ax = plt.subplots()

    im, cbar = makeheatmap.heatmap(same_dataset, emotion_labels, emotion_labels, ax=ax,
                                   cmap="Reds", vmin=0, vmax=580, cbarlabel="Annotate Emotion")  # Heatmap
    texts = makeheatmap.annotate_heatmap(im, valfmt="{x:d}")

    ax.set_title("Annotate Emotion(N≥2) (All data)")
    ax.set_xlabel("Annotated Emotion Label")
    ax.set_ylabel("Answer Emotion Label")

    fig.tight_layout()
    plt.show()

    # 각 피험자별 두명 이상 같은 답 히트맵
    for participant in range(len(annotators_datasets[0])):
        participant_dataset = np.array([[0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0]])

        for i in range(len(emotion_labels)):
            for j in range(len(emotion_labels)):
                if i == j:
                    participant_dataset[i][j] = correct_label_accuracy_list[participant][i]
                else:
                    participant_dataset[i][j] = incorrect_label_accuracy_list[participant][i][j]

        fig, ax = plt.subplots()

        im, cbar = makeheatmap.heatmap(participant_dataset, emotion_labels, emotion_labels, ax=ax,
                                       cmap="Reds", vmin=0, vmax=70, cbarlabel="Annotate Emotion")  # Heatmap
        texts = makeheatmap.annotate_heatmap(im, valfmt="{x:d}")

        ax.set_title("Annotate Emotion(N≥2) (session" + str(participant + 1) + " data)")
        ax.set_xlabel("Annotated Emotion Label")
        ax.set_ylabel("Answer Emotion Label")

        fig.tight_layout()
        plt.show()

get_data_csvfile() # csv 파일 데이터 추출
get_summarize_low_data() # low 데이터 정리
get_accuracy_annotators() # 각 평가자별 정확도
get_accuracy_labels() # 각 Label별 정확도 (세명 다 같은 답)
get_accuracy_labels_more_than_tp() # 각 Label별 정확도 (두명이상 같은 답)




