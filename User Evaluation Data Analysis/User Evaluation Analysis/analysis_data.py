# User Evaluation Data analysis 자동화

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

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
        plt.title('Accuracy of each participant')
        plt.xlabel('Participant', labelpad=5, fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylabel('Accuracy(%)', fontdict={'family': 'serif', 'weight': 'bold', 'size': 12})
        plt.ylim([0, 100])
        colors = ['C2', 'dodgerblue', '#e35f62']

        plt.bar(x, [(int(accuracy) / 406) * 100 for accuracy in annotator[-1]], color = colors[annotator_count], width=0.5, label='Annotator' + str(annotator_count + 1))
        plt.xticks(x, ["P" + str(index + 1) for index in range(10)])
        plt.legend()
        plt.show()
        annotator_count += 1

get_data_csvfile() # csv 파일 데이터 추출
get_accuracy_annotators() # 각 평가자별 정확도





