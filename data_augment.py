import csv
import cv2
import numpy as np
import os
import shutil

data_dir = os.getcwd()
driving_file = 'driving_log.csv'


def multi_cam(folder, output_file, correction=0.2):
    with open('{0}/{1}/{2}'.format(data_dir, folder, output_file), 'w') as target_file:
        writer = csv.writer(target_file)
        with open('{0}/{1}/{2}'.format(data_dir, folder, driving_file), 'r') as log:
            reader = csv.reader(log)
            for line in reader:
                center_file = line[0]
                left_file = line[1]
                right_file = line[2]
                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                writer.writerow([center_file, str(steering_center)])
                writer.writerow([left_file, str(steering_left)])
                writer.writerow([right_file, str(steering_right)])


def flip(input_dir, output_dir):
    with open('{0}/{1}/{2}'.format(data_dir,output_dir,driving_file),'w') as target_file:
        writer = csv.writer(target_file)
        with open('{0}/{1}/{2}'.format(data_dir, input_dir, driving_file),'r') as log:
            reader = csv.reader(log)
            for line in reader:
                # 0 center, 1 left, 2 right 3 steering
                measurement = float(-1 * float(line[3]))
                #print(measurement)
                for i in range(0,3):
                    img_file = '{0}/{1}/IMG/{2}'.format(data_dir, input_dir, line[i])
                    img = cv2.imread(img_file)
                    cv2.imwrite('{0}/{1}/IMG/flipped_{2}'.format(data_dir, output_dir, line[i]),
                                cv2.flip(img, 1))
                new_img_files = ['flipped_{}'.format(line[i]) for i in range(0,3)]
                new_img_files.append(str(measurement))
                #print(new_img_files)
                writer.writerow(new_img_files)


def merge(source, target):
    with open('{0}/{1}/{2}'.format(data_dir,target,driving_file),'a') as target_file:
        writer = csv.writer(target_file)
        with open('{0}/{1}/{2}'.format(data_dir, source, driving_file),'r') as log:
            reader = csv.reader(log)
            for line in reader:
                writer.writerow(line)
    for file in os.listdir('{0}/{1}/IMG/'.format(data_dir, source)):
        shutil.copy2('{0}/{1}/IMG/{2}'.format(data_dir, source, file),
                     '{0}/{1}/IMG/'.format(data_dir, target))

if __name__ == '__main__':
   print()