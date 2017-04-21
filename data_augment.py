import csv
import cv2
import os
import shutil

# default data set folders and files
# raw drive data is the folder to keep the original recorded driving log
data_dir = os.getcwd()
raw_data_folder = '{}/raw_drive_data'.format(data_dir)
flipped_data_folder = '{}/flipped_drive_data'.format(data_dir)
final_data_folder = '{}/final_data'.format(data_dir)
driving_file = 'driving_log.csv'
multi_cam_file = 'multi_cam.csv'


# strip off path and flatten multi cam data sets
def multi_cam(folder, output_file, correction=0.2):
    with open('{0}/{1}'.format(folder, output_file), 'w') as target_file:
        writer = csv.writer(target_file)
        with open('{0}/{1}'.format(folder, driving_file), 'r') as log:
            reader = csv.reader(log)
            for line in reader:
                center_file = line[0].split('/')[-1]
                left_file = line[1].split('/')[-1]
                right_file = line[2].split('/')[-1]
                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                writer.writerow([center_file, str(steering_center)])
                writer.writerow([left_file, str(steering_left)])
                writer.writerow([right_file, str(steering_right)])


# flip images to create more data set
def flip(input_dir, input_log_file, output_dir, output_log_file):
    with open('{0}/{1}'.format(output_dir, output_log_file),'w') as target_file:
        writer = csv.writer(target_file)
        with open('{0}/{1}'.format(input_dir, input_log_file),'r') as log:
            reader = csv.reader(log)
            for line in reader:
                # 0 center, 1 left, 2 right 3 steering
                measurement = float(-1 * float(line[1]))
                img_file = '{0}/IMG/{1}'.format(input_dir, line[0])
                img = cv2.imread(img_file)
                cv2.imwrite('{0}/IMG/flipped_{1}'.format(output_dir, line[0]), cv2.flip(img, 1))
                new_img_file = 'flipped_{}'.format(line[0])
                writer.writerow([new_img_file, measurement])


def merge(source_folder,source_file, target_folder, target_file):
    with open('{0}/{1}'.format(target_folder,target_file),'a') as target_file_writer:
        writer = csv.writer(target_file_writer)
        with open('{0}/{1}'.format(source_folder, source_file),'r') as log:
            reader = csv.reader(log)
            for line in reader:
                writer.writerow(line)
    for file in os.listdir('{0}/IMG/'.format(source_folder)):
        shutil.copy2('{0}/IMG/{1}'.format(source_folder, file),
                     '{0}/IMG/'.format(target_folder))

if __name__ == '__main__':
    # re-create data folders if not exist
    if os.path.exists(flipped_data_folder):
        shutil.rmtree(flipped_data_folder)
    if os.path.exists(final_data_folder):
        shutil.rmtree(final_data_folder)
    os.makedirs(flipped_data_folder + '/IMG')
    os.makedirs(final_data_folder + '/IMG')

    multi_cam(raw_data_folder, multi_cam_file)
    flip(raw_data_folder, multi_cam_file, flipped_data_folder, driving_file)
    merge(raw_data_folder, multi_cam_file, final_data_folder, driving_file)
    merge(flipped_data_folder, driving_file, final_data_folder, driving_file)
