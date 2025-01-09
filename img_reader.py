import cv2
import os
from tqdm import tqdm



def merge_image_to_video(folder_name, vedio_name):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    img_size = (1280,720)
    video = cv2.VideoWriter(vedio_name, fourcc, fps, img_size)
    for num in tqdm(range(len(os.listdir(folder_name)))):
        filename = folder_name + '/RGB_image' + str(num) + '.png'
        frame = cv2.imread(filename)
        video.write(frame)
    video.release()
    

if __name__ == "__main__":
    # file = cv2.imread('./RGB_cam/RGB_image0.png')
    # print(file.shape)
    
    image_folder = './RGB_cam'
    video_name = 'RGB_vedio.mp4'
    merge_image_to_video(image_folder,video_name)
    