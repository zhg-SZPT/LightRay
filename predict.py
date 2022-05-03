# -----------------------------------------------------------------------#
#  predict.py integrates functions such as single image prediction,
#  camera detection, FPS test, and directory traversal detection
# -----------------------------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO



if __name__ == "__main__":

    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the mode of the test:
    #   'predict' represents a single image prediction;
    #   'video' represents video detection, you can call the camera or video for detection;
    #   'fps' represents test fps;
    #   'dir_predict' represents that traverse the folder to detect and save.
    # ----------------------------------------------------------------------------------------------------------#
    mode = "fps"
    #   video_path = 0 表示检测摄像头
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    test_interval = 1000
    # -------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path of the image used for detection
    #   dir_save_path specifies the save path of the detected image
    #   dir_origin_path and dir_save_path are only available when mode='dir_predict'
    # -------------------------------------------------------------------------#
    dir_origin_path = "xiaomubiao-in/"
    dir_save_path = "xiaomubiao-out/"

    if mode == "predict":
        import os
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()

            ref, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #  BGR to RGB
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 1000
        img = Image.open('img_in/P00603.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'PS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        # for img_name in tqdm(img_names,bar_format='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
        #       '{rate_fmt}{postfix}]\n'):
        time_start = time.time()
        with tqdm(img_names, bar_format='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
                                        '{rate_fmt}{postfix}]\n') as t:
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_origin_path, img_name)
                    image = Image.open(image_path)
                    r_image = yolo.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name))
                    BS = t.format_dict['total'] / t.format_dict['elapsed']  # How many pictures to predict per second
        print("\n")
        print("Batch_size/S = ",BS)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
    time_end=time.time()
    print('totally cost = ',time_end-time_start)