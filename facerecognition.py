# from sklearn.metrics.pairwise import pairwise_distances
# from tensorflow.python.platform import gfile
# import tensorflow as tf
# import numpy as np
# import detect_and_align
# import argparse
# import easygui
# import time
# import cv2
# import os
# import pygame
#
# tf.compat.v1.disable_v2_behavior()
#
# class IdData:
#     """Keeps track of known identities and calculates id matches"""
#
#     def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold):
#         print("Loading known identities: ", end="")
#         self.distance_treshold = distance_treshold
#         self.id_folder = id_folder
#         self.mtcnn = mtcnn
#         self.id_names = []
#         self.embeddings = None
#
#         image_paths = []
#         os.makedirs(id_folder, exist_ok=True)
#         ids = os.listdir(os.path.expanduser(id_folder))
#         if not ids:
#             return
#
#         for id_name in ids:
#             id_dir = os.path.join(id_folder, id_name)
#             image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]
#
#         print("Found %d images in id folder" % len(image_paths))
#         aligned_images, id_image_paths = self.detect_id_faces(image_paths)
#         feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
#         self.embeddings = sess.run(embeddings, feed_dict=feed_dict)
#
#         if len(id_image_paths) < 5:
#             self.print_distance_table(id_image_paths)
#
#     def add_id(self, embedding, new_id, face_patch):
#         if self.embeddings is None:
#             self.embeddings = np.atleast_2d(embedding)
#         else:
#             self.embeddings = np.vstack([self.embeddings, embedding])
#         self.id_names.append(new_id)
#         id_folder = os.path.join(self.id_folder, new_id)
#         os.makedirs(id_folder, exist_ok=True)
#         filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
#         numbered_filenames = [int(f) for f in filenames if f.isdigit()]
#         img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
#         cv2.imwrite(os.path.join(id_folder, f"{img_number}.jpg"), face_patch)
#
#     def detect_id_faces(self, image_paths):
#         aligned_images = []
#         id_image_paths = []
#         for image_path in image_paths:
#             image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
#             if len(face_patches) > 1:
#                 print(
#                     "Warning: Found multiple faces in id image: %s" % image_path
#                     + "\nMake sure to only have one face in the id images. "
#                     + "If that's the case then it's a false positive detection and"
#                     + " you can solve it by increasing the thresolds of the cascade network"
#                 )
#             aligned_images = aligned_images + face_patches
#             id_image_paths += [image_path] * len(face_patches)
#             path = os.path.dirname(image_path)
#             self.id_names += [os.path.basename(path)] * len(face_patches)
#
#         return np.stack(aligned_images), id_image_paths
#
#     def print_distance_table(self, id_image_paths):
#         """Prints distances between id embeddings"""
#         distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
#         image_names = [path.split("/")[-1] for path in id_image_paths]
#         print("Distance matrix:\n{:20}".format(""), end="")
#         [print("{:20}".format(name), end="") for name in image_names]
#         for path, distance_row in zip(image_names, distance_matrix):
#             print("\n{:20}".format(path), end="")
#             for distance in distance_row:
#                 print("{:20}".format("%0.3f" % distance), end="")
#         print()
#
#     def find_matching_ids(self, embs):
#         if self.id_names:
#             matching_ids = []
#             matching_distances = []
#             distance_matrix = pairwise_distances(embs, self.embeddings)
#             for distance_row in distance_matrix:
#                 min_index = np.argmin(distance_row)
#                 if distance_row[min_index] < self.distance_treshold:
#                     matching_ids.append(self.id_names[min_index])
#                     matching_distances.append(distance_row[min_index])
#                 else:
#                     matching_ids.append(None)
#                     matching_distances.append(None)
#         else:
#             matching_ids = [None] * len(embs)
#             matching_distances = [np.inf] * len(embs)
#         return matching_ids, matching_distances
#
#
# def load_model(model):
#     model_exp = os.path.expanduser(model)
#     if os.path.isfile(model_exp):
#         print("Loading model filename: %s" % model_exp)
#         with gfile.FastGFile(model_exp, "rb") as f:
#             graph_def = tf.compat.v1.GraphDef()
#             graph_def.ParseFromString(f.read())
#             tf.compat.v1.import_graph_def(graph_def, name="")
#     else:
#         raise ValueError("Specify model file, not directory!")
#
# def play_alarm_sound(alarm_file):
#     pygame.mixer.init()
#     pygame.mixer.music.load(alarm_file)
#     pygame.mixer.music.play()
#
#
# def face_recognition(args, alarm_file):
#     with tf.Graph().as_default():
#         with tf.compat.v1.Session() as sess:
#
#             # Setup models
#             mtcnn = detect_and_align.create_mtcnn(sess, None)
#
#             load_model(args.model)
#             images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
#
#             # Load anchor IDs
#             id_data = IdData(
#                 args.id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, args.threshold
#             )
#
#             cap = cv2.VideoCapture(0)
#             frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#             show_landmarks = False
#             show_bb = False
#             show_id = True
#             show_fps = False
#             frame_detections = None
#
#             recognition_results = []
#
#             while True:
#                 start = time.time()
#                 _, frame = cap.read()
#
#                 # Locate faces and landmarks in frame
#                 face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)
#
#                 if len(face_patches) > 0:
#                     face_patches = np.stack(face_patches)
#                     feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
#                     embs = sess.run(embeddings, feed_dict=feed_dict)
#
#                     matching_ids, matching_distances = id_data.find_matching_ids(embs)
#                     frame_detections = {"embs": embs, "bbs": padded_bounding_boxes, "frame": frame.copy()}
#
#                     print("Matches in frame:")
#                     for bb, landmark, matching_id, dist in zip(
#                         padded_bounding_boxes, landmarks, matching_ids, matching_distances
#                     ):
#                         if matching_id is None:
#                             matching_id = "Unknown"
#                             print("Unknown! Couldn't find match.")
#                         else:
#                             print("Hi %s! Distance: %1.4f" % (matching_id, dist))
#
#                         if show_id:
#                             font = cv2.FONT_HERSHEY_SIMPLEX
#                             cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#                         if show_bb:
#                             cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
#                         if show_landmarks:
#                             for j in range(5):
#                                 size = 1
#                                 top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
#                                 bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
#                                 cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
#                 else:
#                     print("Couldn't find a face")
#                     matching_ids = ["Unknown"]
#
#                 recognition_results.append(matching_ids[0])
#
#                 if len(recognition_results) >= 20:
#                     unknown_count = recognition_results[-20:].count("Unknown")
#                     if unknown_count > 16:
#                         play_alarm_sound(alarm_file)
#
#                 end = time.time()
#
#                 seconds = end - start
#                 fps = round(1 / seconds, 2)
#
#                 if show_fps:
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     cv2.putText(frame, str(fps), (0, int(frame_height) - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#
#                 cv2.imshow("frame", frame)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break
#
#             cap.release()
#             cv2.destroyAllWindows()
# class FaceRecognitionSystem:
#     def __init__(self, model_path, id_folder, threshold, alarm_file):
#         self.model_path = model_path
#         self.id_folder = id_folder
#         self.threshold = threshold
#         self.alarm_file = alarm_file
#
#     def run(self):
#         args = argparse.Namespace(model=self.model_path, id_folder=[self.id_folder], threshold=self.threshold)
#         face_recognition(args, self.alarm_file)
#
# if __name__ == "__main__":
#     frs = FaceRecognitionSystem("20170512-110547.pb", "ids", 2.0, "siren.mp3")
#     frs.run()
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import detect_and_align
import argparse
import easygui
import time
import pygame
import sys
tf.compat.v1.disable_v2_behavior()

name_dict = {
    "ids中的文件名": "中文姓名",

}

recognition_results = []
class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        image_paths = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        if self.embeddings is None:
            self.embeddings = np.atleast_2d(embedding)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_names.append(new_id)
        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)
        filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
        numbered_filenames = [int(f) for f in filenames if f.isdigit()]
        img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
        cv2.imwrite(os.path.join(id_folder, f"{img_number}.jpg"), face_patch)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < 0.8:
                # if distance_row[min_index] < 1:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.compat.v1.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

def play_alarm_sound(alarm_file):
    pygame.mixer.init()
    pygame.mixer.music.load(alarm_file)
    pygame.mixer.music.play()

# def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
#     if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text(position, text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def face_recognition(args, alarm_file):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:

            # Setup models
            mtcnn = detect_and_align.create_mtcnn(sess, None)

            load_model(args.model)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load anchor IDs
            id_data = IdData(
                args.id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, args.threshold
            )

            cap = cv2.VideoCapture(0)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            show_landmarks = False
            show_bb = True
            show_id = True
            show_fps = False
            frame_detections = None
            count = 0
            unknow =0

            while True:
                start = time.time()
                _, frame = cap.read()

                # Locate faces and landmarks in frame
                face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)

                    matching_ids, matching_distances = id_data.find_matching_ids(embs)
                    frame_detections = {"embs": embs, "bbs": padded_bounding_boxes, "frame": frame.copy()}

                    print("Matches in frame:")
                    for bb, landmark, matching_id, dist in zip(padded_bounding_boxes, landmarks, matching_ids, matching_distances):
                        if matching_id is None:
                            matching_id = "Unknown"
                            count += 1
                            unknow += 1
                            # recognition_results.append(matching_id)
                            print("Unknown! Couldn't find match.")
                        else:
                            print("Hi %s! Distance: %1.4f" % (matching_id, dist))
                            count += 1
                            # recognition_results.append(matching_id)

                        if show_id:
                            if matching_id == "Unknown":
                                frame = cv2AddChineseText(frame, "未知用户", (bb[0], bb[3]), (255, 255, 255), 30)
                            else:
                                # 使用字典查找对应的中文名字
                                name = name_dict.get(matching_id)
                                frame = cv2AddChineseText(frame, name, (bb[0], bb[3]), (255, 255, 255), 30)
                        if show_bb:
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        if show_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
                                bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                        # if len(recognition_results) >= 40:
                        #     unknown_count = recognition_results[-30:].count("Unknown")
                        #     if unknown_count > 10:
                        #         play_alarm_sound("siren.mp3")
                    # if count >= 30:
                    #     p = unknow/count
                    #     if p > 0.5:
                    #         play_alarm_sound("siren.mp3")
                    #     count = 0
                    #     unknow = 0


                else:
                 print("Couldn't find a face")
                                    # matching_ids = ["Unknown"]


                # Display FPS
                end = time.time()
                seconds = end - start
                fps = round(1 / seconds, 2)
                if show_fps:
                    frame = cv2.putText(
                        frame,
                        f"FPS: {fps}",
                        (10, int(frame_height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
            # if alarm_file and "Unknown" in recognition_results:
            #     play_alarm_sound(alarm_file)

#

def main(args):
    alarm_file = "警报声音文件.mp3"
    face_recognition(args, alarm_file)

# def parse_arg
class FaceRecognitionSystem:
    def __init__(self, model_path, id_folder, threshold, alarm_file):
        self.model_path = model_path
        self.id_folder = id_folder
        self.threshold = threshold
        self.alarm_file = alarm_file

    def run(self):
        args = argparse.Namespace(model=self.model_path, id_folder=[self.id_folder], threshold=self.threshold)
        face_recognition(args, self.alarm_file)



if __name__ == "__main__":
    frs = FaceRecognitionSystem("20170512-110547.pb", "ids", 2.0, "siren.mp3")
    frs.run()