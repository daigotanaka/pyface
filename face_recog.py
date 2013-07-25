# The MIT License (MIT)
# 
# Copyright (c) 2013 Daigo Tanaka
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import csv
import os
import sys
import time
sys.path.append("/home/pi/opencv/release/lib")

import numpy as np
import cv2


class FaceRecog(object):

    def __init__(
        self,
        cascade_file,
        video_src,
        width=320,
        height=240,
        detect_freq=1.0,
        max_distance=60,
        min_distance=60,
        buffer_size=10,
        display=False):

        if not os.path.exists(cascade_file):
            raise Exception("Cascade file not found")

        self.cascade = cv2.CascadeClassifier(cascade_file)
        self.video_src = video_src
        self.proc_size_w = 80
        self.proc_size_h = 80
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.detect_freq = detect_freq
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.display = display

        self.buffer = self.buffer_size * [None]

        self.cam = None
        self.model = None
        self.last_capture = None
        self.subjects = {}
        self.dir2id = {}
        self.buffer_count = 0

        self.exit_now = False

    def create_model(self):
        self.model = cv2.createLBPHFaceRecognizer()
        # self.model = cv2.createEigenFaceRecognizer()
        # self.model = cv2.createFisherFaceRecognizer()
        return

    def load(self, model_state_file):
        self.create_model()
        model_state_file = model_state_file.split(".")[-2]
        if not (os.path.exists(model_state_file + ".xml")
            and os.path.exists(model_state_file + ".csv")):
            raise Exception("File not found")
        print "Reading model state from %s" % model_state_file
        self.load_subject_index(model_state_file + ".csv")
        self.model.load(model_state_file + ".xml")

    def save(self, model_state_file):
        if self.model is None:
            raise Exception("No model found")

        model_state_file = model_state_file.split(".")[-2]
        self.save_subject_index(model_state_file + ".csv")
        self.model.save(model_state_file + ".xml")
        print "Saved the model state"

    def train(self, image_dir):
        start = time.time()
        self.create_model()

        print "Reading the images in from %s" % image_dir
        X,y = self.read_all_subjects(image_dir)
        y = np.asarray(y, dtype=np.int32)

        print "Training..."
        self.model.train(np.asarray(X), np.asarray(y))
        print "Trained in %.1f seconds" % (time.time() - start)
        return

    def update(self, subject_dir, name):
        start = time.time()
        if self.model is None:
            raise Exception("No model found")
        if not os.path.exists(subject_dir):
            raise Exception("Directory not found")

        print "Reading the subject in from %s" % subject_dir
        X = []
        y = []
        self.read_subject(subject_dir, X, y, name)
        y = np.asarray(y, dtype=np.int32)

        print "Updating model..."
        self.model.update(np.asarray(X), np.asarray(y))
        print "Updated in %.1f seconds" % (time.time() - start)

        return

    def run(self):
        if self.model is None:
            raise Exception("No model found")
        self.run_camera(self.predict)

    def learn(self, image_dir, subject_dir, name):
        while True:
            self.learn_once(image_dir, subject_dir, name)

    def learn_once(self, image_dir, subject_dir, name):
        """ Run the prediction and learn unknown faces """
        if self.model is None:
            raise Exception("No model found")
        if not os.path.exists(image_dir):
            raise Exception("image_dir not found")
 
        subject_id = 0
        self.run_camera(self.predict_and_exit)
        do_save = True
        for subject in self.buffer:
            if subject and subject[2] < self.min_distance:
                print "I may know this face. Not saving..."
                do_save = False
                break
        if not do_save:
            return

        while os.path.exists(os.path.join(image_dir, "%s_%d" % (subject_dir, subject_id))):
            subject_id += 1

        subject_dir_id = "%s_%d" % (subject_dir, subject_id)
        name_id = "%s #%d" % (name, subject_id)
        self.save_subject(image_dir, subject_dir_id, name_id)
        self.update(os.path.join(image_dir, subject_dir_id), name_id)

    def record(self, image_dir, subject_dir, name):
        if not os.path.exists(image_dir):
            raise Exception("image_dir not found")
        self.run_camera(self.check_exit)
        self.save_subject(image_dir, subject_dir, name)

    def run_camera(self, callback):
        if not self.cam:
            self.cam = self.create_capture()
        self.exit_now = False
        self.buffer_count = 0
        while not self.exit_now:
            if self.last_capture and time.time() - self.last_capture < self.detect_freq:
                continue
            self.last_capture = time.time()
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rows, cols = gray.shape
            small_gray = cv2.resize(gray, (cols / 2, rows / 2))
    
            rects = self.detect(small_gray, self.cascade)
            vis = img.copy()

            self.draw_rects(vis, rects, (0, 255, 0))

            for x1, y1, x2, y2 in rects:
                s = max(y2 - y1, x2 - x1)
                y1 = (y1 + y2 - s) / 2
                y2 = y1 + s
                x1 = (x1 + x2 - s) / 2
                x2 = x1 + s
                roi = small_gray[y1:y2, x1:x2]
                resized_roi = cv2.resize(roi, (self.proc_size_h, self.proc_size_w))
                vis_roi = vis[y1:y2, x1:x2]

                print "Detected face at (%s, %s)-(%s, %s)" % (x1, y1, x2, y2)

                # buffer[i] = (image, label, distance)
                self.buffer[self.buffer_count] = [resized_roi, None, None]
                callback(resized_roi)

                self.buffer_count = (self.buffer_count + 1) % self.buffer_size

                # For now process just one face in the frame
                break

            # draw_str(vis, (20, 20))
            if self.display:
                cv2.imshow("face recog", vis)
   
        if self.display: 
            cv2.destroyAllWindows()

    def predict_and_exit(self, resized_roi):
        self.predict(resized_roi)
        self.check_exit(resized_roi)

    def check_exit(self, resized_roi):
        if self.buffer_count >= self.buffer_size - 1:
            self.exit_now = True

    def predict(self, resized_roi): 
        [id, distance] = self.model.predict(np.asarray(resized_roi, dtype=np.uint8))
        name = self.subjects.get(id, {}).get("name", "")
        if distance < self.max_distance:
            print name, id, distance
        else:
            print "Unknown face (distance=%.2f) (%d %s?)" % (distance, id, name)

        self.buffer[self.buffer_count][1] = id
        self.buffer[self.buffer_count][2] = distance
        return (id, distance)

    def detect(self, img, cascade):
        haar_flags = 0
        rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(self.proc_size_h, self.proc_size_w), flags = haar_flags)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        print "Detected faces"
        return rects
    
    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    def create_capture(self):
        cap = cv2.VideoCapture(self.video_src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if cap is None or not cap.isOpened():
            raise Exception("Unable to open video source: %d" % self.video_src)

        print "Set (w, h) = (%d, %d)" % (self.width, self.height)
        return cap

    def load_subject_index(self, path):
        with open(path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                print "Loaded index: ", row[0], row[1], row[2]
                self.subjects[int(row[0])] = {"name": row[1], "dir": row[2]}
                self.dir2id[row[2]] = int(row[0])

    def save_subject_index(self, path):
        with open(path, "w") as file:
            for id in self.subjects:
                file.write("%d, %s, %s\n" % (int(id), self.subjects[id]["name"], self.subjects[id]["dir"]))

    def read_all_subjects(self, image_dir):
        X,y = [], []
        for root, subdirs, files in os.walk(image_dir):
            for subdir in subdirs:
                path = os.path.join(image_dir, subdir)
                self.read_subject(path, X, y)
        return (X, y)

    def read_subject(self, subject_path, images, ids, name=None):
        root, subject_dir = os.path.split(subject_path)

        if name is None and os.path.exists(os.path.join(subject_path, "name.txt")):
            with open(os.path.join(subject_path, "name.txt")) as file:
                name = file.read().strip("\n")
        name = name or subject_dir

        id = self.dir2id.get(subject_dir, None)
        if id is None:
            id = len(self.subjects)
        self.dir2id[subject_dir] = id
        self.subjects[id] = {"name": name, "dir": subject_dir}
        print "%d: %s (%s)" % (id, name, subject_path)
 
        for file in os.listdir(subject_path):
            ext = file.split(".")[-1]
            if not ext in ["jpg", "png", "bmp", "gif"]:
                continue
            try:
                im = cv2.imread(os.path.join(subject_path, file), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                im = cv2.resize(im, (self.proc_size_h, self.proc_size_w))
                images.append(np.asarray(im, dtype=np.uint8))
                ids.append(id)
            except IOError, (errno, strerror):
                print "I/O error({0}): {1}".format(errno, strerror)
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

    def save_subject(self, image_dir, subject_dir, name):
        os.system("mkdir -p %s" % os.path.join(image_dir, subject_dir))
        with open(os.path.join(image_dir, subject_dir, "name.txt"), "w") as file:
            file.write(name)

        count = 0
        os.system("mkdir -p %s" % os.path.join(image_dir, subject_dir))
        while os.path.exists(os.path.join(image_dir, "%s/%s.jpg" % (subject_dir, count))):
            count += 1

        for record in self.buffer:
            if not record:
                continue
            face, label, distance = record
            cv2.imwrite(os.path.join(image_dir, "%s/%s.jpg" % (subject_dir, count)), face)
            count = count + 1
        print "%s's %d images saved" % (subject_dir, count)


if __name__ == '__main__':
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["commands=", "face_detect=", "image_dir=", "model_state=", "subject_dir=", "subject_name="])
    except getopt.GetoptError as err:
        print str(err)
        print "--commands <command,another_command...> --image_dir <dir> --model_state <file> --subject_dir <dir> --subject_name <name>"
        sys.exit(2)

    commands = ""
    face_detect = "facedetect.xml"
    image_dir = ""
    model_state = ""
    subject_dir = ""
    subject_name = ""
    for o, a in opts:
        if o == "--commands":
            commands = a
        elif o == "--face_detect":
            face_detect = a
        elif o == "--image_dir":
            image_dir = a
        elif o == "--model_state":
            model_state = a
        elif o == "--subject_dir":
            subject_dir = a
        elif o == "--subject_name":
            subject_name = a

    face_recog = FaceRecog(
        face_detect,
        video_src=0)

    for command in commands.split(","):
        if command == "train":
            face_recog.train(image_dir)
        elif command == "save":
            face_recog.save(model_state)
        elif command == "load":
            face_recog.load(model_state)
        elif command == "learn":
            face_recog.learn(image_dir, subject_dir, subject_name)
        elif command == "record":
            face_recog.record(image_dir, subject_dir, subject_name)
        elif command == "update":
            face_recog.update(os.path.join(image_dir,subject_dir), subject_name)
        elif command == "run":
            face_recog.run()
