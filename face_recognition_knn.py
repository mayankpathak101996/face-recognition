
import numpy as np
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder
import cv2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','JPEG'}

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False):

    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def predict(frame,knn_clf = None, model_save_path ="", DIST_THRESH = .4):

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_faces_loc=face_recognition.face_locations(frame)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]

def draw_preds(preds,frame):

    source_img = Image.fromarray(frame.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        if name=="Unknown":
            draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="blue")
            draw.rectangle(((loc[3]+1, loc[0]+1), (loc[1]-1, loc[2]-1)), outline="blue")
            draw.rectangle(((loc[3]+2, loc[0]+2), (loc[1]-2, loc[2]-2)), outline="blue")
            #draw.rectangle(((loc[3] + 3, loc[0] + 3), (loc[1] -3, loc[2] - 3)), outline="red")
            #draw.rectangle(((loc[3], loc[0] -35), (loc[1] , loc[0] )), outline="red",fill=(255,0,0,255))
            #draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
        else:
            draw.rectangle(((loc[3], loc[0]), (loc[1], loc[2])), outline="green")
            draw.rectangle(((loc[3] + 1, loc[0] + 1), (loc[1] - 1, loc[2] - 1)), outline="green")
            draw.rectangle(((loc[3] + 2, loc[0] + 2), (loc[1] - 2, loc[2] - 2)), outline="green")
            draw.rectangle(((loc[3] + 3, loc[0] + 3), (loc[1] - 3, loc[2] - 3)), outline="green")
            draw.rectangle(((loc[3], loc[0] - 35), (loc[1], loc[0])), outline="green", fill=(0,255,0,255))
            draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
    return source_img

if __name__ == "__main__":
    #knn_clf = train("knn_examples/train",model_save_path="expression_Knn_clf.p")
    with open("Large_Knn_clf.p", 'rb') as f:
        knn_clf1 = pickle.load(f)
    cap=cv2.VideoCapture("VID_20180618_180346.mp4")
    out = cv2.VideoWriter('test_output1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (1920, 1080))
    while True:
        ok,frame1=cap.read()
        if ok==False:
            break
        frame1 = frame1[:, :, ::-1]
        preds = predict(frame1, knn_clf=knn_clf1)
        frame=draw_preds(preds, frame1)
        frame=np.array(frame)
        frame=frame[:, :, ::-1]
        cv2.resize(frame, (1920, 1080))
        cv2.imshow("output",frame)
        out.write(frame)
        k=cv2.waitKey(1)
        if k==27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

