import zipfile
import io
import pytesseract
import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageDraw

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
# file names with images
small_zip_file = 'readonly/small_img.zip'
large_zip_file = 'readonly/images.zip'

def files_from_zip(zip_file):
    '''
    unpacking zip files
    :zip_file: string with file name
    :return: list of tuples with Image objects and names of files
    '''
    z = zipfile.ZipFile(zip_file)
    images = []
    for zip_f in z.namelist():
        # image in bytes
        a = z.read(zip_f)
        # BytesIO object for the  image
        b = io.BytesIO(a)
        # image object
        images.append((Image.open(b), zip_f))
    return images

def search_for_word(word, images_and_names):
    '''
    searching for a word in images
    :word: string
    :images_and_names: list of tuples with Image objects and names of files
    :return: list of tuples with Image objects and names of files that contain word
    '''
    images_w_word = []
    for image, name in images_and_names:
        # creating a string with text from image
        string_from_image = pytesseract.image_to_string(image)
        # checking for a word
        if word in string_from_image:
            images_w_word.append((image, name))
    return images_w_word

def face_rec(images_w_word):
    '''
    face recognition
    :images_w_word: list of Image objects
    :return: list of tuples with Image objects, names of files, faces' coords
    '''
    images_and_faces = []
    for image, name in images_w_word:
        image_gray = image.convert('L')
        image_array = np.array(image_gray, dtype='uint8')
        faces = np.array(face_cascade.detectMultiScale(image_array, 1.35))
        faces_list = faces.tolist()
        images_and_faces.append((image, name, faces_list))
    return images_and_faces

def show_faces(images_and_faces):
    '''
    output text and image
    :images_and_faces: list of tuples with Image objects, names of files, faces' coords
    :return: no return
    '''
    for image, name, faces in images_and_faces:
        if faces == []:
            print('Results found in {}'.format(name))
            print('But there is no faces in that file!')

        else:
            # searching for max w and h in faces
            max_w = 0
            max_h = 0

            for x, y, w, h in faces:
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h

            if len(faces) > 5:
                output_image = Image.new(mode='RGB', size=(
                max_w * 5, max_h * ((len(faces) // 5) + 1)))
            else:
                output_image = Image.new(mode='RGB', size=(max_w * 5, max_h))

            x1_output = 0
            y1_output = 0

            for x, y, w, h in faces:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                face_crop = image.crop((x1, y1, x2, y2)).resize((max_w, max_h))

                output_image.paste(face_crop, (
                x1_output, y1_output, x1_output + max_w, y1_output + max_h))

                if x1_output + max_w >= output_image.size[0]:
                    y1_output += max_h
                    x1_output = 0
                else:
                    x1_output += max_w
            print('Results found in {}'.format(name))
            display(output_image)


images_and_names_christopher = files_from_zip(small_zip_file)
images_w_word_christopher = search_for_word('Christopher', images_and_names_christopher)
images_and_faces_christopher = face_rec(images_w_word_christopher)
print('----Christopher in small_img.zip----')
show_faces(images_and_faces_christopher)

images_and_names_mark = files_from_zip(large_zip_file)
images_w_word_mark = search_for_word('Mark', images_and_names_mark)
images_and_faces_mark = face_rec(images_w_word_mark)
print('----Mark in images.zip----')
show_faces(images_and_faces_mark)
