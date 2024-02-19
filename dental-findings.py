import re
import io
import cv2
import pandas as pd
from os import environ
import os
import pytesseract
import fitz
import requests
from PIL import Image,ImageEnhance
from frontend import *
import matplotlib.pyplot as plt
import numpy as np
# pytesseract.pytesseract.tesseract_cmd = os.getenv("tesseract_path")
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# print(os.getenv("tesseract_path"))
df = pd.read_excel("./SampleDataAndFindings.xlsx", usecols=["NARRATIVE"])
df = pd.DataFrame(df['NARRATIVE'])
def remove_reference_code(line):
    pattern1 = r'\b(?:CHC|NEA|DXC)#\d+\b'
    # pattern2 = r'(?:\b(?:CHC|NEA|DXC)\d+\b)'
    pattern2= r'\b(?:CHC|NEA|DXC)\d+\b'
    match1 = re.search(pattern1, line)
    if match1 and len(match1.group(0))>4:
      return re.sub(pattern1, '', line)
    match2 = re.search(pattern2, line)
    if match2 and len(match2.group(0))>4:
      return re.sub(pattern2, '', line)
    else:
        return line
# def check_word_existence(word_list, word_to_check):
#     if word_to_check in word_list:
#         return True
#     else:
#         return False
def check_back_and_forward_element(line1,line2):
    # one_index_back_words=["#","hot","pa","pfm","d","onchewing"]
    # one_index_forward_words=["#","cold","io","rctd","l","temp"]
    if(("#" in line1 and "#" in line2) or("hot" in line1 and "cold" in line2) or("pa" in line1 and "io" in line2) or("pfm" in line1 and "rctd" in line2) or("d" in line1 and "l" in line2) or ("onchewing" in line1 and "temp" in line2)):
        return True
    else:
        return False

def check_if_and_exists_between_line(line):
    line_lower = line.replace("&amp;", "and").replace("&", "and").lower()
    words = line_lower.split() 
    for index, word in enumerate(words):
        if(word == "and"):
            if check_back_and_forward_element(words[index-1],words[index+1]):
                continue
            else:
                words[index] = ","
    modified_line = " ".join(words)
    
    return modified_line



def remove_unwanted_strings(line):
    line_lower = line.lower()
    line_lower.replace(")","")
    phrases_to_remove = ["see attched chart notes", "x-ray, photo attached","x-ray attached"]

    for phrase in phrases_to_remove:
        line_lower = line_lower.replace(phrase, "")
        
    return line_lower.strip()
def get_finalized_data(line):
    line = remove_reference_code(line)
    
    line = check_if_and_exists_between_line(line)
    line = remove_unwanted_strings(line)
    line = line.strip().replace(".",",").lower().split(",")
    modified_lines_array = [i for i in line if not ('' == i)]
    return modified_lines_array
def add_space_after_digit(text):
    pattern = r'(\d)(?=[a-zA-Z])'

    modified_text = re.sub(pattern, r'\1 ', text)

    if re.match(r'(?:CHC|NEA|DXC)\d{8,14}\b', modified_text):
        return modified_text
    else:
        return text
def get_preprocessed_data():
    csv_data=[]
    for text in df['NARRATIVE']:
        # pattern = r'([a-zA-Z]{3}\d+)(?=\w)'
        # modified_string = re.sub(pattern, r'\1 ', text)
        modified_string = add_space_after_digit(text)
        if("XRAYS AND NARRATIVE ATTACHED" in text):
            continue
        csv_data.append({"data":get_finalized_data(modified_string)})
    # print(csv_data)
    csv_df = pd.DataFrame(csv_data)
    csv_path = "./preprocessedData.csv"
    csv_df.to_csv(csv_path, index=False)

# get_preprocessed_data()
# print(get_finalized_data("CHC#26247588 X-RAY ATTACHED #30 CRN DISLODGED &amp; LOST"))
#remove ),'', xray and photo,ATTACHMENT
# see attched chart notes, X-RAY, PHOTO ATTACHED

# def handle_default_ocr_file_using_opencv(file_path):
#     image = cv2.imread(file_path)

#     grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     alpha = 0.5 # Contrast control (1.0-3.0)
#     beta = 5    # Brightness control (0-100)
#     enhanced_image = cv2.convertScaleAbs(grayscale_image, alpha=alpha, beta=beta)
#     cv2.imshow('Image', enhanced_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     extracted_text = pytesseract.image_to_string(enhanced_image, lang="eng")
#     cleaned_text = extracted_text.strip()
#     return cleaned_text

def handle_default_ocr_file_using_opencv(file_path):
    # Read the image
    image = cv2.imread(file_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply normalization
    norm_img = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1]))
    grayscale_image = cv2.normalize(grayscale_image, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    # Apply skew correction
    # def deskew(image):
    #     co_ords = np.column_stack(np.where(image > 0))
    #     angle = cv2.minAreaRect(co_ords)[-1]
    #     if angle < -45:
    #         angle = -(90 + angle)
    #     else:
    #         angle = -angle
    #     (h, w) = image.shape[:2]
    #     center = (w // 2, h // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #     return rotated
    
    # skewed_corrected_image = deskew(grayscale_image)
    
    
    # Apply image scaling
    # def set_image_dpi(image):
    #     length_x, width_y = image.shape
    #     max_dimension = max(length_x, width_y)
    #     factor = min(1, float(1024.0 / max_dimension))
    #     size = int(factor * length_x), int(factor * width_y)
    #     im_resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    #     return im_resized
    
    # scaled_image = set_image_dpi(grayscale_image)
    thres,bw_img=cv2.threshold(grayscale_image,200,230,cv2.THRESH_BINARY)
    # Apply noise removal
    def remove_noise(image):
        kernal = np.ones((1,1),np.uint8)
        image = cv2.dilate(image,kernal,iterations=5)
        kernal = np.ones((1,1),np.uint8)
        image = cv2.erode(image,kernal,iterations=5)
        image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernal)
        # image = cv2.medianBlur(image,0.5)
        return (image)
    def thin_font(image):
        image = cv2.bitwise_not(image)
        kernal = np.ones((2,2),np.uint8)
        image = cv2.erode(image,kernal,iterations=1)
        image = cv2.bitwise_not(image)
        return (image)
    noise_removed_image = remove_noise(bw_img)
    thin_font_image = thin_font(noise_removed_image)
    
    # inverted_image=cv2.bitwise_not(noise_removed_image)
   
    cv2.imshow('Image', bw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Perform OCR on the preprocessed image
    extracted_text = pytesseract.image_to_string(thin_font_image, lang="eng")
    cleaned_text = extracted_text.strip()
    
    return cleaned_text






def handle_default_ocr_file_using_tesseract(file_path):
    image = Image.open(file_path)
    enhanced_image = ImageEnhance.Contrast(image).enhance(1.5)
    grayscale_image = enhanced_image.convert('L')
    # plt.imshow(enhanced_image, cmap='gray')
    # plt.axis('off')  # Turn off axis
    # plt.show()

    extracted_text = pytesseract.image_to_string(grayscale_image, lang="eng")
    return extracted_text.strip()

file_path = './dentalImages/NEA290727191-2.jpg'
# cleaned_text=extracted_text = handle_default_o7cr_file_using_tesseract(file_path)
cleaned_text = handle_default_ocr_file_using_opencv(file_path)
print('extracted text=>',cleaned_text)
