from django.shortcuts import render

# Create your views here.
import cv2
import base64
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
import io
from django.core.files.storage import FileSystemStorage
# Hàm phát hiện đối tượng (ví dụ sử dụng ORB)
def blur_and_edges(image):
    image = cv2.GaussianBlur(image, (3, 3), 1.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    edges = cv2.Canny(image, 50, 150)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    return edges


def template_matching(image_edges,template_edges):

    scales = np.arange(0.4,2,0.05) 
    best_val = -1
    best_loc = None
    best_scale = None
   
    for scale in scales:
        # Resize template
        new_w = int(template_edges.shape[1] * scale)
        new_h = int(template_edges.shape[0] * scale)
        resized_template = cv2.resize(template_edges, (new_w, new_h))
        # matchTemplate
        result = cv2.matchTemplate(image_edges, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(max_val)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            print(best_scale,best_val,best_loc)
    return best_loc

def draw_frame(best_loc,template_edges,image):
    top_left = best_loc
    h, w = template_edges.shape[:2]
    bottom_right = (top_left[0] + h , top_left[1] + w)

    # Vẽ rectangle
    image_draw = cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)
    return image_draw


def detection(tem):
    image = cv2.imread('detection/static/image/1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = gray_image[0:,0:1410]
    image = image[0:,0:1410]
    tem = cv2.imread(f'detection/static/image/tem_image/{tem}.jpg')
    tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
    image_edges = blur_and_edges(image_gray)
    temp = blur_and_edges(tem)
    best_loc = template_matching(image_edges,temp)
    detect_image = draw_frame(best_loc,temp,image)
    return detect_image
def index(request):
    return render(request, 'index.html')

def process_click(request):
    # Lấy vị trí click từ request
    template = request.GET.get('template')

    image = detection(template)
    _, buffer = cv2.imencode('.jpg', image)
 
    # Chuyển buffer thành BytesIO để mã hóa base64
    processed_image_io = io.BytesIO(buffer)

    # Mã hóa ảnh thành base64
    encoded_image = base64.b64encode(processed_image_io.getvalue()).decode('utf-8')

    # Trả lại ảnh dưới dạng JSON response
    response = JsonResponse({
        'image': encoded_image  # Mã hóa ảnh thành base64 và trả về
    })

    return response
