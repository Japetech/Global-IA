from __future__ import print_function
import requests
import json
import cv2

addr = 'http://127.0.0.1:5000'
test_url = addr + '/api/test'

# # prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# img = cv2.imread('lena.jpg')
# # encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# # send http request with image and receive response
# response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# # decode response
# print(json.loads(response.text))

# # expected output: {u'message': u'image received. size=124x124'}

def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(test_url, data=img, headers=headers)
    return response