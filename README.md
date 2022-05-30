# clarius-tha-1

---
**Image segmentation ML model deployed as a RESTful API and packaged in a Docker container.**

To build the API I used Flask and Flasgger with SwaggerUI. Flasgger allows us to automatically generate an interactive API documentation where one can try out API calls.

The semantic segmentation model deployed is a Fully-Convolutional Network model with a ResNet-50 backbone, pre-trained on the COCO train2017 dataset and containing the same classes as Pascal VOC.

---

**Running the API from a Docker container:**

1. From the *api* folder, build the docker container image with:
```python
$ docker image build -f Dockerfile.pytorch -t <image-name> .
```

2. Run the container from the image created:
```python
$ docker container run -d --rm -p 8888:8888 <image-name>
```
    
3. Go to http://localhost:8888/apidocs/ to interact with the API. You can test POST calls with the sample images in this repository or uploading your own images.


https://user-images.githubusercontent.com/49324844/156843215-c124078f-6ee3-4a9a-bbd1-c6550046f280.mp4

---

**Visualizing responses:**

- To visualize the outputted segmented image you can download the Response body (it downloads as a .txt file), and then pass it to the function *response2img* that you can find in *utils.py*.

![results](https://github.com/4ndrewparr/clarius-tha-1/blob/main/post-processing/results_200-2500.gif)

---
---
