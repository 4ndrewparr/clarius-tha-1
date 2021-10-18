# clarius-tha-1

Image segmentation ML model deployed as a RESTful API within a Docker container.

To build the API I used Flask and Flasgger with SwaggerUI. Flasgger allow us to automatically generate an interactive API documentation where one can try out API calls.

The semantic segmentation model deployed is a Fully-Convolutional Network model with a ResNet-50 backbone, pre-trained on the COCO train2017 dataset.

Instructions to run the API from a Docker container:

1. From the api-docker folder, build the docker container image with:
```python
$ docker image build -f Dockerfile.pytorch -t <image-name> .
```

2. Run the container from the image created:
```python
$ docker container run -d --rm -p 8888:8888 <image-name>
```
    
3. Go to http://localhost:8888/apidocs/ to interact with the API. You can test POST calls uploading your own images.

To see the outputted segmented image you can download the Response body (it downloads as a .txt file), and then pass it to the function *response2img* that you can find in *utils.py*.
