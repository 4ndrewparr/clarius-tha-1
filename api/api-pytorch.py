import pickle
import numpy as np
import pandas as pd
import json
from PIL import Image

from flask import Flask, request
from flasgger import Swagger

import torch
from torchvision import models
from torchvision import transforms as T


clarius_api = Flask(__name__)
swagger = Swagger(clarius_api)


@clarius_api.route('/segmentation', methods=["POST"])
def segmentation():
	"""Endpoint to perform semantic segmentation. A 'POST' implementation.
	---
	parameters:

		-	name: input_image
			in: formData
			type: file
			required: true
			
	responses:

		200:

			description: "

				__Response:__ JSON array of an image where
					each pixel has been assigned one object class.\n

				__Classes:__ {
					0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
					4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
					9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
					13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
					17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor',
				}
			"
	"""
	img = Image.open(request.files.get("input_image"))
	
	trf = T.Compose([
	    T.Resize(256),
	    T.CenterCrop(224),
	    T.ToTensor(),
	    T.Normalize(  # ImageNet mean, std
	    	mean=[0.485, 0.456, 0.406],
	    	std=[0.229, 0.224, 0.225]
	    ),
	])
	inp = trf(img).unsqueeze(0)  # we need batch dim

	model = models.segmentation.fcn_resnet50(pretrained=True)
	model.eval()

	out = model(inp)["out"]  # model outputs dict with out and aux 
	out_merged = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
	out_json = json.dumps(out_merged.tolist())
	
	return out_json


if __name__ == '__main__':
	#clarius_api.debug = True
	clarius_api.run(host='0.0.0.0', port=8888)