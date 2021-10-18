import matplotlib.pyplot as plt
import numpy as np

def decode_segmap(seg_image, num_classes=21):
    """
    k classes segmentation array -> RGB array

    classes: {
		0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
		4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
		9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
		13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
		17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor',
	}
    """
    label_colors = np.array([
        (0, 0, 0), # 'background'
        
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
  
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
    ])
    r = np.zeros_like(seg_image).astype(np.uint8)
    g = np.zeros_like(seg_image).astype(np.uint8)
    b = np.zeros_like(seg_image).astype(np.uint8)
    for k in range(num_classes):
        idx = seg_image == k
        r[idx] = label_colors[k, 0]
        g[idx] = label_colors[k, 1]
        b[idx] = label_colors[k, 2]
    rgb = np.stack([r, g, b], axis=2)  # imshow expects (H, W, C)
    return rgb

def response2img(response):
	"""
	Plot RGB segmented image JSON .txt response.
	Expects segmentation model merged output (one class per pixel).
	"""
	with open(response) as f:
		txt = f.readlines()

	out_merged = np.array(eval(txt[0]))

	# convert to RGB image array
	img_seg = decode_segmap(out_merged)

	plt.imshow(img_seg)
	plt.show()

