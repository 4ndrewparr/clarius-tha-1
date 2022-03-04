import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


dict_classes = {  # label, RGB color
    0: ['background', (0, 0, 0)],
    1: ['aeroplane', (128, 0, 0)],
    2: ['bicycle', (0, 128, 0)],
    3: ['bird', (128, 128, 0)],
    4: ['boat', (0, 0, 128)],
    5: ['bottle', (128, 0, 128)],
    6: ['bus', (0, 128, 128)],
    7: ['car', (128, 128, 128)],
    8: ['cat', (64, 0, 0)],
    9: ['chair', (192, 0, 0)],
    10: ['cow', (64, 128, 0)],
    11: ['diningtable', (192, 128, 0)],
    12: ['dog', (64, 0, 128)],
    13: ['horse', (192, 0, 128)],
    14: ['motorbike', (64, 128, 128)],
    15: ['person', (192, 128, 128)],
    16: ['pottedplant', (0, 64, 0)],
    17: ['sheep', (128, 64, 0)],
    18: ['sofa', (0, 192, 0)],
    19: ['train', (128, 192, 0)],
    20: ['tvmonitor', (0, 64, 128)],
}

def decode_segmap(seg_image):
    """
    k classes segmentation array -> RGB array
    """
    r = np.zeros_like(seg_image).astype(np.uint8)
    g = np.zeros_like(seg_image).astype(np.uint8)
    b = np.zeros_like(seg_image).astype(np.uint8)
    for k in range(len(dict_classes)):
        idx = seg_image == k
        r[idx] = dict_classes[k][1][0]
        g[idx] = dict_classes[k][1][1]
        b[idx] = dict_classes[k][1][2]
    rgb = np.stack([r, g, b], axis=2)  # imshow expects (H, W, C)
    return rgb

def get_input_img(img_file):
    """
    Applies PIL transformations mimicking those done
    with torchvision transforms in inference pipeline.
    """
    img_input = Image.open(img_file)

    # T.Resize(256)
    r = max(256 / img_input.size[0], 256 / img_input.size[1])
    img_input = img_input.resize((int(img_input.size[0]*r), int(img_input.size[1]*r)))

    # T.CenterCrop(224)
    w, h = img_input.size
    img_input = img_input.crop(((w-224)/2, (h-224)/2, (w+224)/2, (h+224)/2))
    
    return img_input

def response2img(response_file, img_file):
    """
    Plots RGB segmented image from JSON .txt response.
    Expects segmentation model merged output (one class per pixel).
    """
    # get output image array
    with open(response_file) as f:
        txt = f.readlines()
    out_merged = np.array(eval(txt[0]))

    # convert to RGB
    img_seg = decode_segmap(out_merged)

    # get input image
    img_input = get_input_img(img_file)
    
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    plt.setp(axs, xticks=[], yticks=[])
    axs[0].imshow(img_input)
    axs[1].imshow(img_seg)
    # legend
    patches = [
        mpatches.Patch(color=np.array(dict_classes[i][1])/255, label=dict_classes[i][0])
        for i in range(21) if i in np.unique(out_merged)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()