
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


import numpy as np
import cv2
import time

# https://github.com/farrajota/caltech-extract-data-toolbox/blob/master/vbb.m

model = None

# For Caltech Data
img_height = 480  # Height of the input images
img_width = 640  # Width of the input images

n_classes = 1  # Number of positive classes  person

#scale and aspect ratio and other params are not needed during prediction!

normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

def build_prediction_model():
    global model

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

    # Loss function
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    #model_path = '/home/kara9147/ML/ssd_keras_caltech/ssd7_epoch-03_loss-2.4693_val_loss-2.4097.h5'
    model_path = '/home/kara9147/ML/ssd_keras_caltech/conf2_ssd7_epoch-05_loss-2.4001_val_loss-2.3803.h5'



    # 2: Load the saved model

    model = load_model(model_path,
                      custom_objects={'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})

def play():
    start_time_video = time.time()

    cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set00_V000.avi")
    #cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set03_V008.avi")
    #cap = cv2.VideoCapture("/home/kara9147/ML/caltech-pedestrian-dataset-converter/data/plots/set08_V004.avi")
    #cap = cv2.VideoCapture("/home/kara9147/ML/JAAD/clips/video_0006.mp4")


    # Time to read all frames, predict and put bounding boxes around them, and show them.
    i = 0
    total_time = 0.0

    # Capture frame-by-frame
    ret = True
    while(ret):
        ret, origimg = cap.read()

        i = i + 1
        #print("Processing {} th frame".format(i))
        if (ret != False):
            # Our operations on the frame come here
            img = cv2.resize(origimg, (img_width, img_height))
            # Open CV uses BGR color format
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #print(type(img))
            #print(img.shape)

            batch_img = np.expand_dims(frame , axis=0)
            #print(batch_img.shape )

            current = time.time()
            ##################################PREDICTION######################
            y_pred = model.predict(batch_img)
            end = time.time()
            diff = end - current
            total_time  = total_time  + diff
            print(end - current)
            print("Time spent for predicting: {0}".format(diff))

            # 4: Decode the raw prediction `y_pred`

            y_pred_decoded = decode_detections(y_pred,
                                               confidence_thresh=0.3,
                                               iou_threshold=0.45,
                                               top_k=200,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width)

            np.set_printoptions(precision=2, suppress=True, linewidth=90)
            # print("Predicted boxes:\n")
            # print('   class   conf xmin   ymin   xmax   ymax')

            #print(y_pred_decoded)

            #print(time.time() - start_time)

            if (y_pred_decoded and len(y_pred_decoded[0])):
                colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()  # Set the colors for the bounding boxes
                classes = ['background', 'person', 'people']  # Just so we can print class names onto the image instead of IDs


                # Draw the predicted boxes in blue
                for box in y_pred_decoded[0]:
                    xmin = int(box[-4])
                    ymin = int(box[-3])
                    xmax = int(box[-2])
                    ymax = int(box[-1])
                    color = colors[int(box[0])]
                    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

                    #print((xmin, ymin))
                    #print((xmax, ymax))

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax ), (255, 0, 0), 1)

            # Display the resulting frame
            cv2.imshow('frame',img)

        # waitKey: 0, wait indefinitely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time_video = time.time()
    print("No of frames: {}".format(i))
    print("Total Time: {}".format(total_time))
    print("fps: {}".format(i / (total_time)))



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

build_prediction_model()
play()