import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from skimage.transform import resize
import cv2

#To get rid of the comments of TF
os.system("clear")

def Video_to_Frames(Video_file):
    """
    @Video_file: path to video

    return: a nparray which contains frames as narray format
    """
    frames = []
    cap = cv2.VideoCapture(Video_file)  
    
    while True:
        ret, frame = cap.read()  
        if not ret:
            break # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        
    cap.release()
    output = np.zeros((len(frames), frames[0].shape[0], frames[0].shape[1]))
    for i in range(frames):
        output[i]  = frames[i]
    return output


def segmentation_infrared(imgs):
    #Output array : is sample * H * W * 1
    output_array = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    output_array = output_array.astype(float)

    #for i in tqdm(range(imgs.shape[0])):
    for i in tqdm(range(imgs.shape[0])):
        binary, seg_img = segmentation_ir(imgs[i])
        output_array[i] = binary
    return output_array

def dataload_seg(path1, size=(256, 256)):
    """
    inputs: rgb_dataset, ir_dataset : path to rgb, ir dataset respectively
    returns: ir, visible arrays contains segmented ndarrays for imgs 
    """
    onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    onlyfiles1.sort()

    print("\t\tWorking with {0} images".format(len(onlyfiles1)))

    visible = np.ndarray(shape=(len(onlyfiles1), size[0], size[1]), dtype = np.float32)

    print("\t############# Uploading Segmented VISIBLE images #############")
    i = 0
    for img_name in tqdm(onlyfiles1):
        img_path = path1 + "/" + img_name
        img_vis = cv2.imread(img_path)
        img_vis_gray = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
        #TODO: Segmentation here
        #bin_vis, seg_vis = segmentation_vis(img_vis_gray)
        visible[i] = img_vis_gray
        i += 1
    print("\t############# All VISIBLE Segmented images to array ###########")
    return visible

def dataload(path1,  size = (240, 320)):
    """
    inputs: rgb_dataset, ir_dataset : path to rgb, ir dataset respectively
    returns: ir, visible arrays contains ndarrays for imgs 
    """
    onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    onlyfiles1.sort()

    visible = np.ndarray(shape=(len(onlyfiles1), size[0], size[1]), dtype = np.float32)
    i = 0
    for img_name in tqdm(onlyfiles1):
        img_path = path1 + "/" + img_name
        img = load_img(img_path, color_mode = "grayscale")                            
        # Convert to Numpy Array
        x = img_to_array(img)  
        x = resize(x, size)                         #Output: (256, 256, 1)
        #print(x.shape)
        x = x.reshape(size)
        visible[i] = x
        i += 1
    print("\t############# All VISIBLE images to array ##########")
    return visible

def dataload_rgb(path1, size = (256, 256, 3)):
    """
    inputs: rgb_dataset, ir_dataset : path to rgb, ir dataset respectively
    returns: ir, visible arrays contains ndarrays for imgs 
    """

    onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    onlyfiles1.sort()

    print("\t\tWorking with {0} VISIBLE images".format(len(onlyfiles1)))

    visible = np.ndarray(shape=(len(onlyfiles1), size[0], size[1], size[2]), dtype = np.float32)

    print("\t############# Uploading images #############")
    print("\t## Uploading {} ##".format(path1))
    i = 0
    for img_name in tqdm(onlyfiles1):
        img_path = path1 + "/" + img_name
        img = load_img(img_path)                            
        # Convert to Numpy Array
        x = img_to_array(img)  
        x = resize(x, size)                         #Output: (256, 256, 1)
        #print(x.shape)
        x = x.reshape(size)
        visible[i] = x
        i += 1

    print("\t############# All images to array ##########")
    return visible

def parseImages(imgs, size):
    imgsDimSup = torch.from_numpy(imgs).float()
    imgsDimSup = imgsDimSup[None, ...]
    imgsDimSup = imgsDimSup[:, :size, :, :]
    imgsDimSup = imgsDimSup.permute(1, 0, 2, 3)
    imgsDimSup = imgsDimSup/255
    return imgsDimSup

# for splitting out batches of data
def next_batch(moving, fixed, batch_size = 8):    
    vol_shape = moving.shape[1:] # extract data shape
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        #IR : Moving_img, RGB: Fixed
        idx1 = np.random.randint(0, moving.shape[0], size=batch_size)
        moving_images = moving[idx1, ..., np.newaxis]
        fixed_images = fixed[idx1, ..., np.newaxis]
        
        inputs = [moving_images, fixed_images] 
        outputs = [fixed_images, zero_phi]
        
        yield inputs, outputs     

def generate(moving, fixed):
    """
    Generator of inputs to Voxelmorph model in single shot
    """
    vol_shape = moving.shape[1:] # extract data shape
    ndims = len(vol_shape)
    zero_phi = np.zeros([moving.shape[0], *vol_shape, ndims])    
    inputs = [moving, fixed] 
    outputs = [fixed, zero_phi]
    return inputs, outputs    

def plot_history(hist, loss_name='loss'):
    """
    Quick function to plot the history 
    """
    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    loss_name = loss_name + ".png"
    plt.savefig(loss_name)

def save_pic(x, vis, ir, id, name ="flow"):
    fig, ax = plt.subplots(1, 5, figsize=(15, 12))
    ax[0].set_title("Image RGB")
    ax[0].imshow(vis[id, 0], cmap='gray')

    ax[1].set_title("Image IR")
    ax[1].imshow(ir[id, 0], cmap='gray')

    ax[2].set_title("Segmented RGB")
    ax[2].imshow(visible_seg_norm[id, 0], cmap='gray')

    ax[3].set_title("Segmented IR")
    ax[3].imshow(ir_seg_norm[id, 0], cmap='gray')

    ax[4].set_title("Warp")
    ax[4].imshow(x[id, 0].detach().numpy(), cmap='gray')

    name = name + str(id) +".png"
    path = "flow/" + name
    plt.savefig(path)

def save_pred(input_s, val_pred, name = "predictions", samples = 8):
    #Generate a random nbr to display correspong img 
    chaos = np.random.choice(range(samples-1), 4, replace=False)
    #chaos = np.random.randint(low=0, high=samples-1, size=4)
    print("[i, j, k, l]: ", chaos)
    i, j, k, l = chaos[0], chaos[1], chaos[2], chaos[3]
    fig, ax = plt.subplots(4, 4, figsize=(15, 12))
    ##Sample0
    ax[0][0].imshow(input_s[0][i][:,:, 0], cmap='gray') #moving_images: IR
    ax[0][0].set_title("IR Input")

    ax[0][1].imshow(input_s[1][i][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[0][1].set_title("RGB Input")

    ax[0][2].imshow(val_pred[0][i][:,:, 0], cmap='gray') #Flow
    ax[0][2].set_title("Flow")

    ax[0][3].imshow(input_s[1][i][:,:, 0] + val_pred[0][i][:,:, 0], cmap='gray')
    ax[0][3].set_title("Stacking RGB over Flow")
    ##Sample1
    ax[1][0].imshow(input_s[0][j][:,:, 0], cmap='gray') #moving_images: IR
    ax[1][1].imshow(input_s[1][j][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[1][2].imshow(val_pred[0][j][:,:, 0], cmap='gray') #Flow
    ax[1][3].imshow(input_s[1][j][:,:, 0] + val_pred[0][j][:,:, 0], cmap='gray')
    ##Sample2
    ax[2][0].imshow(input_s[0][k][:,:, 0], cmap='gray') #moving_images: IR
    ax[2][1].imshow(input_s[1][k][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[2][2].imshow(val_pred[0][k][:,:, 0], cmap='gray') #Flow
    ax[2][3].imshow(input_s[1][k][:,:, 0] + val_pred[0][k][:,:, 0], cmap='gray')
    ##Sample3
    ax[3][0].imshow(input_s[0][l][:,:, 0], cmap='gray') #moving_images: IR
    ax[3][1].imshow(input_s[1][l][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[3][2].imshow(val_pred[0][l][:,:, 0], cmap='gray') #Flow
    ax[3][3].imshow(input_s[1][l][:,:, 0] + val_pred[0][l][:,:, 0], cmap='gray')

    name = name + ".png"
    plt.savefig(name)

def save_pred_frames(input_s, val_pred, folder= 'sample', samples = 8):
    folder_ir = "./" + folder + "/IR"
    folder_flow = "./" + folder + "/Flow"
    if not os.path.exists(folder):
        os.mkdir(folder_ir)
        os.mkdir(folder_flow)
        print("Directory " , folder_ir ,  " Created ")
        print("Directory " , folder_flow ,  " Created ")
        i = 0
        while i < samples:
            name_img = i + "png"
            
            save = folder_ir + "/" + name_img
            cv2.imwrite(input_s[0][i][:,:, 0], save)
            
            save = folder_flow + "/" + name_img
            cv2.imwrite(val_pred[0][i][:,:, 0], save) 

    else:    
        print("Directory " , folder ,  " already exists")

def save_segmented(rgb_orig, rgb_segmented, ir_orig, ir_segmented,name = "segmentation_sample"):
    fig, ax = plt.subplots(1, 4, figsize=(15, 12))
    ##Sample0
    ax[0].imshow(rgb_orig, cmap="gray") #moving_images: IR
    ax[0].set_title("RGB")

    ax[1].imshow(rgb_segmented, cmap="gray") #Fixed images: RGB
    ax[1].set_title("RGB Segmented")

    ax[2].imshow(ir_orig, cmap="gray") #moving_images: IR
    ax[2].set_title("IR")

    ax[3].imshow(ir_segmented, cmap="gray") #Fixed images: RGB
    ax[3].set_title("IR Segmented")

    name = name + ".png"
    plt.savefig(name)


def save_interpn_imgs(ir, rgb, interpn, name ="interpn_sample"):
    ig, ax = plt.subplots(1, 3, figsize=(15, 12))

    ax[0].imshow(ir, cmap="gray") #moving_images: IR
    ax[0].set_title("IR")

    ax[1].imshow(rgb, cmap="gray") #Fixed images: RGB
    ax[1].set_title("RGB")

    ax[2].imshow(interpn, cmap="gray") #moving_images: IR
    ax[2].set_title("Wrapped")
    name = name + ".png"
    plt.savefig(name)



def prepare_pairs(path1, path2):
    """
    path1: contains only the files that we want to keep
    path2: is the big folder
    """
    onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    onlyfiles2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    onlyfiles1.sort()
    onlyfiles2.sort()

    for file_name in onlyfiles2:                                    #Looking for IR frames which are not in RGB
        if file_name not in onlyfiles1:
            img_path = path2 + "/" + file_name
            os.remove(img_path)                                     #Remove the frame

    onlyfiles1 = [f for f in os.listdir(path1) if os.path.join(path1, f)]
    onlyfiles2 = [f for f in os.listdir(path2) if os.path.join(path2, f)]
    onlyfiles1.sort()
    onlyfiles2.sort()

    print("Working with {0} VISIBLE images".format(len(onlyfiles1)))
    print("Working with {0} IR images".format(len(onlyfiles2)))

def rename(folder, suffix):
    onlyfiles1 = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    onlyfiles1.sort()

    for file_name in onlyfiles1:
        img_path = folder + "/" + file_name
        new_name = "frame" + str(suffix) + ".jpg"
        new_path = folder + "/" + new_name
        os.rename(img_path, new_path)
        suffix = int(suffix) + 1
    #print("suffix: ", suffix)
    print("Done Renaming frames, last one is {}".format(new_name))

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def save_final_results(ir, vis, mask_flow, output, name = "sample", samples = 8):
    #Generate a random nbr to display correspong img 
    chaos = np.random.choice(range(samples-1), 3, replace=False)
    #chaos = np.random.randint(low=0, high=samples-1, size=4)
    print("[i, j, k, l]: ", chaos)
    i, j, k = chaos[0], chaos[1], chaos[2]
    fig, ax = plt.subplots(3, 5, figsize=(15, 12))
    ##Sample0
    ax[0][0].imshow(input_s[0][i][:,:, 0], cmap='gray') #moving_images: IR
    ax[0][0].set_title("IR Input")

    ax[0][1].imshow(input_s[1][i][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[0][1].set_title("RGB Input")

    ax[0][2].imshow(val_pred[0][i][:,:, 0], cmap='gray') #Flow
    ax[0][2].set_title("Mask Flow")

    ax[0][3].imshow(val_pred[0][i][:,:, 0], cmap='gray') #Flow
    ax[0][3].set_title("Output")    
    
    ##Sample1
    ax[1][0].imshow(input_s[0][j][:,:, 0], cmap='gray') #moving_images: IR
    ax[1][1].imshow(input_s[1][j][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[1][2].imshow(val_pred[0][j][:,:, 0], cmap='gray') #Flow
    ax[1][3].imshow(input_s[1][j][:,:, 0] + val_pred[0][j][:,:, 0], cmap='gray')
    ##Sample2
    ax[2][0].imshow(input_s[0][k][:,:, 0], cmap='gray') #moving_images: IR
    ax[2][1].imshow(input_s[1][k][:,:, 0], cmap='gray') #Fixed images: RGB
    ax[2][2].imshow(val_pred[0][k][:,:, 0], cmap='gray') #Flow
    ax[2][3].imshow(input_s[1][k][:,:, 0] + val_pred[0][k][:,:, 0], cmap='gray')

    name = name + ".png"
    plt.savefig(name)


def segmentation_rcnn(imgs):
    #Output array : is sample * H * W * 1
    output_array = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    output_array = output_array.astype(float)

    #for i in tqdm(range(imgs.shape[0])):
    for i in tqdm(range(imgs.shape[0])):
        seg_img = rcnn.detect([imgs[i]], verbose=0)[0]
        labels = [class_names[ele] for ele in seg_img['class_ids']]

        sample = np.zeros((imgs.shape[1], imgs.shape[2]))
        sample = sample.astype(float)
        #print("Mask shape for {} is: {}".format(i, seg_img['masks'].shape))
        #sample = seg_img['masks'][:, :, 0]
        if seg_img['masks'].shape[2] != 0:
            for k in range(seg_img['masks'].shape[2]):
                #print("Class ID: {}".format(labels[k]))
                if labels[k] == 'person':
                    sample += seg_img['masks'][:, :, k]
                    #print("Mask RGB {} added as person!".format(k))
            output_array[i] = sample
        else:
            output_array[i] = np.zeros((imgs.shape[1], imgs.shape[2]))
    output_array = output_array * 255
    return output_array

def saveVideo(frames, fps, width, height, outputName, fourCC='DIVX'):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*fourCC)
    out = cv2.VideoWriter(outputName, fourcc, fps, (width, height))

    for i in range(frames.shape[0]):
    
        # write the frame
        out.write(frames[i])

        cv2.imshow('frame', frames[i])

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def segmentation_ir(img):
    """
    img: path to IR image
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    histogram = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    hist_normalize = histogram.ravel() / histogram.max()
    Q = hist_normalize.cumsum()
    x_axis = np.arange(256)
    mini = np.inf
    epsilon = 10^(-8)
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_normalize, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        b1, b2 = np.hsplit(x_axis, [i])
        m1, m2 = np.sum(p1 * b1) / (q1+epsilon), np.sum(p2 * b2) / (q2+epsilon)
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / (q1+epsilon), np.sum(((b2 - m2) ** 2) * p2) / (epsilon+q2)
        fn = v1 * q1 + v2 * q2
        if fn < mini:
            mini = fn
    #ret, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    ret, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # use the mask to select the "interesting" part of the image
    im_thresh = cv2.bitwise_and(img, binarized)
    return binarized, im_thresh