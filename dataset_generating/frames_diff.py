import os
import random


def rename(folder, suffix):
	onlyfiles1 = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
	onlyfiles1.sort()

	for file_name in onlyfiles1:
		img_path = folder + "/" + file_name
		new_name = "frame" + str(suffix) + ".jpg"
		new_path = folder + "/" + new_name
		os.rename(img_path, new_path)
		suffix = int(suffix) + 1
	print("suffix: ", suffix)
	print("Done Renaming frames, last one is {}".format(new_name))

def prepare_pairs(path1, path2):
    """
    path1: contains only the files that we want to keep
    path2: is the big folder
    """
    onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    onlyfiles2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    onlyfiles1.sort()
    onlyfiles2.sort()
    
    #print("onlyfiles1: ", onlyfiles1[0:10])
    #print("onlyfiles2: ", onlyfiles2[0:10])
    
    file_rm = path1 + "/.DS_Store"
    if os.path.exists(file_rm):
    	os.remove(file_rm)

    file_rm = path2 + "/.DS_Store"
    if os.path.exists(file_rm):
    	os.remove(file_rm)

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


#path_ir = "/Users/user/Desktop/UPMC/EPFL/Research/testing/dataset/SAVE_1_ir0_frames_resized"
#path_vis = "/Users/user/Desktop/UPMC/EPFL/Research/testing/dataset/SAVE_1_visible_frames_resized"



#Working on visible data  SAVE_1_ir0_AFI.zip
path_ir1 = "/Volumes/IEMPROG/dataset/SAVE_1_ir0_AFI"
path_vis1 = "/Volumes/IEMPROG/dataset/SAVE_1_visible_AFI"
prepare_pairs(path_vis1, path_ir1)
#Working on visible data  SAVE_1_ir0_AFI.zip
path_ir2 = "/ivrldata1/students/imad/data/SAVE_2_ir0_AFI"
path_vis2 = "/ivrldata1/students/imad/data/SAVE_2_visible_AFI"
#prepare_pairs(path_vis2, path_ir2)
#Working on visible data  SAVE_2_ir0_HAK.zip
path_ir3 = "/ivrldata1/students/imad/data/SAVE_2_ir0_HAK"
path_vis3 = "/ivrldata1/students/imad/data/SAVE_2_visible_HAK"
#prepare_pairs(path_vis3, path_ir3)
#Working on visible data  SAVE_4_ir0_ARO.zip
path_ir4 = "/ivrldata1/students/imad/data/SAVE_4_ir0_ARO"
path_vis4 = "/ivrldata1/students/imad/data/SAVE_4_visible_ARO"
#prepare_pairs(path_vis4, path_ir4)

#rename(path_ir, 3827)
#rename(path_vis, 3827)