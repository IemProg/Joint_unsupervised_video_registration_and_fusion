import os, sys

path = "/ivrldata1/students/imad/Joint_unsupervised_video_registration_and_fusion"
def remove_jpg(path):
	onlyfiles1 = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	for f in os.listdir(path):
		file = os.path.join(path, f)
		if os.path.isfile(file):
			#print("File: ", file)
			if file.endswith("jpg") or file.endswith("png"):
				os.remove(file)
				print("File: '{}' has been removed!".format(file))
		head, tail = os.path.split(file)
		if tail.startswith("test"):
				os.rmdir(file)
				print("File: '{}' has been removed!".format(file))

				
remove_jpg(path)