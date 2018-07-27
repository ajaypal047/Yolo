import tensorflow as tf
import os
import time
import argparse
import cv2
import numpy as np
import csv
import copy
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt  



fromfile = 'test/person.jpg'
labelFile = 'labels.csv'
tofile_img = 'test/output.jpg'
tofile_txt = 'test/output.txt'
imshow = True
filewrite_img = False
filewrite_txt = False
disp_console = True
weights_file = 'weights/YOLO_small.ckpt'
alpha = 0.1
threshold = 0.2
iou_threshold = 0.5
num_class = 20
num_box = 2
grid_size = 7
classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

w_img = 1920.0
h_img = 1200.0

def conv_layer(idx,inputs,filters,size,stride):
	channels = inputs.get_shape().as_list()[3]
	weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters],stddev=0.1),dtype='float32')
	biases = tf.Variable(tf.constant(0.1,shape=[filters]),dtype='float32')
	
	padsize = size//2
	padmat = np.array([[0,0],[padsize,padsize],[padsize,padsize],[0,0]])
	input_pad = tf.pad(inputs,padmat)
	
	conv = tf.nn.conv2d(input_pad,weight,strides=[1,stride,stride,1],padding='VALID')
	conv = tf.add(conv,biases)
	
	return tf.nn.leaky_relu(conv,alpha=0.1,name=str(idx)+'_conv_relu')
	
def pooling_layer(idx,input,size,stride):
	return tf.nn.max_pool(input,ksize = [1,size,size,1],strides = [1,stride,stride,1],padding = 'SAME',name = str(idx)+'_pool')

def fc_layer(idx,input,hidden,flat = False, linear = False):

	if(flat):
		[_ , height, width, channels] = input.get_shape().as_list()
		dim = height*width*channels
		input = tf.transpose(input,(0,3,1,2))
		input = tf.reshape(input,[-1,dim])
	else:
		dim = input.get_shape().as_list()[1]
	
	weight = tf.Variable(tf.truncated_normal([dim,hidden],stddev = 0.1),dtype='float32')
	biases = tf.Variable(tf.constant(0.1,shape=[hidden]),dtype='float32')
	
	if(linear):
		return tf.add(tf.matmul(input, weight),biases,name= str(idx)+'_FC')
	else:
		return tf.nn.leaky_relu(tf.add(tf.matmul(input, weight),biases),alpha=0.1,name= str(idx)+'_FC_Relu')	

def build_model():
	x = tf.placeholder('float32',[None,448,448,3])
	conv_1 = conv_layer(1,x,64,7,2)
	pool_2 = pooling_layer(2,conv_1,2,2)
	conv_3 = conv_layer(3,pool_2,192,3,1)
	pool_4 = pooling_layer(4,conv_3,2,2)
	conv_5 = conv_layer(5,pool_4,128,1,1)
	conv_6 = conv_layer(6,conv_5,256,3,1)
	conv_7 = conv_layer(7,conv_6,256,1,1)
	conv_8 = conv_layer(8,conv_7,512,3,1)
	pool_9 = pooling_layer(9,conv_8,2,2)
	conv_10 = conv_layer(10,pool_9,256,1,1)
	conv_11 = conv_layer(11,conv_10,512,3,1)
	conv_12 = conv_layer(12,conv_11,256,1,1)
	conv_13 = conv_layer(13,conv_12,512,3,1)
	conv_14 = conv_layer(14,conv_13,256,1,1)
	conv_15 = conv_layer(15,conv_14,512,3,1)
	conv_16 = conv_layer(16,conv_15,256,1,1)
	conv_17 = conv_layer(17,conv_16,512,3,1)
	conv_18 = conv_layer(18,conv_17,512,1,1)
	conv_19 = conv_layer(19,conv_18,1024,3,1)
	pool_20 = pooling_layer(20,conv_19,2,2)
	conv_21 = conv_layer(21,pool_20,512,1,1)
	conv_22 = conv_layer(22,conv_21,1024,3,1)
	conv_23 = conv_layer(23,conv_22,512,1,1)
	conv_24 = conv_layer(24,conv_23,1024,3,1)
	conv_25 = conv_layer(25,conv_24,1024,3,1)
	conv_26 = conv_layer(26,conv_25,1024,3,2)
	conv_27 = conv_layer(27,conv_26,1024,3,1)
	conv_28 = conv_layer(28,conv_27,1024,3,1)
	fc_29 = fc_layer(29,conv_28,512,flat=True,linear=False)
	fc_30 = fc_layer(30,fc_29,4096,flat=False,linear=False)
	#skip dropout_31
	fc_32 = fc_layer(32,fc_30,1470,flat=False,linear=True)
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	saver.restore(sess,weights_file)
	print("Loading complete!" + '\n')
	return sess, fc_32, x

def detect_from_file(filename,sess,fc_32,x):
	img = cv2.imread(filename)
	s = time.time()
	h_img,w_img,_ = img.shape
	img_resized = cv2.resize(img, (448, 448))
	img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
	img_resized_np = np.asarray( img_RGB )
	inputs = np.zeros((1,448,448,3),dtype='float32')
	inputs[0] = (img_resized_np/255.0)*2.0-1.0
	in_dict = {x: inputs}
	net_output = sess.run(fc_32,feed_dict=in_dict)
	result = interpret_output(net_output[0])
	show_results(img,result)
	strtime = str(time.time()-s)

def interpret_output(output):
	probs = np.zeros((7,7,2,20))
	class_probs = np.reshape(output[0:980],(7,7,20))
	scales = np.reshape(output[980:1078],(7,7,2))
	boxes = np.reshape(output[1078:],(7,7,2,4))
	offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

	boxes[:,:,:,0] += offset
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
	
	boxes[:,:,:,0] *= w_img
	boxes[:,:,:,1] *= h_img
	boxes[:,:,:,2] *= w_img
	boxes[:,:,:,3] *= h_img

	for i in range(2):
		for j in range(20):
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

	filter_mat_probs = np.array(probs>=threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	probs_filtered = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

	argsort = np.array(np.argsort(probs_filtered))[::-1]
	boxes_filtered = boxes_filtered[argsort]
	probs_filtered = probs_filtered[argsort]
	classes_num_filtered = classes_num_filtered[argsort]
	
	for i in range(len(boxes_filtered)):
		if probs_filtered[i] == 0 : continue
		for j in range(i+1,len(boxes_filtered)):
			if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
				probs_filtered[j] = 0.0
	
	filter_iou = np.array(probs_filtered>0.0,dtype='bool')
	boxes_filtered = boxes_filtered[filter_iou]
	probs_filtered = probs_filtered[filter_iou]
	classes_num_filtered = classes_num_filtered[filter_iou]

	result = []
	for i in range(len(boxes_filtered)):
		result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

	return result

def show_results(img,results):
	img_cp = img.copy()
	if filewrite_txt :
		ftxt = open(tofile_txt,'w')
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2
		if filewrite_img or imshow:
			cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
			cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
		if filewrite_txt :				
			ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
	if filewrite_img: 
		cv2.imwrite(tofile_img,img_cp)			
	if imshow :
		cv2.imshow('YOLO_small detection',img_cp)
		cv2.waitKey(1)
	if filewrite_txt : 
		ftxt.close()

def iou_(box, clusters):

    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
	
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou = intersection / (box_area + cluster_area - intersection)
	
    return iou

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou_(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def readLabelData():
	labelList = []
	anchorBox= []
	
	with open('D:\DA\YoloTensorflow\myTest\labels.csv','r') as f: 
		next(f)
		reader = csv.reader(f)
		for row in reader:
			anchorBox.append([(int(row[2])-int(row[0]))*448.0/w_img/64.0,(int(row[3])-int(row[1]))*448.0/h_img/64.0])
	
	anchorBoxes = kmeans(np.array(anchorBox),2)
	
	labels =[]
	labelsFinal = []

	with open(labelFile,'r') as f:
		next(f)
		reader = csv.reader(f)
		count = 0
		imageNameTemp = ''
		for row in reader:
			if(count==0):
				imageNameTemp = row[4]
				count=count+1
			
			imageName = row[4]
			
			if(imageName != imageNameTemp):
				labelsFinal.append({'ImageName': imageNameTemp,'labels':labels, })
				imageNameTemp = imageName
				labels = []
				
			width = float(int(row[2])-int(row[0]))*448.0/w_img/64.0
			height = float(int(row[3])-int(row[1]))*448.0/h_img/64.0
			xCenter = float(int(row[2])+int(row[0]))/2.0
			yCenter = float(int(row[3])+int(row[1]))/2.0

			xCenter = xCenter*448.0/w_img
			yCenter = xCenter*448.0/h_img

			xloc = 0
			yloc = 0
			xCenterGrid = 0
			yCenterGrid = 0

			for i in range(8):
				diffX = xCenter-i*64
				diffY = yCenter-i*64

				if(diffX<0):
					xCenterGrid = diffX+64
					xCenter = 448
					xloc = i-1
				if(diffY<0):
					yCenterGrid = diffY+64
					yCenter = 448
					yloc = i-1
			

			labels.append({'hw': [width,height],'coordinates': [float(xCenterGrid/64.0),float(yCenterGrid/64.0)],'class': row[5].lower(),'grid' : [xloc,yloc]})
		labelsFinal.append({'ImageName': imageNameTemp,'labels':labels})
		labels = []
		
		Matrixlabel = np.zeros((50))
		
		oneHot = []
		count = 0
		for lauda in labelsFinal:
			if(count==0):
				print("imageName",lauda['ImageName'])
				for l in lauda['labels']:
					if(count==0):
						for y in range(7):
							for x in range(7):
								if(l['grid']==[x,y]):
									print("grid " , l['grid'])
									iouCheck = iou_(l['hw'],anchorBoxes)
									if(iouCheck[0]>iouCheck[1]):
										Matrixlabel[0] = 1
										Matrixlabel[1] = l['hw'][0]
										Matrixlabel[2] = l['hw'][1]
										Matrixlabel[3] = l['coordinates'][0]
										Matrixlabel[4] = l['coordinates'][1]
										for lo in range(len(classes)):
											if(classes[lo] == l['class']):
												Matrixlabel[5+lo] = 1.0
									else:
										Matrixlabel[25] = 1
										Matrixlabel[26] = l['hw'][0]
										Matrixlabel[27] = l['hw'][1]
										Matrixlabel[28] = l['coordinates'][0]
										Matrixlabel[29] = l['coordinates'][1]
										for lo in range(len(classes)):
											if(classes[lo] == l['class']):
												Matrixlabel[30+lo] = 1.0
								oneHot.extend(Matrixlabel)
								Matrixlabel = np.zeros((50))
					count+=1
		
			
		print("mat ", oneHot)
					



		# with open('csvfile.csv','w') as csv_file:
			# writer = csv.writer(csv_file)
			# for june in labelsFinal:
				# writer.writerow(june['ImageName'])
				
		
		
			
	



def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--Directory',type = str, default= None, help = 'Please provide the input directory')
	# sess, fc_32, x = build_model()
	# detect_from_file(fromfile,sess,fc_32,x)
	# cv2.waitKey(1000)
	readLabelData()
