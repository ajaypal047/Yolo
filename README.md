#How to store & transfer time-series data efficiently.

#What is Bayes classifier?  

#Write a code to implement queue using stacks?  

#How do you handle continuous large amount of data input collected from sensors?   

#Write a code to implement queue using stacks?

#was asked to write a program to combine a time series data

#How will you optimize the upload of a Time-series sensor data from phone to server?



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
import random

from sklearn.model_selection import train_test_split 

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
THRESHOLD = 0.6



fromfile = 'test/person.jpg'
labelFile = 'F:/YoloTensorflow/myTest/train/object-detection-crowdai/labels.csv'
datadir = 'F:/YoloTensorflow/myTest/train/object-detection-crowdai/'
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
classes =  ["Car","Truck","Pedestrian"]
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

def build_model(x):

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
	fc_33 = fc_layer(33,fc_32,2048,flat=False,linear=True)

	print("Loading complete!" + '\n')
	return sess, fc_33, saver

def getFiles(data):
	filelist = []
	for file in os.listdir(data):
		if file.endswith(".npy"):
			filelist.append(os.path.join(data+"/", file))

	random.shuffle(filelist)

	train, test = train_test_split(filelist, test_size=0.33, random_state=42) 
	return train, test

def loss(outPut, trainlabels):
	lossNum = 0
	for i in range(0,2048,8):
		coordLoss = LAMBDA_COORD * tf.reduce_sum(tf.multiply(trainlabels[ : ,i+0],tf.add(tf.square(tf.subtract(trainlabels[ : ,i+3],outPut[ : ,i+3])),(tf.square(tf.subtract(trainlabels[ : ,i+4],outPut[ : ,i+4]))))))

		sizeLoss = LAMBDA_COORD * tf.reduce_sum(tf.multiply(trainlabels[ : ,i+0],tf.add(tf.square(tf.subtract(tf.sqrt(trainlabels[ : ,i+1]),tf.sqrt(outPut[ : ,i+1]))),(tf.square(tf.subtract(tf.sqrt(trainlabels[ : ,i+2]),tf.sqrt(outPut[ : ,i+2])))))))

		objectnessLoss = tf.reduce_sum(tf.multiply(trainlabels[ : ,i+0],tf.square(tf.subtract(trainlabels[ : ,i+0],outPut[ : ,i+0]))))

		noObjectnessLoss = LAMBDA_NOOBJ*tf.reduce_sum(tf.multiply(tf.subtract(1.0,trainlabels[ : ,i+0]),tf.square(tf.subtract(trainlabels[ : ,i+0],outPut[ : ,i+0]))))

		probLoss = tf.reduce_sum(tf.multiply(trainlabels[ : ,i+0],tf.reduce_sum(tf.square(tf.subtract(trainlabels[ : ,i+5:i+7],outPut[ : ,i+5:i+7])),1)))

		lossNum = lossNum + tf.reduce_sum(lossNum + coordLoss + sizeLoss + objectnessLoss + noObjectnessLoss + probLoss)

	return lossNum
		

def train():

	trainSet, valSet = getFiles('F:/YoloTensorflow/myTest/loadedLabels/')
	
	trainInput = tf.placeholder('float32',[None,448,448,3])
	
	sess, outPut, saver = build_model(trainInput)
	
	trainlabels = tf.placeholder('float32',[None,2048])
	
	loss_op = loss(outPut, trainlabels)  
	
	trainOpt = tf.train.AdamOptimizer(0.0005).minimize(loss_op, var_list=[var for var in tf.trainable_variables()])

	#saver.restore(sess, "updated.ckpt")

	for trainFile in trainSet:
		trainSample = np.load(trainFile)
		x_input = []
		labels = []
		for x in trainSample:
			x_input.append(cv2.resize(cv2.imread(x['image']),(448,448)))
			print(x['labels'])
			labels.append(x['labels'])

			feed_dict = {trainInput: x_input, trainlabels: labels}  
			fetched = sess.run([trainOpt, loss_op], feed_dict)
			saver.save(sess,"updated.ckpt")
			print("loss ",fetched[1])
	    # self.writer.add_summary(fetched[2], i)

	    # ckpt = (i+1) % 10 #(self.FLAGS.save // self.FLAGS.batch)
	    # args = [step_now, profile]
	    # if not ckpt: _save_ckpt(self, *args) 


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
	# anchorBox= []
	
	# with open('/Users/claw/Downloads/tensorFlow/object-detection-crowdai/labels.csv','r') as f: 
	# 	next(f)
	# 	reader = csv.reader(f)
	# 	for row in reader:
	# 		anchorBox.append([(int(row[2])-int(row[0]))*448.0/w_img/64.0,(int(row[3])-int(row[1]))*448.0/h_img/64.0])
	
	if(os.path.isfile('FinalLabels.npy')):
		print("labels loaded")
		return np.load('FinalLabels.npy')

	else:
		anchorBoxes = np.array([[0.15677083, 0.2275], [ 0.4484375, 0.65333333]])#kmeans(np.array(anchorBox),2)
		
		labels =[]
		labelsFinal = []
		imageName = []
		# recTangle = []

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

				if(imageNameTemp == ''):
					imageNameTemp = imageName

				width = float(int(row[2])-int(row[0]))*448.0/w_img/16.0
				height = float(int(row[3])-int(row[1]))*448.0/h_img/16.0
				xCenter = float(int(row[2])+int(row[0]))//2.0
				yCenter = float(int(row[3])+int(row[1]))//2.0
				xCenter = xCenter*448.0/w_img
				yCenter = yCenter*448.0/h_img

				xloc = 0
				yloc = 0
				xCenterGrid = 0
				yCenterGrid = 0

				for i in range(28):
					
					diffX = xCenter-i*16
					diffY = yCenter-i*16

					if(diffX<0):
						xCenterGrid = diffX+16
						xloc = i-1
						xCenter = 448

					if(diffY<0):
						yCenterGrid = diffY+16
						yloc = i-1
						yCenter = 448


				if(imageNameTemp!=imageName):
					labelsFinal.append({'ImageName': imageNameTemp,'labels':labels})
					labels = []
					imageNameTemp = imageName
				labels.append({'hw': [width,height],'coordinates': [float(xCenterGrid/16.0),float(yCenterGrid/16.0)],'class': row[5].lower(),'grid' : [xloc,yloc]})

			labelsFinal.append({'ImageName': imageName,'labels':labels})
		#print("label ",labelsFinal)
		Matrixlabel = np.zeros((8))
		FinalLabels = []
		oneHot = []
		countLabel = 1
		countName = 0
		noLabels = len(labelsFinal)
		for lauda in labelsFinal:
			for y in range(28):
				for x in range(28):
					for l in lauda['labels']:
						if(l['grid']==[x,y]):
							Matrixlabel[0] = 1
							Matrixlabel[1] = l['hw'][0]
							Matrixlabel[2] = l['hw'][1]
							Matrixlabel[3] = l['coordinates'][0]
							Matrixlabel[4] = l['coordinates'][1]
							for lo in range(len(classes)):
								if(classes[lo].lower() == l['class'].lower()):
									Matrixlabel[5+lo] = 1.0
					oneHot.extend(Matrixlabel)
					Matrixlabel = np.zeros((8))
			FinalLabels.append({'image': datadir + lauda['ImageName'],'labels':oneHot})
			oneHot = []
			if(countLabel%100==0 or countLabel== noLabels):
				np.save('F:/YoloTensorflow/myTest/loadedLabels/Final_labels_'+str(countName)+'.npy', FinalLabels)
				FinalLabels = []
				countName = countName+1
			countLabel=countLabel+1

		print("labels saving done")
		return FinalLabels

#def loss(pred,labels):				
		
def drawLabel(labelsFinal):
	print("drawing label")
	imageName = labelsFinal['image']
	rect = []
	coordinate = []
	text = []

	img = cv2.imread(imageName)

	img = cv2.resize(img,(448,448))

	font = cv2.FONT_HERSHEY_SIMPLEX

	label = labelsFinal['labels']
	
	j=0
	k=0

	for i in range(0,len(label),8):
			if(label[i]==1):
				pixel_locX = label[i+3]*16.0+j*16.0
				pixel_locY = label[i+4]*16.0+k*16.0
				width = label[i+1]*16.0
				height = label[i+2]*16.0
				classNo = classes[np.argmax(label[i+5:i+8])]
				rect.append([width//2.0,height//2.0])
				coordinate.append([pixel_locX,pixel_locY])
				text.append(classNo)
			j=j+1
			if(j%28==0):
				j=0
				k=k+1

	for i in range(len(rect)):
		cv2.rectangle(img, (int(coordinate[i][0]-rect[i][0]),int(coordinate[i][1]-rect[i][1])), (int(coordinate[i][0]+rect[i][0]),int(coordinate[i][1]+rect[i][1])), (0,255,0), 2)
		cv2.putText(img, text[i], (int(coordinate[i][0]),int(coordinate[i][1])), font, 0.1, (0, 255, 0), 2, cv2.LINE_AA)

	cv2.imshow("1",img)
	cv2.waitKey(0)		
			

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


if __name__ == '__main__':
	train()
