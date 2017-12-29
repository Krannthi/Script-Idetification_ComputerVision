import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics as stat
import sys
import os

def preprocessing(image):

 
	threshold,img = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
	projection=np.zeros(img.shape[0])

	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			if (img[i,j] == 0):
				projection[i]=projection[i]+1
		
	line_list=[]
	linestart =[]
	lineend=[]



	for i in range(1,img.shape[0]):
		if(projection[i-1]==0 and projection[i]>0):
			linestart.append(i)
		if(projection[i-1]>0 and projection[i]==0):
			lineend.append(i)
	
	
	for i in range(0,len(linestart)):
		line_list.append(img[linestart[i]:lineend[i]])
	

	
	
	for i in range(0,len(line_list)):
		line=line_list[i]
		leftend=line.shape[1]
		rightend=line.shape[1]
		for l in range(0,line.shape[0]):
			for m in range(0,leftend):
				if ( line[l,m]==0 and leftend>m):
					leftend=m
			for n in range(0,line.shape[1]):
				p=line.shape[1]-(1+n)
				if(line[l,p]==0):
					if(rightend<p) or rightend==line.shape[1]:
						rightend=p
		line_list[i] = line[0:line.shape[0],leftend:rightend+1]
		line_list[i] = cv2.resize(line_list[i],((40*line.shape[1])/line.shape[0],40), interpolation = cv2.INTER_CUBIC)


	return line_list

def features(line):
    
    top_profile = np.ones(line.shape)
    top_profile_scaled=top_profile
    rows = line.shape[0]
    cols = line.shape[1]


    for col in range(0,cols):
        for row in range(0,rows):
            if line[row,col] == 0:
                top_profile[row,col] = 0
                break


    bottom_profile = np.ones(line.shape)
    bottom_profile_scaled=bottom_profile

    for col in range(0,cols):
        for row in range(0,rows):
            if line[(rows-1)-row,col] == 0:
                bottom_profile[(rows-1)-row,col] = 0
                break

    bottom_profile_scaled=cv2.normalize(bottom_profile,bottom_profile_scaled, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    top_profile_scaled=cv2.normalize(top_profile,top_profile_scaled, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)

    top_max_row = 0
    top_max_row_density = 0.0
    bottom_max_row = 0
    bottom_max_row_density = 0.0

    max_top_pixel_count = 0
    top_profile_pixel_count = 0

    for row in range(0,rows):
        count = 0
        for col in range(0,cols):
            if top_profile[row,col] == 0:
                count = count + 1
        if count > max_top_pixel_count:
            max_top_pixel_count = count
            top_max_row = row


    for row in range(0,rows):
        for col in range(0,cols):
            if top_profile[row,col] == 0:
                top_profile_pixel_count = top_profile_pixel_count + 1


    max_bottom_pixel_count = 0


    for row in range(0,rows):
        count = 0
        for col in range(0,cols):
            if bottom_profile[row,col] == 0:
                count = count + 1
        if count > max_bottom_pixel_count:
            max_bottom_pixel_count = count
            bottom_max_row = row            
            
        
    top_max_row_density = max_top_pixel_count / (cols * 1.0)
    bottom_max_row_density = max_bottom_pixel_count / (cols * 1.0)

    #FEATURE-1
    profile_value = top_max_row_density / ( bottom_max_row_density * 1.0)

    #FEATURE-2
    bottom_max_row_no = bottom_max_row

    #FEATURE-3 (COEFF_PROFILE)

    coeff_top = 0.0
    coeff_bottom = 0.0
    coeff_profile = 0.0

    top_vector = []
    bottom_vector = []

    for col in range(0,cols):
            for row in range(0,rows):
                    if top_profile[row,col] == 0:
                            top_vector.append(row+1)
                            break

    for col in range(0,cols):
            for row in range(0,rows):
                    if bottom_profile[row,col] == 0:
                            bottom_vector.append(row+1)
                            break

    coeff_top = (stat.stdev(top_vector) / (stat.mean(top_vector) * 1.0))* 100

    coeff_bottom = (stat.stdev(bottom_vector) / (stat.mean(bottom_vector) * 1.0)) * 100

    coeff_profile = coeff_top / ( coeff_bottom * 1.0)

    top_component_density = 0.0

    top_component_density = max_top_pixel_count / (top_profile_pixel_count * 1.0)
    
    return (profile_value,bottom_max_row_no,coeff_profile,top_component_density)



def learning(hindi_folder,english_folder,kannada_folder):

    hindi = os.listdir(hindi_folder)
    english = os.listdir(english_folder)
    kannada = os.listdir(kannada_folder)

    print hindi
    print english
    print kannada

    hindi_images = []
    english_images = []
    kannada_images = []

    for g in range(0,len(hindi)):
        hindi_images.append(cv2.imread(hindi_folder+'\\'+hindi[g],0))
        
    for g in range(0,len(english)):
        english_images.append(cv2.imread(english_folder+'\\'+english[g],0))

    for g in range(0,len(kannada)):
        kannada_images.append(cv2.imread(kannada_folder+'\\'+kannada[g],0))

    # training data to bee given for knn. '5' samples for each language    
    training_set = np.zeros((15,4),float)
    # Responses to be given as input to knn.
    responses = np.zeros((15,1)) # hindi-'0' english-'2' kannada-'1'

    hindi_lines = []
    kannada_lines = []
    english_lines = []

    # Adding 500 lines of hindi

    hindi_count = 0

    for i in hindi_images:
        if hindi_count >= 500:
            break
        temp1 = preprocessing(i)
        for j in temp1:
            if hindi_count >= 500:
                break
            else:
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\lines\\' + 'hindi' + str(hindi_count) + '.jpg',j)
                hindi_lines.append(j)
                hindi_count = hindi_count + 1


            
    # Adding 500 lines of English
    
    english_count = 0

    for i in english_images:
        if english_count >= 500:
            break
        temp2 = preprocessing(i)
        for j in temp2:
            if english_count >= 500:
                break
            else:
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\lines\\' + 'english' + str(english_count) + '.jpg',j)
                english_lines.append(j)
                english_count = english_count + 1


    # Adding 500 lines of Kannada
    
    kannada_count = 0

    for i in kannada_images:
        if kannada_count >= 500:
            break
        temp3 = preprocessing(i)
        for j in temp3:
            if kannada_count >= 500:
                break
            else:
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\lines\\' + 'kannada' + str(kannada_count) + '.jpg',j)
                kannada_lines.append(j)
                kannada_count = kannada_count + 1

    
    # adding the features of hindi-lines to the learning-set

    for i in range(0,5):
        features_1 = 0
        features_2 = 0
        features_3 = 0
        features_4 = 0
        for j in range(i*100,(i*100)+100):
            temp = features(hindi_lines[j])
            
            features_1 = features_1 + temp[0]
            features_2 = features_2 + temp[1]
            features_3 = features_3 + temp[2]
            features_4 = features_4 + temp[3]
        training_set[i][0] = features_1/(100 * 1.0)
        training_set[i][1] = features_2/(100 * 1.0)
        training_set[i][2] = features_3/(100 * 1.0)
        training_set[i][3] = features_4/(100 * 1.0)
        responses[i][0] = 0

    print 'completed training hindi'
    # adding the features of kannada_lines to the learning-set

    for i in range(0,5):
        features_1 = 0
        features_2 = 0
        features_3 = 0
        features_4 = 0
        for j in range(i*100,(i*100)+100):
            temp = features(kannada_lines[j])
            features_1 = features_1 + temp[0]
            features_2 = features_2 + temp[1]
            features_3 = features_3 + temp[2]
            features_4 = features_4 + temp[3]
        training_set[i+5][0] = features_1/(100 * 1.0)
        training_set[i+5][1] = features_2/(100 * 1.0)
        training_set[i+5][2] = features_3/(100 * 1.0)
        training_set[i+5][3] = features_4/(100 * 1.0)
        responses[i+5][0] = 1
  
    print 'completed training kannada'
    # adding the features of english-lines to the learning-set

    for i in range(0,5):
        features_1 = 0
        features_2 = 0
        features_3 = 0
        features_4 = 0
        #print i
        for j in range(i*100,(i*100)+100):
            #print j
            #cv2.imwrite('3lin.jpg',english_lines[j])
            temp = features(english_lines[j])
            features_1 = features_1 + temp[0]
            features_2 = features_2 + temp[1]
            features_3 = features_3 + temp[2]
            features_4 = features_4 + temp[3]
        training_set[i+10][0] = features_1/(100 * 1.0)
        training_set[i+10][1] = features_2/(100 * 1.0)
        training_set[i+10][2] = features_3/(100 * 1.0)
        training_set[i+10][3] = features_4/(100 * 1.0)
        responses[i+10][0] = 2

    print 'completed training english'

    return (training_set,responses)


def Testing(hindi_test,english_test,kannada_test,training_set,responses):

    hindi = os.listdir(hindi_test)
    english = os.listdir(english_test)
    kannada = os.listdir(kannada_test)

    print hindi
    print english
    print kannada
    
    hindi_images = []
    english_images = []
    kannada_images = []

    for g in range(0,len(hindi)):
        hindi_images.append(cv2.imread(hindi_test+'\\'+hindi[g],0))

    print 'finished reading hindi'
        
    for g in range(0,len(english)):
        english_images.append(cv2.imread(english_test+'\\'+english[g],0))
    print 'finished reading english'

    for g in range(0,len(kannada)):
        kannada_images.append(cv2.imread(kannada_test+'\\'+kannada[g],0))
        
    print 'finished reading kannada'
    hindi_lines = []
    kannada_lines = []
    english_lines = []

    # Adding testing lines of hindi

    hindi_count = 0

    for i in hindi_images:
        temp1 = preprocessing(i)
        for j in temp1:
                #print "here"
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\Testing\\' + 'hindi' + str(hindi_count) + '.jpg',j)
                hindi_lines.append(j)
                hindi_count = hindi_count + 1


            
    # Adding testing lines of English
    
    english_count = 0

    for i in english_images:
        temp2 = preprocessing(i)
        for j in temp2:
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\Testing\\' + 'english' + str(english_count) + '.jpg',j)
                english_lines.append(j)
                english_count = english_count + 1


    # Adding testing lines of Kannada
    
    kannada_count = 0

    for i in kannada_images:
        temp3 = preprocessing(i)
        for j in temp3:
                cv2.imwrite('C:\Users\kgopu\Desktop\Computer-Vision\project\Testing\\' + 'kannada' + str(kannada_count) + '.jpg',j)
                kannada_lines.append(j)
                kannada_count = kannada_count + 1

    
    knn = cv2.KNearest()
    hindi_testing_data = np.zeros((len(hindi_lines),4))
    english_testing_data = np.zeros((len(english_lines),4))
    kannada_testing_data = np.zeros((len(kannada_lines),4))
    
    
    for j in range(0,len(hindi_lines)):
            hindi_testing_data[j][0],hindi_testing_data[j][1],hindi_testing_data[j][2],hindi_testing_data[j][3] = features(hindi_lines[j])

    for j in range(0,len(english_lines)):
            english_testing_data[j][0],english_testing_data[j][1],english_testing_data[j][2],english_testing_data[j][3] = features(english_lines[j])

    for j in range(0,len(kannada_lines)):
            kannada_testing_data[j][0],kannada_testing_data[j][1],kannada_testing_data[j][2],kannada_testing_data[j][3] = features(kannada_lines[j])


    knn.train(np.float32(training_set),np.float32(responses))

    print 'hindi'
    print "total_hindi_lines",hindi_count
        # testing hindi lines
    knn3_results = knn.find_nearest(np.float32(hindi_testing_data),3)

    count = 0
    for i in knn3_results[1]:
            if i == [0]:
                    count = count + 1
    
    print "correctly predicted hindi lines using 3 neighbors",count
                    
    knn5_results = knn.find_nearest(np.float32(hindi_testing_data),5)

    count = 0
    for i in knn5_results[1]:
            if i == [0]:
                    count = count + 1

    print "correctly predicted hindi lines using 5 neighbors",count
   
    knn7_results = knn.find_nearest(np.float32(hindi_testing_data),7)

    count = 0
    for i in knn7_results[1]:
            if i == [0]:
                    count = count + 1

    print "correctly predicted hindi lines using 7 neighbors",count

    print 'english'
    print "total-english-count",english_count

        # testing english lines
    knn3_results = knn.find_nearest(np.float32(english_testing_data),3)

    count = 0
    for i in knn3_results[1]:
            if i == [2]:
                    count = count + 1
    
    print "correctly predicted english lines using 3 neighbors",count
                    
    knn5_results = knn.find_nearest(np.float32(english_testing_data),5)

    count = 0
    for i in knn5_results[1]:
            if i == [2]:
                    count = count + 1

    print "correctly predicted english lines using 5 neighbors",count
  
    knn7_results = knn.find_nearest(np.float32(english_testing_data),7)

    count = 0
    for i in knn7_results[1]:
            if i == [2]:
                    count = count + 1

    print "correctly predicted english lines using 7 neighbors",count

    print 'kannada'
    print "total-kannada-count",kannada_count


        # testing kannada lines
    knn3_results = knn.find_nearest(np.float32(kannada_testing_data),3)

    count = 0
    for i in knn3_results[1]:
            if i == [1]:
                    count = count + 1
    
    print "correctly predicted kannada lines using 3 neighbors",count    
                    
    knn5_results = knn.find_nearest(np.float32(kannada_testing_data),5)

    count = 0
    for i in knn5_results[1]:
            if i == [1]:
                    count = count + 1

    print "correctly predicted kannada lines using 5 neighbors",count
    
    knn7_results = knn.find_nearest(np.float32(kannada_testing_data),7)

    count = 0
    for i in knn7_results[1]:
            if i == [1]:
                    count = count + 1

    print "correctly predicted kannada lines using 7 neighbors",count



tset,response=learning('C:\Users\kgopu\Desktop\Computer-Vision\project\Hindi_Training','C:\Users\kgopu\Desktop\Computer-Vision\project\English_Training','C:\Users\kgopu\Desktop\Computer-Vision\project\Kannada_Training')
Testing('C:\Users\kgopu\Desktop\Computer-Vision\project\Hindi_Testing','C:\Users\kgopu\Desktop\Computer-Vision\project\English_Testing','C:\Users\kgopu\Desktop\Computer-Vision\project\Kannada_Testing',tset,response)

print tset

