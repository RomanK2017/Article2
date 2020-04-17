
#import matplotlib.pyplot as plt 
import numpy as np
import cv2
import scipy.stats
from itertools import chain
import os


def Get_KS_Pi(ext_c_img,Pi_xy,A):
    '''
    P1,P2,P3,P4 - corners of the rotated enclosing rectangle
    calculate arrays of distances from each Pi to each point of a filled outer contour
    get a sorted array of pairs (Pi,Pj) and KS test 
    di -list of distances from cornets to all not black pixels of mask(filled outer contour)
    pairs - list of unique combinations of corners, 6 pairs
    KS_Pi- sortet list of KS DISTANCES(not p)!! of pairs of di, 6 numbers
    It is supposed that similar not circular contours have similar KS_Pi
    '''
    di=[]
    tmp = np.where(ext_c_img != [0])
    for point in Pi_xy:
        Cx=point[0]
        Cy=point[1]
        di.append(np.sqrt( ((tmp[1]-Cx)**2+(tmp[0]-Cy)**2)/A))
    pairs=[]
    for i in range (0,len(Pi_xy)-1):
        for j in range (i+1,len(Pi_xy)):
            pairs.append([i,j])
    KS_Pi=[]
    for p in pairs:
         KS_Pi.append(scipy.stats.ks_2samp(di[p[0]], di[p[1]])[0])
    KS_Pi.sort(reverse = True)
    return KS_Pi

def Matching_Criteria(KS_test_p, DC_test, Criterion3,rc,noise_level=0.0):
# Matching images. 
#rc - minimal size ratio of a couple of images of (h1/h2,h2/h1,w1/w2,w2/w1)                                                                      
    Matched_C='N'
    #
    KS_threshold=10**((-1.0)*(20+15*(1-rc)+0.05*noise_level))
    DC_threshold=5+0.1*noise_level+2*(1-rc)
    if (KS_test_p>=KS_threshold) and (DC_test<=DC_threshold) and (Criterion3=='Y'):
        Matched_C='Y'    
    return Matched_C


def Criterion3_test(Areas1,Dist1,Aext1,Areas2,Dist2,Aext2,t1=10,t2=10,t_min_area=5):
#Criterion3 test - similarity of internal contour's structure
#Areas1,Areas2 - desc. sorted lists of areas of internal contours, always <100%
# input areas arrays have contours >1% of area
#Dist1, Diar2 - desc. sorted lists of distances between internal contours 100%*[dist[i]/ER], always <100%
#Aext1,Aext2 - areas of outer contour; t1, t2 - criterion's thresholds
#t_min_area - minimal total sum on internal contour areas (% of ext contour) we assume that contours exist
#Lists MUST BE sortet!!!
    C3='N'
    C3_areas='N'
    C3_location='N'
    if np.sum(Areas1)<=t_min_area and np.sum(Areas2)<=t_min_area:
        # if contours in total less than 5% we assume it is noise/distortion. Neglect 
        C3='Y'
        return C3
   
    n=min(len(Areas1),len(Areas2))
    LOD=[abs(Areas1[i]-Areas2[i]) for i in range(0,n)]
    # Check that all LOD elements are less than t1 and return areas test C3_areas='Y'
    if sum(map(lambda x : x>t1, LOD))==0:
        C3_areas='Y'
    #check that all contour locations ()distances are similar, test C3_location='Y'
    # look at n biggest contours if len(Dist1)<>len(Dist2)
    n=min(len(Dist1),len(Dist2))
    LOdist=[abs(Dist1[i]-Dist2[i]) for i in range(0,n)]
    if sum(map(lambda x : x>t2, LOdist))==0:
        C3_location='Y'
    if C3_areas=='Y' and C3_location=='Y':
        C3='Y'    
    return C3



def Test_Images (im1,im2):
    #Gets image -- image description
    #read and resize images

    h1, w1, c = im1.shape
    hmax, wmax, c = im2.shape
    rc=min(1.0*h1/hmax,1.0*hmax/h1,1.0*w1/wmax,1.0*wmax/w1)
    if h1>hmax:
        hmax=h1
    if w1>wmax:
        wmax=w1
    image1 = cv2.resize(im1, (wmax, hmax), interpolation = cv2.INTER_AREA)
    image2 = cv2.resize(im2, (wmax, hmax), interpolation = cv2.INTER_AREA)
    
    # The first image. brightness only 0 or 255, then select the largest contour and exclude all rubbish
    tmp_im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.normalize(tmp_im1,tmp_im1,0,255,cv2.NORM_MINMAX)
    ret,tmp_im_gray1=cv2.threshold(
            tmp_im1, np.mean([i for i in list(chain.from_iterable(tmp_im1)) if i>0]), 255, cv2.THRESH_BINARY) 
    mask1 = np.zeros(tmp_im_gray1.shape, np.uint8)
      
    contours, hierarchy = cv2.findContours(tmp_im_gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    A1=cv2.contourArea(contours[0])
    cont1=contours[0]
    for c in contours:
        if cv2.contourArea(c)>A1:
            A1=cv2.contourArea(c)
            cont1=c
    cv2.drawContours(mask1, [cont1], -1, (255),-1)
    
    #get rotated encl. rectangle of outer contour of image1
    rect1 = cv2.minAreaRect(cont1)
    box11 = cv2.cv.BoxPoints(rect1)
    box1 = np.int0(box11)
    rwidth1=min(rect1[1][0],rect1[1][1])
    rheigth1=max(rect1[1][0],rect1[1][1])
    (x1,y1),radius1 = cv2.minEnclosingCircle(cont1)
    center1 = (x1,y1)
    Pi_KS1=Get_KS_Pi(mask1,box11,A1)
    
    #get aras of all internal contours
    contours1,hierarchy1=cv2.findContours(tmp_im_gray1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes1 = [contours1[i] for i in range(len(contours1)) if hierarchy1[0][i][3] >= 0]
    Internal_cont_r_areas1=sorted([100.0*cv2.contourArea(x)/A1 for x in holes1 if 100.0*cv2.contourArea(x)/A1>=1.0], reverse=True)
    Internal_cont_centres1=sorted([[int(cv2.moments(x)["m10"] / cv2.moments(x)["m00"]),
                                    int(cv2.moments(x)["m01"] / cv2.moments(x)["m00"])] for x in holes1 if 100.0*cv2.contourArea(x)/A1>=1.0], reverse=True)
    inter_c_DCREC1=sorted([(100.0/radius1)*np.sqrt((
            Internal_cont_centres1[i1][0]-Internal_cont_centres1[i2][0])**2+
    (Internal_cont_centres1[i1][1]-Internal_cont_centres1[i2][1])**2)  
    for i1 in range(0,len(Internal_cont_centres1)-1) for 
                   i2 in range(i1,len(Internal_cont_centres1)) if i1!=i2], reverse=True)
  
    #mask- filled outer contour
    #tmp_im_gray1=mask1
    M1 = cv2.moments(cont1)
    Cx1 = int(M1["m10"] / M1["m00"])
    Cy1 = int(M1["m01"] / M1["m00"])
    tmp = np.where(mask1 != [0])
    d1=np.sqrt( ((tmp[1]-Cx1)**2+(tmp[0]-Cy1)**2)/A1)
    
    #The second image, the same processing as the first image
    tmp_im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.normalize(tmp_im2,tmp_im2,0,255,cv2.NORM_MINMAX)
    ret,tmp_im_gray2=cv2.threshold(
            tmp_im2, np.mean([i for i in list(chain.from_iterable(tmp_im2)) if i>0]), 255, cv2.THRESH_BINARY)
    mask2 = np.zeros(tmp_im_gray2.shape, np.uint8)
    contours, hierarchy = cv2.findContours(tmp_im_gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours)==0:
        test={'Area_test':0,
          'minEnclosingCircle_test':0,
          'Equi_diameter_test':0,
          'K-S_test_p':0}
        print '!!! Contours not found:image2!!!'
        return test,'N'    
    A2=cv2.contourArea(contours[0])
    cont2=contours[0]
    for c in contours:
        if cv2.contourArea(c)>A2:
            A2=cv2.contourArea(c)
            cont2=c
    cv2.drawContours(mask2, [cont2], -1, (255),-1)
    
    #get rotated encl. rectangle of outer contour of image2
    rect2 = cv2.minAreaRect(cont2)
    box22 = cv2.cv.BoxPoints(rect2)
    box2 = np.int0(box22)
    rwidth2=min(rect2[1][0],rect2[1][1])
    rheigth2=max(rect2[1][0],rect2[1][1])
    (x2,y2),radius2 = cv2.minEnclosingCircle(cont2)
    center2 = (x2,y2)
    Pi_KS2=Get_KS_Pi(mask2,box22,A2)
    
    #get aras of all internal contours
    contours2,hierarchy2=cv2.findContours(tmp_im_gray2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes2 = [contours2[i] for i in range(len(contours2)) if hierarchy2[0][i][3] >= 0]
    Internal_cont_r_areas2=sorted([100.0*cv2.contourArea(x)/A2 for x in holes2 if 100.0*cv2.contourArea(x)/A2>=1.0], reverse=True)
    Internal_cont_centres2=sorted([[int(cv2.moments(x)["m10"] / cv2.moments(x)["m00"]),
                                    int(cv2.moments(x)["m01"] / cv2.moments(x)["m00"])] for x in holes2 if 100.0*cv2.contourArea(x)/A2>=1.0], reverse=True)
    inter_c_DCREC2=sorted([(100.0/radius2)*np.sqrt((
            Internal_cont_centres2[i1][0]-Internal_cont_centres2[i2][0])**2+
    (Internal_cont_centres2[i1][1]-Internal_cont_centres2[i2][1])**2)  
    for i1 in range(0,len(Internal_cont_centres2)-1) for 
                   i2 in range(i1,len(Internal_cont_centres2)) if i1!=i2], reverse=True)    

    #tmp_im_gray2=mask2    
    M2 = cv2.moments(cont2)
    Cx2 = int(M2["m10"] / M2["m00"])
    Cy2 = int(M2["m01"] / M2["m00"])
    tmp = np.where(mask2 != [0])
    d2=np.sqrt( ((tmp[1]-Cx2)**2+(tmp[0]-Cy2)**2)/A2)
    
    #Estimate possible circularity of 2 external contours from image1 and image2
    #case 1 - if their rwidth/rheigth ratios difference >10% then we assume that
    #contours are different and skip other tests, doesn't matter they are circular or not
    if 100.0*abs(rwidth1/rheigth1-rwidth2/rheigth2)>10.0:
        #case 1
        test={'K-S_test_p':0,'DC_test':0, 'Criterion3':0,'P4_test':0,'Case':'Case1'}
        Match_test='N'
        return test,Match_test
    #case 2 - they both aren't circular
    #circular rwidth/rheigth=1, we assume - possible circular symmetry
    if (100.0*abs(1.0-rwidth1/rheigth1)>5.0 or 100.0*abs(1.0-rwidth2/rheigth2)>5.0):
        P4_test=scipy.stats.ks_2samp(Pi_KS1,Pi_KS2)[1]
        KS_d1_d2=scipy.stats.ks_2samp(d1, d2)[1]
        test={'K-S_test_p':KS_d1_d2,'DC_test':0, 'Criterion3':0,'P4_test':P4_test,'Case':'Case2'}
        Match_test='N'
        if (P4_test>=0.075) and (KS_d1_d2>=1E-90):
            Match_test='Y'
        return test,Match_test
     #case 3  - contours possibly circular and difference rwidth/rheigth is not significant
     # estimate KS(centroid), DC, and Criterion3
    Criterion3=Criterion3_test(Internal_cont_r_areas1,inter_c_DCREC1,A1,Internal_cont_r_areas2,
                    inter_c_DCREC2,A2,t1=15,t2=15,t_min_area=7)    
    equi_diameter1 = np.sqrt(4*A1/np.pi)
    equi_diameter2 = np.sqrt(4*A2/np.pi)
    DC1=A1/(equi_diameter1*radius1)
    DC2=A2/(equi_diameter2*radius2)
    DC_test=abs(1-min(DC1,DC2)/max(DC1,DC2))*100.0
    test={'K-S_test_p':scipy.stats.ks_2samp(d1, d2)[1],'DC_test':DC_test, 'Criterion3':Criterion3,'P4_test':0,'Case':'Case3'}
    
    Match_test= Matching_Criteria(
            test['K-S_test_p'],test['DC_test'],Criterion3, rc, noise_level=0.0)
    if Match_test=='N' :    
        return test,Match_test
    if Match_test=='Y':
        #case 4
        # add enc circles to images (gray with internal contours, image1 and image2)
        # apply Crriterion3; if Criterion3=='N' then Match_test='N'!
        cv2.circle(tmp_im_gray1,(int(Cx1),int(Cy1)),int(radius1*0.95),(255,255,255),1+int(0.02*radius1))
        cv2.circle(tmp_im_gray2,(int(Cx2),int(Cy2)),int(radius2*0.95),(255,255,255),1+int(0.02*radius2))
        
        contours, hierarchy = cv2.findContours(tmp_im_gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        A1=cv2.contourArea(contours[0])
        cont1=contours[0]
        for c in contours:
            if cv2.contourArea(c)>A1:
                A1=cv2.contourArea(c)
                cont1=c
        contours1,hierarchy1=cv2.findContours(tmp_im_gray1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        holes1 = [contours1[i] for i in range(len(contours1)) if hierarchy1[0][i][3] >= 0]
        Internal_cont_r_areas1=sorted([100.0*cv2.contourArea(x)/A1 for x in holes1 if 100.0*cv2.contourArea(x)/A1>=1.0], reverse=True)
        Internal_cont_centres1=sorted([[int(cv2.moments(x)["m10"] / cv2.moments(x)["m00"]),int(cv2.moments(x)["m01"] / cv2.moments(x)["m00"])] for x in holes1 if 100.0*cv2.contourArea(x)/A1>=1.0], reverse=True)
        inter_c_DCREC1=sorted([(100.0/radius1)*np.sqrt((Internal_cont_centres1[i1][0]-Internal_cont_centres1[i2][0])**2+(Internal_cont_centres1[i1][1]-Internal_cont_centres1[i2][1])**2)  for i1 in range(0,len(Internal_cont_centres1)-1) for i2 in range(i1,len(Internal_cont_centres1)) if i1!=i2], reverse=True)
         
        A2=cv2.contourArea(contours[0])
        cont2=contours[0]
        for c in contours:
            if cv2.contourArea(c)>A2:
                A2=cv2.contourArea(c)
                cont2=c
        contours2,hierarchy2=cv2.findContours(tmp_im_gray2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        holes2 = [contours2[i] for i in range(len(contours2)) if hierarchy2[0][i][3] >= 0]
        Internal_cont_r_areas2=sorted([100.0*cv2.contourArea(x)/A2 for x in holes2 if 100.0*cv2.contourArea(x)/A2>=1.0], reverse=True)
        Internal_cont_centres2=sorted([[int(cv2.moments(x)["m10"] / cv2.moments(x)["m00"]),int(cv2.moments(x)["m01"] / cv2.moments(x)["m00"])] for x in holes2 if 100.0*cv2.contourArea(x)/A2>=1.0], reverse=True)
        inter_c_DCREC2=sorted([(100.0/radius2)*np.sqrt((Internal_cont_centres2[i1][0]-Internal_cont_centres2[i2][0])**2+(Internal_cont_centres2[i1][1]-Internal_cont_centres2[i2][1])**2)  for i1 in range(0,len(Internal_cont_centres2)-1) for i2 in range(i1,len(Internal_cont_centres2)) if i1!=i2], reverse=True)
        Criterion3=Criterion3_test(Internal_cont_r_areas1,inter_c_DCREC1,A1,Internal_cont_r_areas2,
                                   inter_c_DCREC2,A2,t1=5,t2=5,t_min_area=5)
        
        test={'K-S_test_p':0,'DC_test':0, 'Criterion3':Criterion3,'P4_test':0,'Case':'Case4'}
        if Criterion3=='N':
            Match_test='N'    
    return test,Match_test



im1tmp= cv2.imread('X:/Templates/template1.jpg')
files_dir='X:/Images/'
jpg_files = [f for f in os.listdir(files_dir) if f.upper().endswith('.JPG')]
for f in jpg_files:
    print ('Processing of file:',f)
    im2tmp= cv2.imread(files_dir+f)
    Result=Test_Images(im1tmp,im2tmp)
    print 'Details',Result[0]
    print 'Matched:',Result[1]
    print '_______________________________________'
