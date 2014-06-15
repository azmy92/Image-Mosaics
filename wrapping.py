__author__ = 'azmy'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import matplotlib.image as mpimg
import itertools
import sys



#fileA = raw_input("Please insert the file name for the first image: ")
#fileB = raw_input("Please insert the file name for the second image: ")

fileA = "imgs/imageA.jpg"
fileB = "imgs/imageB.jpg"
#imageA = cv2.imread("imgs/imageA.jpg")
#imageB = cv2.imread("imgs/imageB.jpg")


def findKeyPoints(img, template, distance=200):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final

def drawKeyPoints(img, template, skp, tkp, num=-1):
    pts = []
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        pts.append(pt_a)
        pts.append(pt_b)
        print pt_a
        print pt_b
    return pts


def match(imageA,imageB):
    img = imageA
    temp = imageB

    dist = 200
    num = 4

    skp, tkp = findKeyPoints(img, temp, dist)
    return drawKeyPoints(img, temp, skp, tkp, num)






imageA = mpimg.imread(fileA)
imageB = mpimg.imread(fileB)
numPoints = 8
mode = raw_input("for sift enter sift: ")
pts = []
if (mode == "sift"):
    pts = match(imageA,imageB)

else:
    fig = plt.figure()
    figA = fig.add_subplot(1,2,1)
    figB = fig.add_subplot(1,2,2)
    # Display the image
    figB.imshow(imageB,origin='upper')
    figA.imshow(imageA,origin='upper')
    plt.axis('image')
    pts = plt.ginput(numPoints,timeout=0)

pts = np.reshape(pts, (numPoints/2,4))
xy = pts[:,[2,3]]

A=np.zeros((numPoints,8),'float64')




for i in range(numPoints/2):
   A[2*i][0]=pts[i][0]
   A[2*i][1]=pts[i][1]
   A[2*i][2]=1
   A[2*i][6]=-pts[i][0]*pts[i][2]
   A[2*i][7]=-pts[i][1]*pts[i][2]
   A[2*i+1][3]=pts[i][0]
   A[2*i+1][4]=pts[i][1]
   A[2*i+1][5]=1
   A[2*i+1][6]=-pts[i][0]*pts[i][3]
   A[2*i+1][7]=-pts[i][1]*pts[i][3]

Y=np.reshape(xy,(numPoints,1))

a,b,c,d,e,f,g,h = np.linalg.lstsq(A, Y)[0]

H=[[a,b,c],[d,e,f],[g,h,1]]

#verifining the H matrix
fig = plt.figure()
figB = fig.add_subplot(1,2,2)
figA = fig.add_subplot(1,2,1)
figB.imshow(imageB,origin='upper')
figA.imshow(imageA,origin='upper')
plt.axis('image')
i = 0
while (i < (numPoints/2)):
    pts = plt.ginput(1,timeout=0)
    pts = np.reshape(pts,(1*2,1))
    toTrans = np.ones((3,1))
    toTrans[0][0] = pts[0]
    toTrans[1][0] = pts[1]
    p = np.dot(H,toTrans)
    x = p[0][0]/p[2][0]
    y = p[1][0]/p[2][0]
    figA.scatter([x],[y])
    i = i + 1



img2 = cv2.imread(fileA)
img1 = cv2.imread(fileB)
mv1 = []
mv2 = []
rows = img1.shape[0]
cols = img2.shape[1] + img2.shape[1]
results = [np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8)]
mv1 = cv2.split(img1,mv1)
mv2 = cv2.split(img2,mv2)
for ii in range(0,3):
    img1 =mv1[ii]
    img2 = mv2[ii]



    #H = [[5.93314812e-01,6.41782199e-02,2.64710654e+02], [ -2.70368012e-01,7.92183580e-01,5.89275035e+01], [ -7.01814541e-04,-2.20932155e-04,1.00000000e+00]]
    #H = [[0.585512,0.128589,259.396],[-0.285436,0.86002,52.1516],[-0.000748086,-0.000101119,1]]
    #H2 = [[1.5995,-0.28828,-399.5444],[0.455588717,1.438633,-192.9034],[0.00124,math.pow(10,-5)*-7.51721074934485,1]]
    Hinv = np.linalg.inv(H)
    pixel = np.ones((3,1))
    transPix = np.zeros((3,1),np.float64)

#print pixel
    for i in range(0,img1.shape[0]):    #loop on y
        for j in range(0,img1.shape[1]):    #loop on x
            pixel[0][0] = j
            pixel[1][0] = i
            pixel[2][0] = 1

            transPix = np.dot(H,pixel)
            x = transPix[0][0] / transPix[2][0]
            y = transPix[1][0] / transPix[2][0]
            l = math.floor(x)
            k = math.floor(y)


            if(k< results[ii].shape[0]and l < results[ii].shape[1] and k >=0 and l>=0):
                results[ii][k][l] = img1[i][j]
                #fill holes using inverse wrapping
                invWrap = np.zeros((3,1),np.float64)
                uprow = np.int(k-1)
                leftcol = np.int(l-1)
                downrow = np.int(k+1)
                rightcol = np.int(l+1)
                for r in range(uprow,downrow):
                    for c in range(leftcol,rightcol):
                        if (r == k and c == l):
                            continue
                        if(r>0 and r <results[ii].shape[0] and c > 0 and c < results[ii].shape[1]):
                            invWrap[0][0] = c
                            invWrap[1][0] = r
                            invWrap[2][0] = 1
                            invWrap = np.dot(Hinv,invWrap)
                            x = invWrap[0][0] / invWrap[2][0]
                            y = invWrap[1][0] / invWrap[2][0]
                            if(x < img1.shape[1] and y < img1.shape[0]):
                                results[ii][r][c] = img1[y][x]







    for i in range(0,img2.shape[0]):
        for j in range(0,img2.shape[1]):
            results[ii][i][j] = img2[i][j]
    print "channel done"

    for i in range(0,results[ii].shape[0]):
        for j in range(0,results[ii].shape[1]):
            if(results[ii][i][j]==0):
                jj = j
                while(jj<results[ii].shape[1] and results[ii][i][jj]==0):
                    results[ii][i][jj] = results[ii][i][jj-1]
                    jj = jj +1
                j=jj



res = cv2.merge(results)
cv2.imshow("window",res)
cv2.waitKey(0)



