Problem Statement
-----------------
Implementing an image stitcher that uses image warping and homographies to automatically
create an image mosaic. We have two input images that should form the mosaic, where we
warp one image into the plane of the second image and display the combined views

Getting correspondences
-----------------------
The provided code is used to get manually identified corresponding points from two views using ginput

![alt text](https://raw.githubusercontent.com/azmy92/Image-Mosaics/master/pics/corres.PNG "correspondance")


Computing the homography parameters
------------------------------------
We set up a solution using a system of linear equations Ax = B, where the 8 unknowns of H are stacked into an 8vector (x).

input points <--- user input

adjust A

adjust X

compute B using least square algorithm


Warping between image planes
-----------------------------

for each pixel in img1

--transformed pixel <Homography

--matrix * pixel Homogeneous dimensions

--convert pixel to cartesian coordinates

--result[pixel.x][pixel.y] <pixel intensity

--for each pixel in 8 neighbors of current pixel in result

---original coordinated <-- Homography inverse * neighbour

---neighbour < -- img1[original coordinates]

--end for

end for

repeat empty boundaries

Sample runs
-----------
run1:
------

![alt text](https://raw.githubusercontent.com/azmy92/Image-Mosaics/master/pics/res1.PNG "Logo Title Text 1")

run 2:
-------
![alt text](https://raw.githubusercontent.com/azmy92/Image-Mosaics/master/pics/res2a.PNG "Logo Title Text 1")

![alt text](https://raw.githubusercontent.com/azmy92/Image-Mosaics/master/pics/res2b.PNG "Logo Title Text 1")

