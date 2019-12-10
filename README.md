# FindHomography
This repository shows how to solve homography by given four corresponding points(c++)

# Solve Process
First, we set four src points and corresponding dst points.  
Then, we construct 8 homography equations.  
Finally, solve these equations by gaussian elimination.   
Compared with OpenCV findHomography function, the results are similar. 

# How to Run
`cd mac_os_bin`  
`sh build.sh`  
Then, you will get the homography calculated and the result by OpenCV.
