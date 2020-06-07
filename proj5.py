import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from scipy.optimize import least_squares
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

# Read camera matrix from given model file
def getCameraMatrix(path):
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
	K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
	return K, LUT

# Undistort given camera frames and convert to grayscale
def undistortImageToGray(img,LUT):
	colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
	undistortedimage = UndistortImage(colorimage, LUT)
	gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
	# equ = cv2.equalizeHist(gray1)
	return gray

# Extract and match features
def features(img1, img2):
	MIN_MATCH_COUNT = 10

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()
	#orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_TREE = 0
	index_params = dict(algorithm = FLANN_INDEX_TREE, trees=5)#table_number = 12, key_size=20, multi_probe_level=2)
	search_params = dict(checks = 100)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for i in range(len(matches)):
		#print(matches)
		mat = matches[i]
		if len(mat)<2:
			continue
		#print(mat)
		m,n = mat
		if m.distance < 0.5*n.distance:
			good.append(m)

	pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
	pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

return pts1, pts2

# Calculate Pose from Essential Matrix
def ExtractCameraPose(E):
	poses = []
	W = np.array(([0,-1,0],[1,0,0],[0,0,1]))
	U,S,V = np.linalg.svd(E)
	C1 = -U[:,2].reshape(-1,1)
	C2 = U[:,2].reshape(-1,1)
	C3 = -U[:,2].reshape(-1,1)
	C4 =  U[:,2].reshape(-1,1)
	R1 = np.matmul(np.matmul(U,W),V)
	R2 = np.matmul(np.matmul(U,W),V)
	R3 = np.matmul(np.matmul(U,W.T),V)
	R4 = np.matmul(np.matmul(U,W.T),V)

	if np.linalg.det(R1)<0:
		R1=-R1
	P1 = np.concatenate((R1,C1),axis = 1)
	poses.append(P1)

	if np.linalg.det(R2)<0:
		R2=-R2
	P2 = np.concatenate((R2,C2),axis = 1)
	poses.append(P2)

	if np.linalg.det(R3)<0:
		R3=-R3
	P3 = np.concatenate((R3,C3),axis = 1)
	poses.append(P3)

	if np.linalg.det(R4)<0:
		R4=-R4
	P4 = np.concatenate((R4,C4),axis = 1)
	poses.append(P4)

	return poses

# Create skew symmetric matrix from 3D vector
def skew(v):
	a=v[0]
	b=v[1]
	c=v[2]
	return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

# Perform linear triangulation
def LinearTriangulation(K,P0,P1,pt1,pt2):
	pt1 = np.insert(np.float32(pt1),2,1)
	pt2 = np.insert(np.float32(pt2),2,1)
	homo_pt1 = np.matmul(np.linalg.inv(K),pt1.reshape((-1,1)))
	homo_pt2 = np.matmul(np.linalg.inv(K),pt2.reshape((-1,1)))

	skew0 = skew(homo_pt1.reshape((-1,)))
	skew1 = skew(homo_pt2.reshape((-1,)))

	P0 = np.concatenate((P0[:,:3], -np.matmul(P0[:,:3],P0[:,3].reshape(-1,1))),axis=1)
	P1 = np.concatenate((P1[:,:3], -np.matmul(P1[:,:3],P1[:,3].reshape(-1,1))),axis=1)

	pose1 = np.matmul(skew0,P0[:3,:])
	pose2 = np.matmul(skew1,P1[:3,:])

	#Solve the equation Ax=0
	A = np.concatenate((pose1,pose2),axis=0)
	u,s,vt = np.linalg.svd(A)
	X = vt[-1]
	X = X/X[3]
	return X

# Extract correct pose from set of poses based on correct feature point orientation
def DisambiguateCameraPose(P0,poses, allPts):
	max = 0
	flag = False
	for i in range(4):
		P = poses[i]
		# print("Each"+str(i),P)
		r3 = P[:,3]
		r3 = np.reshape(r3,(1,3))
		C = P[:,3]
		C = np.reshape(C,(3,1))
		pts_list = allPts[i]
		pts = np.array(pts_list)
		pts = pts[:,0:3].T

		diff = np.subtract(pts,C)
		Z = np.matmul(r3,diff)
		Z = Z>0
		_,idx = np.where(Z==True)
		if max < idx.shape[0]:
			poseid = i
			correctPose = P
			indices = idx
			max = idx.shape[0]
	if max==0:
		flag = True
		correctPose = None
	return correctPose,flag,poseid

# Calculate Essential Matrix
def getEssentialMat(K,pts1,pts2):
	F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, 1,0.90)
	u,s,vt = np.linalg.svd(F)
	s[2] = 0
	snew = np.diag(s)
	F = np.matmul(np.matmul(u,snew),vt)
	assert np.linalg.matrix_rank(F)==2,"Rank of F not 2"

	E = np.matmul(np.matmul(K.T,F),K)
	U,S,Vt = np.linalg.svd(E)
	S[0] = 1
	S[1] = 1
	S[2] = 0
	Snew = np.diag(S)
	E = np.matmul(U,np.matmul(Snew,Vt))
	return E

# Perform non-linear triangulation (helper function)
def nonLinearFunctionTriangulation(X,P1,P2,x1,x2,K):
	Xpt=np.reshape(X,(np.shape(X)[0],1))
	P1 = np.concatenate((P1[:,:3], -np.matmul(P1[:,:3],P1[:,3].reshape(-1,1))),axis=1)
	P2 = np.concatenate((P2[:,:3], -np.matmul(P2[:,:3],P2[:,3].reshape(-1,1))),axis=1)
	P1=np.matmul(K,P1)
	P2=np.matmul(K,P2)
	sumError = []
	err1 = (x1[0] - (np.dot(P1[0],Xpt)/np.dot(P1[2],Xpt)))**2 + (x1[1] - (np.dot(P1[1],Xpt)/np.dot(P1[2],Xpt)))**2
	err2 = (x2[0] - (np.dot(P2[0],Xpt)/np.dot(P2[2],Xpt)))**2 + (x2[1] - (np.dot(P2[1],Xpt)/np.dot(P2[2],Xpt)))**2

	err = err1+err2

	return err

# Perform non-linear triangulation
def NonlinearTriangulation(K,pose1,pose2,pts0,pts1,X):
	Xinit=np.reshape(X,(np.shape(X)[0],))
	filteredPoints=least_squares(nonLinearFunctionTriangulation,
					  x0=Xinit,
					 args=(pose1,pose2,pts0,pts1,K), ftol = 1e-2, xtol = 1e-2, gtol = 1e-2, max_nfev = 1000)
	return filteredPoints.x/filteredPoints.x[3]


def fun(P,X,x,K):
	P = np.reshape(P,(3,4))
	P = np.matmul(K,P)
	sum = 0
	for pt in range(len(X)):
		Xpt = np.reshape(X[pt],(4,1))
		err = (x[pt,0] - (np.dot(P[0],Xpt)/np.dot(P[2],Xpt)))**2 + (x[pt,1] - (np.dot(P[1],Xpt)/np.dot(P[2],Xpt)))**2
		sum = sum +err
	return sum

# Perfrom non-linear pnp to optimize obtained pose
def nonlinearPnP(X,x,K,Cnew,Rnew):
	Cnew = Cnew.reshape(-1,)
	t = np.array([[1,0,0,-Cnew[0]],
				  [0,1,0,-Cnew[1]],
				  [0,0,1,-Cnew[2]]])
	
	P = np.matmul(Rnew,t)
	P_init = np.reshape(P,(P.shape[0]*P.shape[1],))
	res = least_squares(fun, P_init, args = (X,x,K), ftol = 1e-2, xtol = 1e-2, gtol = 1e-2, max_nfev = 1000)
	P_refined = res.x
	cost = res.cost
	P_refined = np.reshape(P_refined, (3,4))
	
	R_opt = np.reshape(P_refined[:,0:3],(3,3))
	C_opt = np.reshape(P_refined[:,3],(3,1))

	return R_opt, C_opt

#########################################################################################

def main():
	BasePath = './Oxford_dataset/stereo/centre/'
	K, LUT = getCameraMatrix('./Oxford_dataset/model')
	images = []
	H1 = np.array([[1,0,0,0],
				   [0,1,0,0],
				   [0,0,1,0],
				   [0,0,0,1]])
	H1_calc = H1
	P0 = H1[:3]
	cam_pos = np.array([0,0,0])
	cam_pos = np.reshape(cam_pos,(1,3))
	test = os.listdir(BasePath)
	builtin = []
	points = []
	rot_calc = []
	trans_calc = []
	pnp = []
	for image in sorted(test):
	   images.append(image)

	for img,_ in enumerate(images[:-2]):
		img1 = cv2.imread("%s/%s"%(BasePath,images[img]),0)
		img2 = cv2.imread("%s/%s"%(BasePath,images[img+1]),0)
		und1 = undistortImageToGray(img1,LUT)
		und2 = undistortImageToGray(img2,LUT)

		# Get features
		pts1, pts2 = features(und1,und2)
		if pts1.shape[0] <= 5:
			continue

		# Calculate fundamental,essential and projection matrix
		# F,_ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
		# E,_ = cv2.findEssentialMat(pts1,pts2,focal =K[0][0], pp = (K[0,2],K[1,2]), method = cv2.RANSAC,prob=0.999,threshold=0.5)
		E_calc = getEssentialMat(K,pts1,pts2)
		poses = ExtractCameraPose(E_calc)
		allPts = dict()
	
		# perform triangulation to obtain 3d position of feature points
		for j in range(4):
			X = []
			for i in range(len(pts1)):
				# Use linear triangulation
				pt = LinearTriangulation(K,P0,poses[j],pts1[i],pts2[i])

				# Use Non-Linear triangulation
				pt = NonlinearTriangulation(K,P0,poses[j],pts1[i],pts2[i],pt)
			#    pt.x = pt.x/pt.x[3]
				X.append(pt)
			allPts.update({j:X})
		correctPose,no_inlier,poseid = DisambiguateCameraPose(P0, poses, allPts)
		R_calc = correctPose[:,:3].reshape(3,3)
		C_calc = correctPose[:,3].reshape(3,1)

		# Perfrom nonlinear pnp to improve pose
		R_calc, C_calc = nonlinearPnP(allPts[poseid],pts2,K,C_calc,R_calc)
		if np.linalg.det(R_calc)<0:
			R_calc = -R_calc
		H2_calc = np.hstack((R_calc,C_calc))
		H2_calc = np.vstack((H2_calc,[0,0,0,1]))
		H1_calc = np.matmul(H1_calc,H2_calc)
		xpt_calc = H1_calc[0,3]
		zpt_calc = H1_calc[2,3]
		rot_calc.append(R_calc)
		trans_calc.append(C_calc)
		points.append((xpt_calc,zpt_calc))
		pnp = [rot_calc,trans_calc,points]
		# save data
		if (img%1)==0:
			file2 = './results/pnp/iters'+str(img)+'.npy'
			np.save(file2,pnp)

		# Plot trajectory
		plt.plot(xpt_calc,zpt_calc,'.r')
		# plt.pause(0.01)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	main()
