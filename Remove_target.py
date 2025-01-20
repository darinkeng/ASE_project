
def remove_black(data):
	a=0
	for i in data:
  		img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
  		x,y,h,w=210,30,360,360
  		crop_img=img[x:x+h, y:y+w]
  		threshold = 25000
  		for j in range(crop_img.shape[0]):
    		for k in range(crop_img.shape[1]):
      			diff=(crop_img[j][k]-255)**2
      			if diff > threshold:
          			crop_img[j][k] = 255

  cv2.imwrite(f"/content/drive/MyDrive/sun_moon_crop_img/A158200_normal/normal{a}.jpg",crop_img)
  a+=1