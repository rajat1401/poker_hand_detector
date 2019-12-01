r=['2','3','4','5','6','7','8','9','10','A','Q']
s=['Clubs','Diamond','Hearts','Spades']

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')
for i in range(0,len(r)):
    img=cv2.imread(std_r+str(r[i])+'/1.png', 0)
    img= cv2.resize(img,(20,35)) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)   
    j = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=std_r+str(r[i])+'/', save_format='png'):
        j += 1
        print (i)
        if j >= 100:
            break

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')
for i in range(0,len(s)):

    img=cv2.imread(std_s+str(s[i])+'/1.png', 0)
    print(img.shape)
    # x = img_to_array(img)
    
    # x = x.reshape((1,) + x.shape) 
    # print("ERD")
    # print(x.shape)
    img= cv2.resize(img,(20,35)) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) 


    j = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=std_s+str(s[i])+'/', save_format='png'):
        j += 1
        print (j)
        if j >= 100:
            break