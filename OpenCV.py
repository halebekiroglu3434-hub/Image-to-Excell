import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('Photos/meow.jpg')


def rescaleFrame (img1, scale=0.5):
    widht = int(img1.shape[1] * scale)
    hight = int(img1.shape[0] * scale)
    dim = (widht, hight)

    return cv.resize(img1, dim, interpolation=cv.INTER_AREA)


#cv.imshow('meow', img1)
#cv.waitKey(0)

#capture = cv.VideoCapture('Photos/rlvid.mp4')
#while True:
    #isTrue, frame = capture.read()#eşitlik varken devam et

    #cv.imshow('rlvid', frame)
    #if cv.waitKey(15) & 0xFF == ord('d'):
        #break
#capture.release()
#cv.destroyALLWindows()
#cv.waitkey(0)

# 2. Boş bir resim oluşturuyoruz (img2 bu olacak)
# np.zeros: "İçi sıfırlarla dolu (yani siyah) bir alan yarat" demektir.
# (500,   500, 3): Yükseklik 500, Genişlik 500, 3 tane renk kanalı (B-G-R) olsun.
# dtype='uint8': Resim verisi olduğu için veri tipi standart (0-255 arası) olsun.
img2 = np.zeros((500, 500, 3), dtype='uint8')

cv.rectangle(img2, (50, 50), (450, 450), color=(130, 2, 250), thickness=-1)
    #   (X1, Y1), (X2, Y2)        #B - G - R     Thickness= Kalınlık
#         Start  End                              #-1 = Fill the Gap


cv.circle(img2, (250,250), 40 , color=(220, 30, 30), thickness=-1)
#        (Merkez_X, Merkez_Y), Yarıçap

cv.putText(img2, 'Image', (200,90), cv.FONT_HERSHEY_TRIPLEX, 1.0, (100,30,30), thickness=2)
#                                   font seçenekleri= cv.FONT_ yazınca çıkıyor
cv.line(img2, (0,0), (500,500), color=(5,5,5), thickness=5)

#cv.imshow('circle', img2)
#cv.waitKey(0)

convert_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#cv.imshow('photos', convert_gray)
# 📌convert color📌

#cv.waitKey(0)

hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

#cv.imshow('hsv photos', hsv)
#cv.waitKey(0)

blur1 = cv.GaussianBlur(img1, (25,5), cv.BORDER_DEFAULT)
#                              📌 (DERECESİ)
#📌   belli derece bulurlama 📌GÜRÜLTÜYÜ AZALTIR

#cv.imshow('photos', blur1)
#cv.waitKey(0)


edges = cv.Canny(img1, 50, 175)
#     📌min renk değişimi sertliği 📌net renk değişimi ne kadar sert
#cv.imshow('photos', edges)
#cv.waitKey(0)

kernel = np.ones((5,5), np.uint8)
#        📌5x5 lik beyaz bir kutu yapıp çizgilerin üstünden geçiriyoruz
dilation = cv.dilate(edges, kernel, iterations=2)
#   📌dilate = genişletmek        iteration== 📌mathematical repetition
#cv.imshow('photos', dilation)
#cv.waitKey(0)

eroded = cv.erode(edges, (4,4), iterations=1)
#cv.imshow('photos', eroded)
#cv.waitKey(0)

resizing = cv.resize(img1, (850,800), interpolation=cv.INTER_CUBIC)
#cv.imshow('photos', resizing)
#cv.waitKey(0)

crop = img1[50:750, 50:820]
#          height , width
#cv.imshow('photos', crop)
#cv.waitKey(0)

def transformation(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = transformation(img1, 100, 100)
translated_flip = cv.flip(translated, 1)
#cv.imshow('photos', translated_flip)
#cv.waitKey(0)



#📌cv.cntcolor(img, cv.COLORBGR_2GRAY9
canny_gray = cv.Canny(convert_gray, 125, 175)
#cv.imshow('photos', canny_gray)
#cv.waitKey(0)

kernel2 = np.ones((4,4), np.uint8)
img3 = cv.imread('photos/images3.jfif')
img3_resize = cv.resize(img3, (700,550), interpolation=cv.INTER_AREA)
img3_gray = cv.cvtColor(img3_resize, cv.COLOR_BGR2GRAY)
img3_gray_canny = cv.Canny(img3_gray, 145, 185)
img3_gray_canny_dilation = cv.dilate(img3_gray_canny, kernel2, iterations=1)
#cv.imshow('photos', img3_gray_canny_dilation)
#cv.waitKey(0)

# 📌cv.findContours(Kaynak_Resim, Mod, Yöntem)📌                      📌Kenarların kordinatlarını kaydeder
contours, hierarchies = cv.findContours(img3_gray_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#                                      📌RETR_LIST hepsi // RETR_EXTERNAL sadece dış
print(len(contours))


blank = np.zeros(img3_resize.shape, dtype='uint8')

blank1 = cv.drawContours(blank, contours, -1, (0,50,205), 1)
#cv.imshow('photos', blank1)#                 B- G-R     Thickness
#cv.waitKey(0)

img3_resize_blur = cv.blur(img3_resize, (3,3), cv.BORDER_DEFAULT)
img3_resize_blur_canny = cv.Canny(img3_resize_blur, 125, 175)
#cv.imshow('blur_canny', img3_resize_blur_canny)
#cv.waitKey(0)
#📌Biraz blur ekkleyip canny yaparsan gürültüden kurtulursun📌
img3_blur_contours = cv.findContours(img3_resize_blur_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(len(img3_blur_contours))
#📌-----------------------------------threshold
#📌gri resimde parlaklık seçmek daha kolaydır!

threshold, threshold_img = cv.threshold(img3_gray, 140,240, cv.THRESH_BINARY)
#                                             parlaklık sınırı(150 üstü 1/altı 0) Bınary= 0/1
cv.imshow('threshold', threshold_img)
#cv.waitKey(0)
#gri yapmazsan >

threshold, threshold_img2= cv.threshold(img3_resize, 150, 255, cv.THRESH_BINARY)
cv.imshow('thresh', threshold_img2)
#cv.waitKey(0)

#📌 --------------------------PLT
#📌 plt ile Görsel üzerindeki (X,Y) yi görürürüz
img4 = cv.imread('photos/images4.png')
cv.imshow('photos', img4)

plt.imshow(img4)
plt.show()

#--------------------Renklere ayırma/birleştirme
img4_resize = cv.resize(img4, (730,750), interpolation=cv.INTER_AREA)


img4_resize_BGR = cv.cvtColor(img4_resize, cv.COLOR_BGRA2BGR)
b,g,r = cv.split(img4_resize_BGR)


#cv.imshow('blue',b)
#cv.imshow('green', g)#yeşilin yoğun olduğu yer BEYAZ olur
#cv.imshow('red', r)

blank2 = np.zeros(b.shape, dtype='uint8')

merge_B_G_R = cv.merge([b, g, blank2])
red_img = cv.merge([blank2, blank2, r])

#cv.imshow('blue_green', merge_B_G_R)
#cv.imshow('red', red_img)


#                                                📌Kenarları yansıtarak yap
img4_resize_averageblur = cv.blur(img4_resize, (5,5), cv.BORDER_REFLECT)
img4_resize_gaussianblur = cv.GaussianBlur(img4_resize, (5,5), 0, cv.BORDER_REFLECT)
#                                                            📌deviation'ı kendin hesapla


#cv.imshow('average_blur', img4_resize_averageblur)
#cv.imshow('gaussian_blur', img4_resize_gaussianblur)

median_blur = cv.medianBlur(img4_resize, 5, cv.BORDER_REFLECT)
# #           📌 5 li karedeki sayıların 📌medyanını📌 , Merkze koyar
#cv.imshow('median_blur', median_blur)
#📌 Leke gidermede 1 numara

bilateral_blur = cv.bilateralFilter(img4_resize, 5, 75,75, cv.BORDER_REFLECT)
#                                                      📌
#cv.imshow('bilateral_blur', bilateral_blur)

#---------------Bitwise

blank3 = np.zeros((400,400), dtype='uint8')

rectange2 = cv.rectangle(blank3.copy(), (30,30), (370,370), 255, -1)
circle2 = cv.circle(blank3.copy(), (200,200), 200, 255, -1)

#📌 Bitwise_and  == intersection
bitwise_and1 = cv.bitwise_and(rectange2, circle2)
#📌 Bitwise_or == non intersection + intersection
bitwise_or1 = cv.bitwise_or(rectange2, circle2)
#📌 Bitwise_xor == non intersection
bitwise_xor1 = cv.bitwise_xor(rectange2, circle2)
#📌 Bitwise_not == color 1->0 , 0->1 converter
bitwise_not1 = cv.bitwise_not(bitwise_xor1)


#cv.imshow('bitwise_xor', bitwise_xor1)
#cv.imshow('bitwise_or', bitwise_or1)
#cv.imshow('bitwise_and', bitwise_and1)
#cv.imshow('bitwise_not', bitwise_not1)

#--------------Masking
img5 = cv.imread('photos/image5.jpg')

blank4 = np.zeros(img5.shape[:2], dtype='uint8')

#yazı okumada rectangle mask çok çok iş yapacak
mask = cv.circle(blank4, (img5.shape[1]//2, img5.shape[0]//2), 306, 255, -1)
#b1, g1, r1 = cv.split(img5)
#merge_bg = cv.merge([b1, g1, blank4])

masked_img = cv.bitwise_and(img5, img5, mask=mask)
cv.imshow('masked', masked_img)

#|-----------------------------------Renk  histogramı

graylandscape = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)

gray_hist = cv.calcHist([graylandscape], [0], None, [256], [0, 256])

plt.figure()
plt.title('Gray hist')
plt.xlabel('Pixels')
plt.ylabel('Values')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

#-colored version
blank5 = np.zeros(img5.shape[:2], dtype='uint8')
mask2 = cv.rectangle(blank5, (30,30), (250,300), 255, -1)
masked2 = cv.bitwise_and(img5, img5, mask=mask2)


colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist_colorful = cv.calcHist([img5], [i], mask2, [256], [0,256])
    plt.plot(hist_colorful, color=col)
    plt.xlim([0,256])

plt.show()

#-----------------Simple Threshold

threshold2 = cv.threshold(graylandscape, 150 ,255, cv.THRESH_BINARY)

threshold2_inverse = cv.threshold(graylandscape, 150, 255, cv.THRESH_BINARY_INV)
#----Advanced Thresholding

adaptive_threshold = cv.adaptiveThreshold(graylandscape,
                                          255,
                                          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY,
                                          11,3)#blocksize:kaç blockluk yerin ortalamasına göre olsun
#                                C:bulduğu ortalama değer-C == thresh değeri

cv.imshow('adaptive_threshold', adaptive_threshold)

#-------------------Egde Detection
#-Laplacian
laplacian = cv.Laplacian(graylandscape, cv.CV_64F)#o yüzden 64float yapıyoruz
#UİNT8 sadece pozitiflerden oluştuğu için negatif geçişleri kaybederiz

#görüntğ formatına geri çeviriyoruz
lap = np.uint8(np.absolute(laplacian))
cv.imshow('laplacian1', laplacian)


#-Sobel
sobelx = cv.Sobel(graylandscape, cv.CV_64F, 1, 0)
sobely = cv.Sobel(graylandscape, cv.CV_64F, 0, 1)

sobelx_gorselformat = np.uint8(np.absolute(sobelx))
sobely_gorselformat = np.uint8(np.absolute(sobely))
sobelxy = cv.bitwise_or(sobelx, sobely)

cv.imshow('sobelxy', sobelxy)
#cv.imshow('sobelx', sobelx)
#cv.imshow('sobely', sobely)

#-----------------------Visual Alignment
img6 = cv.imread('photos/yoklama.jpg')
width, height = 250, 350

pts1 = np.float32([[13,13], [383, 18], [380, 285], [18, 287]])

pts2 = np.float32([[0,0], [width,0], [width, height], [0, height]])

matrixcalc = cv.getPerspectiveTransform(pts1, pts2)
output = cv.warpPerspective(img6, matrixcalc, (width, height))
cv.imshow('output', output)
cv.imshow('img6', img6)
#13, 13,383 18,380 285,18 287


cv.waitKey(0)
cv.destroyAllWindows()


