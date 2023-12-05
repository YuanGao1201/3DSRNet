import PIL.Image as Image
import numpy as np
import cv2


def FillHole(im_in):
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    return im_out


def img_split(img_path, img_new_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    img_array[-70:][:] = 0
    img_array[:70][:] = 0
    img_new = Image.fromarray(img_array)
    img_new.save(img_new_path)
    print('done')


def img_process(img_path, img_new_path):
    th = cv2.imread(img_path, 0)
    # contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for i, value in enumerate(contours):
    #     if cv2.contourArea(value) < 10:
    #         cv2.drawContours(th, contours, i, 0, thickness=-1)

    # _, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
    # for stat in stats:
    #     if stat[4] > 0 and stat[4] < 10:1
    #         cv2.rectangle(th, tuple(stat[0:2]), tuple(stat[0:2] + stat[2:4]), 128, thickness=-1)

    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dil = cv2.dilate(th, kernel_dilate)
    # fill = FillHole(dil)
    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # ero = cv2.erode(fill, kernel_erode)

    dst = cv2.fastNlMeansDenoising(th, None, 5, 7, 21)

    cv2.imwrite(img_new_path, dst)


if __name__ == '__main__':
    img_process('/media/gy/Data/VerSe/drr/real001/00xray1.png', '/media/gy/Data/VerSe/drr/real001/xray1.png')
    img_split('/media/gy/Data/VerSe/drr/real001/xray1.png', '/media/gy/Data/VerSe/drr/real001/xray1.png')
    img_process('/media/gy/Data/VerSe/drr/real001/00xray2.png', '/media/gy/Data/VerSe/drr/real001/xray2.png')
    img_split('/media/gy/Data/VerSe/drr/real001/xray2.png', '/media/gy/Data/VerSe/drr/real001/xray2.png')



