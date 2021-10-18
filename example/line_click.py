import numpy as np
import cv2

#H, W = img.shape[:2]
#print({"H":H,"W":W})

old_ij = (0,0)
totyu = 0

#test_map = np.array([[255]*10]*10)
test_map = cv2.imread('test.png')
print(type(test_map))

def onMouse(event, j, i, flags, params):
    global old_ij,totyu,test_map
    if event == cv2.EVENT_LBUTTONDOWN:
        print({"i":i,"j":j})
        if totyu == 0:
            old_ij = (i,j)
            totyu = 1
            test_map[i][j] = 0
        else:
            totyu = 0
            line_array = calc_points(old_ij[0],old_ij[1],i,j)
            test_map[i][j] = 0
            for ij in line_array:
                test_map[ij[0]][ij[1]] = 0
            for a in line_array:
                print(a)
    cv2.imshow('test',test_map.astype(np.float32))

def calc_points(i1, j1, i2, j2):
    '''ピクセル (i1, j1) と ピクセル (i2, j2) が通るピクセルの一覧を返す
    '''
    if i1 == i2:
        points_j = np.arange(j1, j2+1)
        points_i = np.full(len(points_j),i1)
    elif j1 == j2:
        points_i = np.arange(i1, i2+1)
        points_j = np.full(len(points_i),j1)

    else:
        di = i2 - i1
        dj = j2 - j1
        adi = abs(di)
        adj = abs(dj)

        max_d = adi if adi>adj else adj

        step_i = di/(max_d)
        step_j = dj/(max_d)

        points_i = np.round(np.arange(i1, i2+step_i, step_i))
        points_j = np.round(np.arange(j1, j2+step_j, step_j))
        
        if len(points_i) > len(points_j):
            points_i = points_i[1:]
        elif len(points_i) < len(points_j):
            points_j = points_j[1:]
            
    points = np.column_stack([points_i,points_j])
    return np.array(points).astype(int)
"""線太くうまくいかない
def calc_points(i1, j1, i2, j2):
    '''ピクセル (i1, j1) と ピクセル (i2, j2) が通るピクセルの一覧を返す
    '''
    if i1 == i2:
        points_j = np.arange(j1, j2+1)
        points_i = np.full(len(points_j),i1)
        points = np.column_stack([points_i,points_j])
    elif j1 == j2:
        points_i = np.arange(i1, i2+1)
        points_j = np.full(len(points_i),j1)
        points = np.column_stack([points_i,points_j])

    else:
        di = i2 - i1
        dj = j2 - j1
        adi = abs(di)
        adj = abs(dj)

        max_d = adi if adi>adj else adj

        step_i = di/(max_d*2)
        step_j = dj/(max_d*2)

        points_i = np.round(np.arange(i1, i2+step_i, step_i))
        points_j = np.round(np.arange(j1, j2+step_j, step_j))
        
        if len(points_i) > len(points_j):
            points_i = points_i[1:]
        elif len(points_i) < len(points_j):
            points_j = points_j[1:]
            
        points = np.column_stack([points_i,points_j])
        points = np.unique(points,axis=0)
        if points[0][0] != i1:
            points = np.flipud(points)
            #points = points[::-1]

    return np.array(points).astype(int)
"""
"""

def calc_points(i1, j1, i2, j2):
    '''ピクセル (i1, j1) と ピクセル (i2, j2) が通るピクセルの一覧を返す
    '''
    if i1 == i2:
        points_j = np.arange(j1, j2+1)
        points_i = np.full(len(points_j),i1)
        points = np.column_stack([points_i,points_j])
    elif j1 == j2:
        points_i = np.arange(i1, i2+1)
        points_j = np.full(len(points_i),j1)
        points = np.column_stack([points_i,points_j])

    else:
        di = i2 - i1
        dj = j2 - j1
        adi = abs(di)
        adj = abs(dj)

        max_d = adi if adi>adj else adj

        step_i = di/(max_d*2)
        step_j = dj/(max_d*2)

        points_i = np.round(np.arange(i1, i2+step_i, step_i))
        points_j = np.round(np.arange(j1, j2+step_j, step_j))
        
        if len(points_i) > len(points_j):
            points_i = points_i[1:]
        elif len(points_i) < len(points_j):
            points_j = points_j[1:]
            
        points = np.column_stack([points_i,points_j])
        points = np.unique(points,axis=0)
        if points[0][0] != i1:
            points = np.flipud(points)
            #points = points[::-1]

    return np.array(points).astype(int)
def calc_points(i1, j1, i2, j2):
    '''ピクセル (i1, j1) と ピクセル (i2, j2) が通るピクセルの一覧を返す
    '''
    if i1 == i2:
        points_j = np.arange(j1, j2+1)
        points_i = np.full(len(points_j),i1)
        points = np.column_stack([points_i,points_j])
    elif j1 == j2:
        points_i = np.arange(i1, i2+1)
        points_j = np.full(len(points_i),j1)
        points = np.column_stack([points_i,points_j])

    else:
        di = i2 - i1
        dj = j2 - j1
        adi = abs(di)
        adj = abs(dj)

        max_d = adi if adi>adj else adj

        step_i = di/(max_d*2)
        step_j = dj/(max_d*2)

        points_i = np.round(np.arange(i1, i2+step_i, step_i))
        points_j = np.round(np.arange(j1, j2+step_j, step_j))
        
        if len(points_i) > len(points_j):
            points_i = points_i[1:]
        elif len(points_i) < len(points_j):
            points_j = points_j[1:]
            
        points = np.column_stack([points_i,points_j])
        points = np.unique(points,axis=0)
        if points[0][0] != i1:
            points = np.flipud(points)
            #points = points[::-1]

    return np.array(points).astype(int)

def calc_points(x1, y1, x2, y2):
    points = []

    if x1 == x2:
        for i in range(y1,y2+np.sign(y2-y1),np.sign(y2-y1)):
            points.append([x1,i])
        return np.array(points).astype(int)
    if y1 == y2:
        for i in range(x1,x2+np.sign(x2-x1),np.sign(x2-x1)):
            points.append([i,y1])
        return np.array(points).astype(int)


    # ピクセル座標を実2次元座標に変換する。
    # 例: ピクセル (1, 1) から (3, 2) へ線を引く場合、実2次元座標で点 (1.5, 1.5) から
    #     (3.5, 2.5) を結ぶ線とする。
    x1, y1 = x1 + 0.5, y1 + 0.5  # ピクセルの中心
    x2, y2 = x2 + 0.5, y2 + 0.5  # ピクセルの中心

    # 初期化
    x, y = x1, y1
    cell_w, cell_h = 1, 1  # セルの大きさ、今回はピクセルなので (1, 1)
    step_x = np.sign(x2 - x1)  # (x1, y1) から (x2, y2) へ進むときの x 方向のステップ数
    step_y = np.sign(y2 - y1)  # (x1, y1) から (x2, y2) へ進むときの y 方向のステップ数
    delta_x = cell_w / abs(x2 - x1)
    delta_y = cell_h / abs(y2 - y1)
    # a / b % 1 は a / b の計算値の小数部分 (例: 3 / 2 % 1 = 1.5 % 1 = 0.5)
    max_x = delta_x * (x1 / cell_w % 1)
    max_y = delta_y * (y1 / cell_h % 1)

    points.append([x, y])  # 開始点
    reached_x, reached_y = False, False  # 到達判定用フラグ
    while not (reached_x and reached_y):
        if max_x < max_y:
            max_x += delta_x
            x += step_x
        else:
            max_y += delta_y
            y += step_y

        points.append([x, y])  # 点を追加

        # 終点に到達したかどうか
        if (step_x > 0 and x >= x2) or (step_x <= 0 and x <= x2):
            reached_x = True
        if (step_y > 0 and y >= y2) or (step_y <= 0 and y <= y2):
            reached_y = True

    return np.array(points).astype(int)
"""

cv2.imshow('test',test_map.astype(np.float32))
cv2.setMouseCallback('test', onMouse)
cv2.waitKey(0)

