import json
import glob
import pickle
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from report import make_report
from particle_track import Particle_track
from excel_report import save_tracks_to_excel
from connected_component import Connected_component


RECTIFY_IMAGES = False
        

def rectifyImages(img1, img2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img1.shape[::-1], R, T, alpha=0, flags=0)

    mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img1.shape[::-1], cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img1.shape[::-1], cv2.CV_32F)

    #width = max(roi1[2], roi2[2])
    #height = max(roi1[3], roi2[3])
    
    img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)#[roi1[1]:roi1[1]+height, roi1[0]:roi1[0]+width]
    img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)#[roi2[1]:roi2[1]+height, roi2[0]:roi2[0]+width]

    return img_rect1, img_rect2, P1, P2


def drawRectifiedImages(img1, img2):
    
    img = np.hstack((img1, img2))

    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
            
    cv2.namedWindow('imgRectified', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgRectified', 1000, 600)
    cv2.imshow('imgRectified', img)            
    
    cv2.waitKey()

    return


def draw_tracks(tracks):

    if len(tracks) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mins = []
        maxs = []

        for track in tracks:
            points = track.flatten_3d_points
        
            ax.scatter(points[:,0,0], points[:,0,1], points[:,0,2], marker='o')
            mins.append(min(points[:,0,2]))
            maxs.append(max(points[:,0,2]))
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ax.set_zlim(min(mins) - 2, max(maxs) + 2)

        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        ax.set_zlabel('Z, mm')

        ax.view_init(-70, -90)
            
        plt.show()

    
def preprocess_image(img: np.ndarray, threshold: int, do_morph: bool=False, img_to_sub: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Производит предобработку изображений: вычитание фонового изображения (если задано),
    бинаризацию и морфологическую обработку (если задано)
    '''
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if img_to_sub is not None:
        res = cv2.subtract(gray, img_to_sub)
    else:
        res = gray
            
    retval, img_binary = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)

    if do_morph:
        kernel = np.ones((3,3),np.uint8)
        res = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    return res, img_binary


def get_connected_components(img: np.ndarray, binary_img: np.ndarray) -> List[Connected_component]:
    '''
    Находит связанные компоненты на изображении
    '''
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    cols = np.arange(labels.size)
    labels_sparse = csr_matrix((cols, (labels.ravel(), cols)), shape=(labels.max() + 1, labels.size))

    con_components = Connected_component.filter_components(nb_components, stats, img, binary_img, labels, labels_sparse, swt=False)

    return con_components


def find_corresponding_component(
        con_components1: List[Connected_component], 
        con_components2: List[Connected_component], 
        current_id: int, 
        preproc_img1: np.ndarray, 
        bianry_img1: np.ndarray, 
        search_area_range) -> Tuple[Connected_component, Connected_component]:
    '''
    Находит соответсвующие связанные компоненты на двух изображениях
    '''

    cur_component = None
    matched_component = None

    if len(con_components2) > 0 and current_id < len(con_components2):
        cur_component = con_components2[current_id]

        contour_zoomed = cur_component.raw_img
            
        search_area_left, search_area_right = cur_component.left - search_area_range, cur_component.right + search_area_range
        if search_area_left < 0:
            search_area_left = 0
        if search_area_right > preproc_img1.shape[1]:
            search_area_right = preproc_img1.shape[1]
        search_area_top, search_area_bottom = cur_component.top - 50, cur_component.bottom + 50
        if search_area_top < 0:
            search_area_top = 0
        if search_area_bottom > preproc_img1.shape[0]:
            search_area_bottom = preproc_img1.shape[0]

        search_dst1 = preproc_img1[search_area_top:search_area_bottom, search_area_left:search_area_right]

        try:
            res = cv2.matchTemplate(search_dst1, contour_zoomed, cv2.TM_CCOEFF_NORMED)
        except:
            pass
        finally:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.25:               
                matched_dst = bianry_img1[max_loc[1] + search_area_top:max_loc[1] + search_area_top + cur_component.height,
                                          max_loc[0] + search_area_left:max_loc[0] + search_area_left + cur_component.width]
                components_num, labels, stats, _ = cv2.connectedComponentsWithStats(matched_dst, connectivity=8)

                if components_num > 1:
                    matched_component_id = stats[1:,4].argmax() + 1
                    matched_component_points = [(x + max_loc[0] + search_area_left, y + max_loc[1] + search_area_top) for y, x in zip(np.where(labels == matched_component_id)[0], np.where(labels == matched_component_id)[1])]

                    for component in con_components1:
                        if component.point_belongs_num(matched_component_points):
                            matched_component = component
                            break

                if matched_component is not None:
                    matched_component.correlation_max = max_val
                    cur_component.correlation_max = max_val
                    matched_component.shift = max_loc
                    cur_component.shift = max_loc
    
    return cur_component, matched_component


def display_processing_results(
        img1: np.ndarray,
        img2: np.ndarray,
        con_components1: List[Connected_component],
        con_components2: List[Connected_component],
        cur_component: Connected_component,
        matched_component: Connected_component,
        filename1: str, 
        filename2: str,
        draw_graphics: bool, wait_time: int=0
    ) -> int:
    '''
    Отображает результаты обработки
    '''
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 700, 600)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 700, 600)
    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img3', 300, 300)
    cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img4', 300, 300)

    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if draw_graphics:
        for component in con_components1:
            img1 = component.draw_component(img1)
            point = (component.left, component.top)
            cv2.putText(img1, f'id={component.index}', point, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)

        for component in con_components2:
            img2 = component.draw_component(img2)
            point = (component.left, component.top)
            cv2.putText(img2, f'id={component.index}', point, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)

        if cur_component is not None:
            cv2.rectangle(img2, (cur_component.left, cur_component.top),
                                (cur_component.right, cur_component.bottom), (0,255,0), 3)
            cv2.imshow('img4', cur_component.visualized_img)
        else:
            cv2.imshow('img4', np.zeros((1,1), dtype=np.byte))
        if matched_component is not None:
            cv2.rectangle(img1, (matched_component.left, matched_component.top),
                                    (matched_component.right, matched_component.bottom), (0,255,0), 3)
            cv2.imshow('img3', matched_component.visualized_img)
        else:
            cv2.imshow('img3', np.zeros((1,1), dtype=np.byte))
    
    cv2.putText(img1, f'file={filename1}', (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    cv2.putText(img2, f'file={filename2}', (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)  
    
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)        

    key = cv2.waitKey(wait_time)
    
    return key


def main_processing_loop():
    
    with open('experiments.json', 'r', encoding='utf8') as f:
        experiments = json.load(f)

    experiment = experiments[EXPERIMENT_NUMBER]

    camera1_path = experiment['camera1_path']
    camera2_path = experiment['camera2_path']

    images_for_camera1 = sorted(glob.glob(camera1_path + experiment['image_file_mask']))

    assert len(images_for_camera1) > 0, f'Image files for processing not found!\nPath "{camera1_path}" is set in experiments.json file.'

    image_to_substruct1 = cv2.imread(images_for_camera1[experiment['background_image_index']])
    img_to_sub1 = cv2.cvtColor(image_to_substruct1, cv2.COLOR_BGR2GRAY)

    images_for_camera2 = sorted(glob.glob(camera2_path + experiment['image_file_mask']))
    image_to_substruct2 = cv2.imread(images_for_camera2[experiment['background_image_index']])
    img_to_sub2 = cv2.cvtColor(image_to_substruct2, cv2.COLOR_BGR2GRAY)

    try:        
        path_to_calibration_file = experiment['path_to_calibration_data']
        
        with open(path_to_calibration_file, 'r') as fp:
            calibration_data = json.load(fp)
            cameraMatrix1 = np.array(calibration_data['Camera1']['CameraMatrix'])
            distCoeffs1 = np.array(calibration_data['Camera1']['DistorsionCoefficients'])
            cameraMatrix2 = np.array(calibration_data['Camera2']['CameraMatrix'])
            distCoeffs2 = np.array(calibration_data['Camera2']['DistorsionCoefficients'])
            R = np.array(calibration_data['R'])
            T = np.array(calibration_data['T'])
        print('Calibration data is load from calibration_data.json') 
    except:
        print(f'Calibration data is not founded by path "{path_to_calibration_file}"') 

    tracks = []

    do_processing = True
    current_position = 0
    position_changed = True
    current_id = 0
    threshold= [5, 5]
    draw_graphics = True
    do_morph = False
    search_area_range = 600

    threshold[0] = experiment.get('initial_threshold', 5)
    threshold[1] = experiment.get('initial_threshold', 5)
    search_area_range = experiment.get('search_area_range', 500)

    while True:
        if position_changed:
            
            if current_position == len(images_for_camera1) - 1:
                current_position = 0
            fname1, fname2 = images_for_camera1[current_position], images_for_camera2[current_position]

            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)

            if RECTIFY_IMAGES:
                img1, img2, P1, P2 = rectifyImages(img1, img2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T)
            else:
                P1 = cameraMatrix1 @ np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 0, 0]]).T))
                P2 = cameraMatrix2 @ np.hstack((R, np.array([T]).T))

            preproc_img1, binary_img1 = preprocess_image(img1, threshold[0], do_morph=False, img_to_sub=img_to_sub1)
            preproc_img2, binary_img2 = preprocess_image(img2, threshold[1], do_morph=False, img_to_sub=img_to_sub2)

            con_components1 = get_connected_components(preproc_img1, binary_img1)
            con_components2 = get_connected_components(preproc_img2, binary_img2)
            
            position_changed = False
        
        cur_component, matched_component = find_corresponding_component(con_components1, con_components2, current_id, preproc_img1, binary_img1, search_area_range)

        key = display_processing_results(
            binary_img1,
            binary_img2,
            con_components1,
            con_components2,
            cur_component, 
            matched_component,
            fname1.split("\\")[-1],
            fname2.split("\\")[-1],
            draw_graphics,
            wait_time=0
        )

        if key==27:    # Esc key to stop
            break
        elif key==-1:  # normally -1 returned,so don't print it
            continue

        elif key==44:
            current_position -= 1
            current_id = 0
            position_changed = True
        elif key==46:
            current_position += 1
            current_id = 0
            position_changed = True

        elif key==47:
            if current_id > 0:
                current_id -= 1
        elif key==39:
            if current_id < len(con_components2) - 1:
                current_id += 1
        elif key == 49: # 1
            if threshold[0] > 0:
                threshold[0] -= 1
                print(f'Threshold[0] = {threshold[0]}')
                position_changed = True
        elif key == 50: # 2
            if threshold[0] < 255:
                threshold[0] += 1
                print(f'Threshold[0] = {threshold[0]}')
                position_changed = True
        elif key == 51: # 3
            if threshold[1] > 0:
                threshold[1] -= 1
                print(f'Threshold[1] = {threshold[1]}')
                position_changed = True
        elif key == 52: # 4
            if threshold[1] < 255:
                threshold[1] += 1
                print(f'Threshold[1] = {threshold[1]}')
                position_changed = True
        elif key == 100: # d
            draw_graphics = not draw_graphics
            position_changed = True
        elif key == 109: # m
            do_morph = not do_morph
            position_changed = True
        elif key==32:   # Space
            tracks.append(Particle_track(matched_component, cur_component, P1, P2))
            tracks[-1].particle_radius = experiment['particles_radius']
            tracks[-1].particle_density = experiment['particles_density']
            tracks[-1].voltage = experiment['voltage']
            tracks[-1].grid_height = experiment['grid_height']
            tracks[-1].image_name = fname1
            print(f'Сохранено {len(tracks)} треков')
        elif key == 112: # p
            draw_tracks(tracks)
        elif key == 114: # r
            drawRectifiedImages(binary_img1, binary_img2)
        elif key == 116: # t
            make_report(tracks)
        elif key == 118: # v
            save_tracks_to_excel(experiment, tracks)
        elif key == 96: # `
            do_processing = not do_processing
            position_changed = True
        elif key == 122: # z
            with open('results.dat', 'wb') as f:
                pickle.dump(tracks, f)
        elif key == 120: # x
            with open('results.dat', 'rb') as f:
                tracks = pickle.load(f)
                print(f'Загружено {len(tracks)} треков, последний обработанный файл {tracks[-1].image_name}')
        else:
            print(key) # else print its value
    

if __name__ == "__main__":

    EXPERIMENT_NUMBER = 0

    main_processing_loop()
