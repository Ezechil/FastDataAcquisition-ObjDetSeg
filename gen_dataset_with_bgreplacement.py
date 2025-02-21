import copy
import os
import click
import numpy as np
import cv2
import io
import sys
import re
import json
import PIL.Image as Image
from tqdm import tqdm
from glob import glob
from typing import Tuple
from multiprocessing import Process, Manager, Lock
import noise
import time
import signal


def generate_perlin_noise(width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    # Generate Perlin noise using the noise library
    if seed is not None:
        np.random.seed(seed)
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=np.random.randint(0, 1000))
    return world
 
def normalize_array(arr):
    # Normalize array values to the range [0, 1]
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)
 
def perlin_noise(img, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    # Generate Perlin noise and create an image
    perlin_noise = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed)
    normalized_noise = normalize_array(perlin_noise)
    normalized_noise = np.repeat(np.expand_dims(normalized_noise, axis=2),3 ,axis=2)
    img = img * normalized_noise
    return img
    
def fill_holes(binary_mask):
    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
 
    # Create a mask to fill holes
    holes_mask = np.zeros_like(binary_mask)
 
    # Draw contours on the holes_mask
    for cnt in contours:
        cv2.drawContours(holes_mask,[cnt],0,255,-1)
 
    # Bitwise OR with the original binary mask to fill the holes
    filled_mask = cv2.bitwise_or(binary_mask, holes_mask)
 
    return filled_mask

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255).astype('uint8')
 
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size
 
    # Add salt noise
    salt_pixels = int(total_pixels * salt_prob)
    salt_coordinates = [np.random.randint(0, i - 1, salt_pixels) for i in image.shape]
    noisy_image[salt_coordinates] = 255
 
    # Add pepper noise
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coordinates = [np.random.randint(0, i - 1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coordinates] = 0
 
    return noisy_image
 
def add_speckle_noise(image, scale=0.1):
    noise = np.random.normal(0, scale, image.shape).astype('uint8')
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype('uint8')
    
def gaussian_blurr(image):
    ''' Add some noise to the image '''
    kx, ky = (0, 0)
    while True:
        kx = np.random.randint(low=4, high=8) - 1
        ky = np.random.randint(low=4, high=8) - 1
        if kx % 2 == 1 and ky % 2 == 1: break
    img = cv2.GaussianBlur(image, (kx, ky), 0)
    return img
    
'''
img: used background image
returns:
  img: the background image with cropped objects
  id: the categories of the objects
  label: the labels of the objects [e.g. bbox, class etc.]
'''
def apply_BGreplacement(img, crop_names):
    num_crops = 5
    height, width = img.shape[0:2]
    crops_to_place = sample_crops(crop_names, (img.shape[0], img.shape[1]))
    
    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    bboxes = []
    areas = []
    # these two lists are used to cover the condition if the new object overlaps objects or not
    accepted_masks = []
    overlapped_masks = [] 
        
    success = 0
    for idx, crop in enumerate(crops_to_place):
        scale_decrease = 0
        while success < num_crops:
            try:
                ''' sample size relative to bg 
                between 0.20 and 0.3 of image height'''
                
                R = cv2.getRotationMatrix2D((crop.shape[1]//2, crop.shape[0]//2), np.random.randint(low=1, high=360), 1.0)
                crop_rotated = cv2.warpAffine(crop, R, (crop.shape[0], crop.shape[1]))
                # we want big objects but without massive intersections. If it is not possible, lower the scale by scale_decrease iteratively.
                scale = 0.6 + np.random.rand() * 0.1 - min(scale_decrease, 0.4)
                crop_rotated_scaled = cv2.resize(crop_rotated, (int(scale * height), int(scale * width)), interpolation=cv2.INTER_LANCZOS4)
                crop_mask = cv2.threshold(crop_rotated_scaled[...,0], thresh=1, maxval=255, type=0)[1]

                # ''' erode mask to remove jaggies '''
                kernel = np.ones((9, 9), np.uint8)  
                crop_mask = cv2.erode(crop_mask, kernel) 
                    
                # sample position to place scaled crop
                w_end = max(crop_rotated_scaled.shape[1], int(np.random.rand() * (width - 1)))
                h_end = max(crop_rotated_scaled.shape[0], int(np.random.rand() * (height - 1)))
                w_start = w_end - crop_rotated_scaled.shape[1]
                h_start = h_end - crop_rotated_scaled.shape[0]
                
                new_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
                new_mask[h_start:h_start+crop_rotated_scaled.shape[0], w_start:w_start+crop_rotated_scaled.shape[1]] = crop_mask

                ''' Get BB '''
                contours, hierarchy = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                    # find the biggest countour (c) by the area
                    c = max(contours, key = cv2.contourArea)
                    area = int(cv2.contourArea(c))
                    bb_x,bb_y,bb_w,bb_h = cv2.boundingRect(c) 

                    # check overlap of current object with objects that already exist
                    object_accepted = True
                    # ensure only small overlapping of background objects
                    inter_cnt = 0
                    for acc_idx, acc_mask in enumerate(accepted_masks):
                        mask_intersect = cv2.bitwise_and(acc_mask, new_mask)                           
                        # object is not overlapped -> accepted for this object
                        if np.count_nonzero(mask_intersect) == 0: continue
                        # object overlapps to much -> not accepted
                        if np.count_nonzero(mask_intersect) >= (np.count_nonzero(acc_mask)*0.2):
                            object_accepted = False
                            break
                        else:
                            # every object should be overlapped at most once
                            if acc_idx in overlapped_masks:
                                object_accepted = False
                                break
                            else: overlapped_masks.append(acc_idx)
                    
                    if object_accepted:
                        # place crop and break
                        new_img = np.zeros(shape=img.shape, dtype=np.uint8)
                        new_img[h_start:h_start+crop_rotated_scaled.shape[0], w_start:w_start+crop_rotated_scaled.shape[1]] = crop_rotated_scaled#_rotated

                        ''' erode mask to remove jaggies '''
                        kernel = np.ones((6, 6), np.uint8)  
                        new_mask = cv2.erode(new_mask, kernel) 
                        ''' add gaussian blur for alpha blending '''
                        new_mask = cv2.GaussianBlur(new_mask, (3, 3), 0)

                        img = img.astype(float)/255
                        new_img = new_img.astype(float)/255
                        alpha = new_mask.astype(float)/255
                        alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2)
                        # Multiply the foreground with the alpha matte
                        foreground = cv2.multiply(alpha, new_img)
                        # Multiply the background with ( 1 - alpha )
                        background = cv2.multiply(1.0 - alpha, img)
                        # Add the masked foreground and background.
                        img = (cv2.add(foreground, background)*255).astype(np.uint8)
                        img[img > 255] = 255
                        mask = cv2.bitwise_or(mask, new_mask)

                        x1 = np.max((0, bb_x))
                        y1 = np.max((0, bb_y))
                        x2 = np.max((0, bb_w))
                        y2 = np.max((0, bb_h))
                        
                        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                        areas.append(area)
                        accepted_masks.append(np.copy(new_mask))
                        success += 1
                        break
                    else: 
                        scale_decrease += 0.02
            except Exception as e: 
                print("Error in background replacement: ", e)
                break
        if success >= num_crops:
            break
    
    assert img.shape[2] == 3, "image from bgrep has invalid shape"
    return img, bboxes, areas, int(height), int(width)

def load_image(img_file, resolution, is_background=False, soft_resize=True):
    if is_background:
        img = Image.open(img_file)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = Image.open(img_file)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    # resize image to given resolution
    if soft_resize:
        scale = min(resolution[0]/img.shape[1], resolution[1]/img.shape[0])
        img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])), interpolation=cv2.INTER_LANCZOS4)
    else: 
        # random cropping
        if img.shape[0] < resolution[1] or img.shape[1] < resolution[0]:
            # image too small for cropping -> first resize than crop
            scale = max(resolution[0]/img.shape[1], resolution[1]/img.shape[0])
            img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])), interpolation=cv2.INTER_LANCZOS4)
            
        w_end = max(resolution[0], int(np.random.rand() * (img.shape[1] - 1)))
        h_end = max(resolution[1], int(np.random.rand() * (img.shape[0] - 1)))
        w_start = w_end - resolution[0]
        h_start = h_end - resolution[1]
        img = img[h_start:h_start+resolution[1], w_start:w_start+resolution[0]]
        
        if resolution[0] != img.shape[1] or resolution[1] != img.shape[0]:
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_LANCZOS4)
    assert img is not None, f"file named {img_file} not found"
    return img

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

def sample_crops(names, resolution):
    ''' Process only images, not masks'''
    crop_images = []
    for idx, crop in enumerate(names):
        # load bgra image
        crop_img = load_image(crop, resolution)
        # extract mask and rgb part
        mask = fill_holes(crop_img[:,:,-1]) # fill holes
        mask = np.repeat(np.expand_dims(mask, axis=2),3 ,axis=2)
        crop_img = crop_img[:,:,:3]
        img_masked = cv2.bitwise_and(crop_img, mask)
        crop_images.append(np.copy(img_masked))
    return crop_images     
        
def task_pp(lock, rank, backgrounds, resolution, empty_percentage, img_nr, annotation_id, current_annotations, coco_images, crops, images, image_folder, annotations, no_repl_percentage):
    np.random.seed((rank * int(time.time())) % 123456789)
    # iterate over 
    for bg_id, bg in enumerate(backgrounds):
        annos = []
        if no_repl_percentage != None and bg_id % no_repl_percentage == 0:
            # load cropped object
            crop_index = np.random.choice(np.arange(0,len(coco_images)-1,1), size=1, replace=False)
            crop_name = os.path.join(crops, coco_images[crop_index[0]]['file_name'])
            # get annotation
            new_anno = copy.deepcopy(current_annotations[crop_index[0]])
            new_anno['category_id'] = int(new_anno['category_id'])
            annos.append(new_anno)
            bboxes = [new_anno['bbox']]
            areas = [new_anno['area']]
            # get image
            img = sample_crops([crop_name], resolution)[0]
            height, width = int(img.shape[0]), int(img.shape[1])
        else:
            # load Background
            img = load_image(bg, resolution, True, False)
            height, width = int(img.shape[0]), int(img.shape[1])
            assert resolution[0] == width and resolution[1] == height, "background cropping invalid %d %d" % (width, height) 
            # random background blurr
            if np.random.rand() > 0.7:
                img = gaussian_blurr(img)
                    
            # do BGrepl otherwise just pass empty background
            if not (empty_percentage != None and bg_id % empty_percentage == 0):
                # create random indices
                crop_indices = np.random.choice(np.arange(0,len(coco_images)-1,1), size=5, replace=False)
                
                crop_names = []
                annos = []
                # process annotation
                for index in crop_indices:
                    crop_names.append(os.path.join(crops, coco_images[index]['file_name']))
                    # get annotation
                    new_anno = copy.deepcopy(current_annotations[index])
                    new_anno['category_id'] = int(new_anno['category_id'])
                    annos.append(new_anno)
                                
                img, bboxes, areas, height, width = apply_BGreplacement(img, crop_names)
            
        lock.acquire()
        try:
            for idx, anno in enumerate(annos):
                new_anno = copy.deepcopy(anno) 
                new_anno['bbox'] = bboxes[idx]
                new_anno['area'] = areas[idx]
                new_anno['id'] = int(annotation_id.value)
                new_anno['image_id'] = int(img_nr.value)
                annotations.append(new_anno)
                annotation_id.value += 1 
            # store processed image
            new_image_name = '{:06d}'.format(img_nr.value)+'.png'
            cv2.imwrite(os.path.join(image_folder,new_image_name), img)
            
            # process image data
            images.append({'id': int(img_nr.value), 'file_name': 'images/'+new_image_name, 'width': width, 'height': height, 'date_captured': '', 'license': 1, 'coco_url': '', 'flickr_url': ''})
            if int(img_nr.value) % 200 == 0:
                print("image: %d" % img_nr.value)
                sys.stdout.flush()
            img_nr.value += 1
        finally:
            lock.release()
    
@click.command()
@click.pass_context
@click.option('--num_procs', help='Number of processes', required=False, type=int)
@click.option('--crops', help='Path to the luma data', required=False, metavar='PATH')
@click.option('--bgs', help='Path to the folder with images to be used as background', required=False, metavar='PATH')
@click.option('--num_imgs', help='Number of resulting images', required=False, type=int)
@click.option('--dest', help='Output folder', required=False, metavar='PATH')
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', required=False, metavar='WxH', type=parse_tuple)
@click.option('--num_crops', help='Number of resulting images without BGreplacement (the cropped object)', required=False, type=float, default=0.0)
@click.option('--num_empty', help='Number of resulting images with only BG, no objects', required=False, type=float, default=0.0)
def convert_dataset(
    ctx: click.Context,
    num_procs: int,
    crops: str,
    bgs: str,
    num_imgs: int,
    dest: str,
    resolution: str,
    num_crops: float,
    num_empty: float
):    
    num_procs = 6
    crops = "new_out"
    bgs = "Backgrounds"    
    dest = "Final_Data"
    num_imgs = 1000
    num_crops = 0.2
    num_empty = 0.2
    resolution=(1024, 1024)
              
    backgrounds = glob(f"%s/*.png" % bgs)
    # artificially extend dataset
    bg_choices = np.random.choice(np.arange(0,len(backgrounds)-1,1), size=num_imgs, replace=True)
    backgrounds = [backgrounds[idx] for idx in bg_choices]
    output_directory = dest+"_bgrepl_%dx%d" % (resolution[0], resolution[1])
    
    # create output directory
    print("output_directory: ", output_directory)
    os.makedirs(output_directory, exist_ok=True)
    image_folder = os.path.join(output_directory, "images")
    os.makedirs(image_folder, exist_ok=True)
    
    with open(os.path.join(crops, 'scene_gt_coco.json'), 'r') as f:
        coco_annotations = json.load(f)
    current_annotations = coco_annotations['annotations']
    coco_images = coco_annotations['images']
    
    # init categories
    i = 1
    categories = []
    while i < 22:
        categories.append({'id': i, 'name': i, 'supercategory': output_directory})
        i += 1
     
    # create output coco annotation
    annotation = {}
    annotation['categories'] = categories
    
    print("Start processing images")
    images = []
    annotations = []
    
    img_nr = 1
    annotation_id = 1
    
    # percentage of empty images
    empty_percentage = int(1/num_empty) if num_empty > 0.0 else None 
    no_repl_percentage = int(1/num_crops) if num_crops > 0.0 else None
    
    print("empty_percentage, no_repl_percentage: ", empty_percentage, no_repl_percentage)
    
    # spawn processes
    with Manager() as manager:
        img_nr = manager.Value('i', img_nr)
        annotation_id = manager.Value('i', annotation_id)
        images = manager.list(images)
        annotations = manager.list(annotations)
        lock = manager.Lock()
        
        global processes
        processes = []
        bgs_per_process = len(backgrounds)//num_procs
        for rank in range(num_procs):
            if rank != num_procs-1:
                backgrounds_pp = backgrounds[rank*bgs_per_process:(rank+1)*bgs_per_process]
            else: backgrounds_pp = backgrounds[rank*bgs_per_process:]
            print("number backgrounds, rank: ", len(backgrounds_pp), rank)
            print("Start on rank %s with %d bgs: "% (rank, len(backgrounds_pp)))
            p = Process(target=task_pp, args=(lock, rank, backgrounds_pp, resolution, empty_percentage, img_nr, annotation_id, current_annotations, coco_images, crops, images, image_folder, annotations, no_repl_percentage))
            processes.append(p)
            
        # start processes
        for p in processes:
            p.start()
        # finish processes
        for p in processes:
            p.join()
        
        annotation['images'] = list(images)
        annotation['annotations'] = list(annotations)
           
        # write json file
        with open(os.path.join(output_directory, 'scene_gt_coco.json'), 'w') as json_file:
            json.dump(annotation, json_file)
           
        for p in processes:
            p.close()
            
def interrupt_handler(signum, frame):
    print('Cleaning/exiting...')
    for p in processes:
        if p.is_alive():
            p.close()
    time.sleep(1)
    sys.exit(0)
    
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGABRT, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler) 
    convert_dataset()