"""
Description:
pipeline class

Requirements:
this pipeline class, needs to have the following methods:
    load
    clean_up
    warm_up
    predict
"""

import os
import time
import shutil
import collections
import numpy as np
import cv2
import logging
import sys

#own modules
sys.path.append('/home/gadget/workspace/LMI_AI_Solutions/object_detectors')
sys.path.append('/home/gadget/workspace/LMI_AI_Solutions/lmi_utils')

from yolov8_lmi.model import Yolov8
import gadget_utils.pipeline_utils as pipeline_utils

RESIZE_W = 640
RESIZE_H = 640
TARGET_CLASSES = ['inner', 'outer']

class ModelPipeline:

    logger = logging.getLogger()

    def __init__(self, **kwargs):
        """
        class_map: the class id -> class name
        """
        np.random.seed(777)
        COLORMAP=[(0,255,0), (0,0,255)] #RGB
        self.det_configs = {}
        self.det_configs['engine_path'] = os.path.realpath(os.path.expandvars(kwargs.get('path_engine',"")))
        self.det_configs['id_to_cls'] = pipeline_utils.convert_key_to_int(kwargs.get('class_map', {}))
        self.det_configs['cls_to_id'] = pipeline_utils.val_to_key(self.det_configs['id_to_cls'])
        
        self.det_configs['conf_thres'] = {
            c: kwargs.get(f'confidence_{c}',0.5) for c in TARGET_CLASSES
        }
        
        self.det_configs['color_map'] = {
            c:COLORMAP[i] if COLORMAP and i<len(COLORMAP) else np.random.randint(0,255,size=3).tolist() 
            for i,c in enumerate(TARGET_CLASSES)
        }

        #map model name -> model instance
        self.models = collections.OrderedDict()
        self.frame_index = 0
        
        
    def load(self):
        """
        create model instances with weight files
        if loading files fail, then don't create model instances
        """
        try:
            self.models['det'] = Yolov8(self.det_configs['engine_path'])
            self.logger.info('models are loaded')
        except Exception:
            self.logger.exception('models are failed to load')
            self.models = None
        

    def clean_up(self):
        """
        clean up the pipeline in REVERSED order, i.e., the last models get destroyed first
        """
        L = list(reversed(self.models.keys())) if self.models else []
        self.logger.info('cleanning up pipeline...')
        for model_name in L:
            del self.models[model_name]
            self.logger.info(f'{model_name} is cleaned up')

        #del the pipeline
        del self.models
        self.models = None
        self.logger.info('pipeline is cleaned up')


    def warm_up(self):
        """
        warm up all the models in the pipeline
        """
        if not self.models:
            return
        for model_name in self.models:
            imgsz = self.models[model_name].imgsz
            self.models[model_name].warmup(imgsz)
            self.logger.info(f'warming up {model_name} on the input size of {imgsz}')
        
    def internal_preprocess(self, image, profile) -> tuple:
        """_summary_

        Args:
            image (np.ndarray): _description_

        Returns:
            tuple: _description_
        """
        
        from image_utils.img_stretch import stretch
        BLACK = (0,0,0)

        def process_p(img_p):
            # empty pixels receive value=0
            # replace empty pixels with min above 0
            minval=img_p[img_p>0].min()
            floor=minval-1
            empty_ind=np.where(img_p==0)
            img_p[empty_ind]=floor
            img_p=img_p-floor
            maxval=img_p.max()
            img_p=(img_p.astype(np.float32)/maxval)*255.0
            img_p=img_p.astype(np.uint8)
            img_p=cv2.applyColorMap(img_p,cv2.COLORMAP_JET)
            img_p[empty_ind]=BLACK
            img_p = img_p[:,:,::-1]
            return img_p

        def blend(img_p, img_i):
            # Blend the images
            alpha = 0.3  # Weight for image1
            beta = 0.7  # Weight for image2
            gamma = 0  # Scalar added to each pixel
            blended_image = cv2.addWeighted(img_p, alpha, img_i, beta, gamma)
            return blended_image
        
        intensity = stretch(image, 0.67)
        profile = stretch(profile, 0.67)
        profile = process_p(profile)
        self.logger.info(f'{intensity.shape}, {profile.shape}')
        blended = blend(profile, intensity)
        
        operators_det = {}
        img_det = cv2.resize(blended, [RESIZE_W, RESIZE_H], interpolation=cv2.INTER_LINEAR)
        th,tw = img_det.shape[:2]
        operators_det = [{'resize':[tw,th,RESIZE_W,RESIZE_H]}]
        return img_det,operators_det


    def det_predict(self, image, operators, configs, annotated_image=None):
        time_info = {}
        
        # preprocess
        t0 = time.time()
        im = self.models['det'].preprocess(image)
        time_info['preproc'] = time.time()-t0
        
        # infer
        t0 = time.time()
        pred = self.models['det'].forward(im)
        time_info['proc'] = time.time()-t0
        
        # postprocess
        t0 = time.time()
        conf_thres = {}
        for k in configs:
            if k not in self.det_configs['cls_to_id']:
                continue
            if k in TARGET_CLASSES:
                conf_thres[self.det_configs['cls_to_id'][k]] = configs[k]
        self.logger.debug(f'[DET] configs: {configs}')
        self.logger.debug(f'[DET] conf_thres: {conf_thres}')
        result_dets = self.models['det'].postprocess(pred,im,image, conf_thres[self.det_configs['cls_to_id'][TARGET_CLASSES[0]]])
        
        # get the first batch
        boxes = result_dets['boxes'][0]
        scores = result_dets['scores'][0]
        classes = result_dets['classes'][0]
        
        # convert coordinates to sensor space
        boxes = pipeline_utils.revert_to_origin(boxes, operators)
        
        # annotation
        if annotated_image is not None:
            for j,box in enumerate(boxes):
                pipeline_utils.plot_one_box(
                    box,
                    annotated_image,
                    label="{}: {:.2f}".format(
                        classes[j], scores[j]
                    ),
                    color=self.det_configs['color_map'][classes[j]],
                )
        
        results_dict = {
            'boxes': boxes,
            'classes': classes,
            'scores': scores.tolist(),
        }
        time_info['postproc'] = time.time()-t0
        return results_dict, time_info



    def predict(self, configs: dict, image: np.ndarray, profile: np.ndarray = None,  **kwargs) -> dict:
        errors = []
        result_dict = {
                'anotated_output': None,
                'automation_keys': [],
                'factory_keys': [],
                'should_archive': False,
                'errors': errors,
                'total_proc_time': 0,
            }
        
        if not self.models:
            errors.append('failed to load pipeline model(s)')
            self.logger.exception('failed to load pipeline model(s)')
            return result_dict
        
        # preprocess
        t0 = time.time()
        img_det,operators_det = self.internal_preprocess(image, profile)
        preproc_time = time.time()-t0
        
        # load runtime config
        test_mode = configs.get('test_mode', False)
        conf_thres = {
            c:configs.get(f'confidence_{c}', self.det_configs['conf_thres'][c]) for c in TARGET_CLASSES 
        }
        
        annotated_image = img_det.copy()
        
        try:
            results_dict1, time_info1 = self.det_predict(img_det, operators_det, conf_thres, annotated_image)
        except Exception as e:
            self.logger.exception('failed to run detection model')
            errors.append(str(e))
            return result_dict
            
        if test_mode:
            outpath = kwargs.get('results_storage_path','./outputs')
            annotated_image_path = os.path.join(outpath, 'annotated_'+str(self.frame_index).zfill(4))+'.png'
            annot_bgr = cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(annotated_image_path, annot_bgr)
            
        # gather time info
        preprocess_time = time_info1['preproc'] + preproc_time
        inference_time = time_info1['proc']
        postprocess_time = time_info1['postproc']
        total_time = preprocess_time+inference_time+postprocess_time
        
        result_dict['anotated_output'] = annotated_image
        obj_list = results_dict1['classes']
        if not len(obj_list):
            result_dict['should_archive'] = True
        result_dict['automation_keys'] = []
        result_dict['factory_keys'] = ['det_boxes','det_scores','det_classes','total_proc_time']
        result_dict['det_boxes'] = results_dict1['boxes']
        result_dict['det_scores'] = results_dict1['scores']
        result_dict['det_classes'] = results_dict1['classes']
        result_dict['errors'] = errors
        
        self.logger.info(f'found objects: {obj_list}')
        self.logger.info(f'preprocess:{preprocess_time:.4f}, inference:{inference_time:.4f}, ' +
            f'postprocess:{postprocess_time:.4f}, total:{total_time:.4f}\n')
        self.frame_index += 1
        
        return result_dict



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    BATCH_SIZE = 1
    os.environ['PIPELINE_SERVER_SETTINGS_MODELS_ROOT'] = './pipeline/trt-engines'
    pipeline_def_file = './pipeline/pipeline_def.json'
    image_dir = './data/test_images'
    output_dir = './data/outputs'
    im_fmt = 'png'
    
    kwargs = pipeline_utils.load_pipeline_def(pipeline_def_file)
    pipeline = ModelPipeline(**kwargs)
    
    logger.info('start loading the pipeline...')
    pipeline.load()
    pipeline.warm_up()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    intensity_dir = os.path.join('./data', 'intensity')
    profile_dir = os.path.join('./data', 'profile')
    image_path_batches = pipeline_utils.get_gadget_img_batches(BATCH_SIZE, profile_dir, intensity_dir, fmt=im_fmt)
    for batch in image_path_batches:
        for dt in batch:
            if im_fmt=='png':
                im_bgr = cv2.imread(dt['intensity'], cv2.IMREAD_UNCHANGED)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                profile = cv2.imread(dt['profile'], cv2.IMREAD_UNCHANGED)
            # elif im_fmt=='npy':
            #     im = np.load(image_path)
            pipeline.predict(image=im, profile=profile, configs={'test_mode':True}, results_storage_path=output_dir)
            break
    pipeline.clean_up()