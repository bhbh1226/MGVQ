import numpy as np
from PIL import Image
from typing import Union

def resize_center_crop(img_input: Union[str, Image.Image], test_res_w: int, test_res_h: int) -> np.ndarray:
        """
        짧은 쪽을 목표 크기에 맞추고 긴 쪽을 center crop
        
        Args:
            img_input: 이미지 파일 경로 또는 PIL Image 객체
            test_res_w: 목표 너비
            test_res_h: 목표 높이
        
        Returns:
            numpy array: test_res_h x test_res_w x 3 크기의 이미지
        """
        # PIL Image 객체인지 파일 경로인지 확인
        if isinstance(img_input, Image.Image):
            im_pil = img_input.convert('RGB')
        else:
            im_pil = Image.open(img_input).convert('RGB')
        
        w, h = im_pil.size
        scale = test_res_w / test_res_h
        scale_im = w / h
        
        if scale_im < scale:
            h_1 = round(h / w * test_res_w) 
            im = np.array(im_pil.resize((test_res_w, h_1), resample=Image.BICUBIC))
        else:
            w_1 = round(w / h * test_res_h) 
            im = np.array(im_pil.resize((w_1, test_res_h), resample=Image.BICUBIC)) 
        
        # center crop
        ih, iw, _ = im.shape
        if iw == test_res_w:
            x = int(ih/2-test_res_h/2)
            y = 0
            assert y + test_res_w == iw
            im = im[x:x+test_res_h, y:y+test_res_w, :]
        else: # ih == test_res_h
            x = 0
            y = int(iw/2-test_res_w/2)
            assert x + test_res_h == ih
            im = im[x:x+test_res_h, y:y+test_res_w, :]
        return im

def resize_fit_pad(img_input: Union[str, Image.Image], test_res_w: int, test_res_h: int, pad_value: int = 0) -> np.ndarray:
        """
        긴 쪽을 목표 크기에 맞추고 짧은 쪽은 패딩으로 채움
        
        Args:
            img_input: 이미지 파일 경로 또는 PIL Image 객체
            test_res_w: 목표 너비
            test_res_h: 목표 높이
            pad_value: 패딩 값 (0-255), 기본값은 0 (검은색)
        
        Returns:
            numpy array: test_res_h x test_res_w x 3 크기의 이미지
        """
        # PIL Image 객체인지 파일 경로인지 확인
        if isinstance(img_input, Image.Image):
            im_pil = img_input.convert('RGB')
        else:
            im_pil = Image.open(img_input).convert('RGB')
        
        w, h = im_pil.size
        scale = test_res_w / test_res_h
        scale_im = w / h
        
        # 긴 쪽을 목표 크기에 맞춤
        if scale_im > scale:
            # 너비가 더 길면, 너비를 목표 크기에 맞춤
            w_1 = test_res_w
            h_1 = round(h / w * test_res_w)
            im = np.array(im_pil.resize((w_1, h_1), resample=Image.BICUBIC))
        else:
            # 높이가 더 길면, 높이를 목표 크기에 맞춤
            h_1 = test_res_h
            w_1 = round(w / h * test_res_h)
            im = np.array(im_pil.resize((w_1, h_1), resample=Image.BICUBIC))
        
        # 패딩 추가
        ih, iw, _ = im.shape
        pad_h = test_res_h - ih
        pad_w = test_res_w - iw
        
        # 중앙에 배치하도록 상하좌우 패딩 계산
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # numpy pad 사용
        im_padded = np.pad(
            im, 
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
        
        return im_padded

def resize_stretch(img_input: Union[str, Image.Image], test_res_w: int, test_res_h: int) -> np.ndarray:
        """
        비율 무시하고 목표 크기에 맞게 강제 resize (stretch)
        
        Args:
            img_input: 이미지 파일 경로 또는 PIL Image 객체
            test_res_w: 목표 너비
            test_res_h: 목표 높이
        
        Returns:
            numpy array: test_res_h x test_res_w x 3 크기의 이미지
        """
        # PIL Image 객체인지 파일 경로인지 확인
        if isinstance(img_input, Image.Image):
            im_pil = img_input.convert('RGB')
        else:
            im_pil = Image.open(img_input).convert('RGB')
        
        # 비율 무시하고 강제로 목표 크기에 맞춤
        im = np.array(im_pil.resize((test_res_w, test_res_h), resample=Image.BICUBIC))
        
        return im