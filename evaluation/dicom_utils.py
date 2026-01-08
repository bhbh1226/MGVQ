"""
DICOM 이미지 읽기 유틸리티

단일 DICOM 파일을 2D RGB 이미지로 변환하는 기능을 제공합니다.
주요 기능:
- DICOM 파일 로드 및 HU 변환
- Window/Level 적용 (CT 이미지용)
- 8bit RGB 이미지로 변환
"""
from pathlib import Path
from typing import Tuple, Optional, Union, List
import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


# 사전 정의된 HU Window 프리셋
HU_WINDOWS = {
    "lung": (-1000, 500),
    "soft_tissue": (-150, 250),
    "bone": (-500, 1500),
    "brain": (0, 80),
    "liver": (-20, 200),
    "mediastinum": (-125, 225),
    "abdomen": (-125, 225),
    "default": (-1000, 1000),
}


def check_pydicom_available() -> bool:
    """pydicom 패키지 설치 여부 확인"""
    return PYDICOM_AVAILABLE


def get_hu_window(window_name: str) -> Tuple[float, float]:
    """사전 정의된 HU window 값 반환
    
    Args:
        window_name: HU window 이름 (lung, soft_tissue, bone, brain, liver, mediastinum, abdomen, default)
        
    Returns:
        (min_hu, max_hu) 튜플
    """
    return HU_WINDOWS.get(window_name, HU_WINDOWS["default"])


def apply_hu_transform_single(ds: 'FileDataset') -> np.ndarray:
    """단일 DICOM 슬라이스에 HU 변환 적용
    
    Args:
        ds: pydicom FileDataset 객체
        
    Returns:
        HU 변환된 2D numpy 배열
    """
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu_array = pixel_array * slope + intercept
    return hu_array


def apply_window_level(
    hu_array: np.ndarray, 
    window_min: float, 
    window_max: float
) -> np.ndarray:
    """Window/Level 적용하여 [0, 255] 범위로 변환
    
    Args:
        hu_array: HU 값 배열
        window_min: 최소 HU 값
        window_max: 최대 HU 값
        
    Returns:
        [0, 255] 범위의 uint8 배열
    """
    # HU 값을 window 범위로 클리핑
    clipped = np.clip(hu_array, window_min, window_max)
    # [0, 255]로 정규화
    normalized = (clipped - window_min) / (window_max - window_min) * 255
    return normalized.astype(np.uint8)


def read_dicom_as_rgb(
    dicom_path: Union[str, Path],
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
) -> np.ndarray:
    """DICOM 파일을 RGB 이미지로 읽기
    
    Args:
        dicom_path: DICOM 파일 경로
        window_name: 사전 정의된 window 이름 (window_min/max 미지정시 사용)
        window_min: 최소 HU 값 (직접 지정)
        window_max: 최대 HU 값 (직접 지정)
        
    Returns:
        RGB 이미지 numpy 배열 (H, W, 3), uint8
        
    Raises:
        ImportError: pydicom이 설치되지 않은 경우
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom이 설치되지 않았습니다. "
            "'pip install pydicom' 명령으로 설치해주세요."
        )
    
    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM 파일을 찾을 수 없습니다: {dicom_path}")
    
    # DICOM 파일 로드
    ds = pydicom.dcmread(str(dicom_path), force=True)
    
    # HU 변환 적용
    hu_array = apply_hu_transform_single(ds)
    
    # Window/Level 결정
    if window_min is None or window_max is None:
        # DICOM 헤더에서 window 정보 시도
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            # 다중 window인 경우 첫 번째 사용
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])
            else:
                wc = float(wc)
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])
            else:
                ww = float(ww)
            window_min = wc - ww / 2
            window_max = wc + ww / 2
        elif window_name:
            window_min, window_max = get_hu_window(window_name)
        else:
            # 기본값 사용
            window_min, window_max = get_hu_window("default")
    
    # Window 적용
    gray_array = apply_window_level(hu_array, window_min, window_max)
    
    # RGB로 변환 (grayscale -> RGB)
    rgb_array = np.stack([gray_array, gray_array, gray_array], axis=-1)
    
    return rgb_array


def resize_center_crop_dicom(
    dicom_path: Union[str, Path],
    target_width: int,
    target_height: int,
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
) -> np.ndarray:
    """DICOM 파일을 읽고 resize 및 center crop 적용
    
    eval_aug.py의 resize_center_crop과 동일한 로직을 DICOM에 적용
    
    Args:
        dicom_path: DICOM 파일 경로
        target_width: 목표 너비
        target_height: 목표 높이
        window_name: HU window 이름
        window_min: 최소 HU 값
        window_max: 최대 HU 값
        
    Returns:
        center crop된 RGB 이미지 numpy 배열 (H, W, 3), uint8
    """
    from PIL import Image
    
    # DICOM을 RGB로 읽기
    rgb_array = read_dicom_as_rgb(
        dicom_path, 
        window_name=window_name,
        window_min=window_min, 
        window_max=window_max
    )
    
    # PIL Image로 변환
    im_pil = Image.fromarray(rgb_array)
    w, h = im_pil.size
    
    # resize to eval_res
    scale = target_width / target_height
    scale_im = w / h
    
    if scale_im < scale:
        h_1 = round(h / w * target_width) 
        im = np.array(im_pil.resize((target_width, h_1), resample=Image.BICUBIC))
    else:
        w_1 = round(w / h * target_height) 
        im = np.array(im_pil.resize((w_1, target_height), resample=Image.BICUBIC)) 
    
    # center crop
    ih, iw, _ = im.shape
    if iw == target_width:
        x = int(ih / 2 - target_height / 2)
        y = 0
        im = im[x:x+target_height, y:y+target_width, :]
    else:  # ih == target_height
        x = 0
        y = int(iw / 2 - target_width / 2)
        im = im[x:x+target_height, y:y+target_width, :]
    
    return im


def collect_dicom_files(root_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    """폴더에서 DICOM 파일 수집
    
    Args:
        root_path: 검색할 루트 폴더
        recursive: 하위 폴더도 검색할지 여부
        
    Returns:
        DICOM 파일 경로 리스트
    """
    root_path = Path(root_path)
    dicom_files = []
    
    # DICOM 확장자 패턴
    extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
    
    for ext in extensions:
        if recursive:
            dicom_files.extend(root_path.rglob(ext))
        else:
            dicom_files.extend(root_path.glob(ext))
    
    # 확장자가 없는 DICOM 파일도 확인 (선택적)
    # 파일이 DICOM인지 확인하려면 pydicom으로 실제 읽어봐야 함
    
    return sorted(dicom_files)


def is_dicom_file(file_path: Union[str, Path]) -> bool:
    """파일이 DICOM 파일인지 확인
    
    Args:
        file_path: 확인할 파일 경로
        
    Returns:
        DICOM 파일이면 True
    """
    if not PYDICOM_AVAILABLE:
        # pydicom 없으면 확장자로만 판단
        file_path = Path(file_path)
        return file_path.suffix.lower() in ['.dcm', '.dicom']
    
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False
