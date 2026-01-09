"""
NIfTI 및 Tensor 이미지 읽기 유틸리티

NIfTI 파일 및 전처리된 Tensor 파일을 2D RGB 이미지로 변환하는 기능을 제공합니다.
주요 기능:
- NIfTI 파일 로드 및 슬라이스 추출
- Tensor 파일 로드 및 슬라이스 추출
- HU Windowing 적용 (CT 이미지용)
- 8bit RGB 이미지로 변환
"""
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np
from PIL import Image
import json

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import monai
    from monai.transforms import Compose
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# 사전 정의된 HU Window 프리셋
HU_WINDOWS = {
    "lung": (-1000, 500),
    "soft_tissue": (-150, 250),
    "bone": (-500, 1500),
    "brain": (0, 80),
    "liver": (-20, 200),
    "mediastinum": (-125, 225),
    "abdomen": (-125, 225),
}


def check_nibabel_available() -> bool:
    """nibabel 패키지 설치 여부 확인"""
    return NIBABEL_AVAILABLE


def check_torch_available() -> bool:
    """torch 패키지 설치 여부 확인"""
    return TORCH_AVAILABLE

def check_monai_available() -> bool:
    """monai 패키지 설치 여부 확인"""
    return MONAI_AVAILABLE

def get_hu_window(window_name: str) -> Tuple[float, float]:
    """사전 정의된 HU window 값 반환
    
    Args:
        window_name: HU window 이름 (lung, soft_tissue, bone, brain, liver, mediastinum, abdomen)
        
    Returns:
        (min_hu, max_hu) 튜플
    """
    return HU_WINDOWS.get(window_name, HU_WINDOWS["abdomen"])


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
    clipped = np.clip(hu_array, window_min, window_max)
    normalized = (clipped - window_min) / (window_max - window_min) * 255
    return normalized.astype(np.uint8)


def load_nifti_volume(nifti_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """NIfTI 파일에서 볼륨 데이터 로드
    
    Args:
        nifti_path: NIfTI 파일 경로
        
    Returns:
        (volume_data, metadata) 튜플
        - volume_data: 3D numpy 배열 (X, Y, Z) 또는 (Z, Y, X) - nibabel 순서
        - metadata: spacing, affine 등 메타데이터
        
    Raises:
        ImportError: nibabel이 설치되지 않은 경우
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError(
            "nibabel이 설치되지 않았습니다. "
            "'pip install nibabel' 명령으로 설치해주세요."
        )
    
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI 파일을 찾을 수 없습니다: {nifti_path}")
    
    img = nib.load(str(nifti_path))
    volume = img.get_fdata()
    
    header = img.header
    metadata = {
        "shape": volume.shape,
        "spacing": tuple(header.get_zooms()),
        "affine": img.affine,
    }
    
    return volume, metadata


def load_tensor_volume(tensor_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tensor 파일에서 볼륨 데이터 로드
    
    Args:
        tensor_path: .pt Tensor 파일 경로
        
    Returns:
        (volume_data, metadata) 튜플
        - volume_data: 3D numpy 배열 (Z, Y, X) 또는 (C, Z, Y, X)
        - metadata: 메타데이터 (있으면)
        
    Raises:
        ImportError: torch가 설치되지 않은 경우
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch가 설치되지 않았습니다. "
            "'pip install torch' 명령으로 설치해주세요."
        )
    
    if not MONAI_AVAILABLE:
        raise ImportError(
            "monai가 설치되지 않았습니다. "
            "'pip install monai' 명령으로 설치해주세요."
        )
    
    tensor_path = Path(tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor 파일을 찾을 수 없습니다: {tensor_path}")
    
    tensor = torch.load(str(tensor_path), map_location='cpu', weights_only=False)
    
    # tensor가 dict인 경우 (data, metadata 포함)
    if isinstance(tensor, dict):
        volume = tensor.get('data', tensor.get('volume', tensor.get('image')))
        if volume is None:
            # dict의 첫 번째 텐서 사용
            for key, val in tensor.items():
                if isinstance(val, torch.Tensor):
                    volume = val
                    break
        metadata = {k: v for k, v in tensor.items() if not isinstance(v, torch.Tensor)}
    else:
        volume = tensor
        metadata = {}
    
    # torch.Tensor -> numpy
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    
    # 채널 차원 제거 (C, Z, Y, X) -> (Z, Y, X)
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume.squeeze(0)
    
    # 메타데이터 파일 확인
    meta_path = tensor_path.with_suffix('.json')
    if not meta_path.exists():
        meta_path = tensor_path.parent / (tensor_path.stem + '_meta.json')
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            file_metadata = json.load(f)
            metadata.update(file_metadata)
    
    metadata['shape'] = volume.shape
    
    return volume, metadata


def extract_slice_from_volume(
    volume: np.ndarray,
    slice_idx: int,
    axis: int = 0,
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
    apply_windowing: bool = True,
) -> np.ndarray:
    """볼륨에서 2D 슬라이스 추출 및 RGB 변환
    
    Args:
        volume: 3D 볼륨 데이터
        slice_idx: 슬라이스 인덱스
        axis: 슬라이스 축 (0=sagittal, 1=coronal, 2=axial for RAS)
        window_name: 사전 정의된 window 이름
        window_min: 최소 HU 값 (직접 지정)
        window_max: 최대 HU 값 (직접 지정)
        apply_windowing: True이면 windowing 적용
        
    Returns:
        RGB 이미지 numpy 배열 (H, W, 3), uint8
    """
    # 슬라이스 추출
    if axis == 0:
        slice_2d = volume[slice_idx, :, :]
    elif axis == 1:
        slice_2d = volume[:, slice_idx, :]
    else:  # axis == 2
        slice_2d = volume[:, :, slice_idx]
    
    slice_2d = slice_2d.astype(np.float32)
    
    # Windowing 적용
    if apply_windowing:
        if window_min is not None and window_max is not None:
            pass
        elif window_name:
            window_min, window_max = get_hu_window(window_name)
        else:
            # 기본값: 데이터 범위 사용
            window_min = float(slice_2d.min())
            window_max = float(slice_2d.max())
    else:
        window_min = float(slice_2d.min())
        window_max = float(slice_2d.max())
    
    gray_array = apply_window_level(slice_2d, window_min, window_max)
    rgb_array = np.stack([gray_array, gray_array, gray_array], axis=-1)
    
    return rgb_array


def read_nifti_slice_as_rgb(
    nifti_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    axis: int = 2,  # axial by default
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
    apply_windowing: bool = True,
) -> np.ndarray:
    """NIfTI 파일에서 단일 슬라이스를 RGB 이미지로 읽기
    
    Args:
        nifti_path: NIfTI 파일 경로
        slice_idx: 슬라이스 인덱스 (None이면 중앙 슬라이스)
        axis: 슬라이스 축 (0=sagittal, 1=coronal, 2=axial)
        window_name: 사전 정의된 window 이름
        window_min: 최소 HU 값
        window_max: 최대 HU 값
        apply_windowing: windowing 적용 여부
        
    Returns:
        RGB 이미지 numpy 배열 (H, W, 3), uint8
    """
    volume, metadata = load_nifti_volume(nifti_path)
    
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    slice_idx = max(0, min(slice_idx, volume.shape[axis] - 1))
    
    return extract_slice_from_volume(
        volume, slice_idx, axis,
        window_name=window_name,
        window_min=window_min,
        window_max=window_max,
        apply_windowing=apply_windowing,
    )


def read_nifti_slice_as_pil(
    nifti_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    axis: int = 2,
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
    apply_windowing: bool = True,
) -> Image.Image:
    """NIfTI 파일에서 단일 슬라이스를 PIL Image로 읽기"""
    rgb_array = read_nifti_slice_as_rgb(
        nifti_path, slice_idx, axis,
        window_name=window_name,
        window_min=window_min,
        window_max=window_max,
        apply_windowing=apply_windowing,
    )
    return Image.fromarray(rgb_array)


def read_tensor_slice_as_rgb(
    tensor_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    axis: int = 2,  # axial by default
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
    apply_windowing: bool = True,
) -> np.ndarray:
    """Tensor 파일에서 단일 슬라이스를 RGB 이미지로 읽기
    
    Args:
        tensor_path: .pt Tensor 파일 경로
        slice_idx: 슬라이스 인덱스 (None이면 중앙 슬라이스)
        axis: 슬라이스 축 (0=sagittal, 1=coronal, 2=axial)
        window_name: 사전 정의된 window 이름
        window_min: 최소 HU 값
        window_max: 최대 HU 값
        apply_windowing: windowing 적용 여부
        
    Returns:
        RGB 이미지 numpy 배열 (H, W, 3), uint8
    """
    volume, metadata = load_tensor_volume(tensor_path)
    
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    slice_idx = max(0, min(slice_idx, volume.shape[axis] - 1))
    
    return extract_slice_from_volume(
        volume, slice_idx, axis,
        window_name=window_name,
        window_min=window_min,
        window_max=window_max,
        apply_windowing=apply_windowing,
    )


def read_tensor_slice_as_pil(
    tensor_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    axis: int = 2,
    window_name: Optional[str] = None,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
    apply_windowing: bool = True,
) -> Image.Image:
    """Tensor 파일에서 단일 슬라이스를 PIL Image로 읽기"""
    rgb_array = read_tensor_slice_as_rgb(
        tensor_path, slice_idx, axis,
        window_name=window_name,
        window_min=window_min,
        window_max=window_max,
        apply_windowing=apply_windowing,
    )
    return Image.fromarray(rgb_array)


def collect_nifti_files(root_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    """폴더에서 NIfTI 파일 수집
    
    Args:
        root_path: 검색할 루트 폴더
        recursive: 하위 폴더도 검색할지 여부
        
    Returns:
        NIfTI 파일 경로 리스트
    """
    root_path = Path(root_path)
    nifti_files = []
    
    # NIfTI 확장자 패턴
    extensions = ['*.nii', '*.nii.gz', '*.NII', '*.NII.GZ']
    
    for ext in extensions:
        if recursive:
            nifti_files.extend(root_path.rglob(ext))
        else:
            nifti_files.extend(root_path.glob(ext))
    
    return sorted(nifti_files)


def collect_tensor_files(root_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    """폴더에서 Tensor (.pt) 파일 수집
    
    Args:
        root_path: 검색할 루트 폴더
        recursive: 하위 폴더도 검색할지 여부
        
    Returns:
        Tensor 파일 경로 리스트 (메타데이터 json 제외)
    """
    root_path = Path(root_path)
    tensor_files = []
    
    # .pt 확장자 패턴
    extensions = ['*.pt', '*.pth']
    
    for ext in extensions:
        if recursive:
            tensor_files.extend(root_path.rglob(ext))
        else:
            tensor_files.extend(root_path.glob(ext))
    
    # 메타데이터 파일 제외 (_meta.pt 등)
    tensor_files = [f for f in tensor_files if '_meta' not in f.stem]
    
    return sorted(tensor_files)


def get_volume_slice_count(
    file_path: Union[str, Path],
    file_type: str,  # 'nifti' or 'tensor'
    axis: int = 2
) -> int:
    """볼륨 파일의 슬라이스 개수 반환
    
    Args:
        file_path: 파일 경로
        file_type: 'nifti' 또는 'tensor'
        axis: 슬라이스 축
        
    Returns:
        해당 축의 슬라이스 개수
    """
    if file_type == 'nifti':
        volume, _ = load_nifti_volume(file_path)
    elif file_type == 'tensor':
        volume, _ = load_tensor_volume(file_path)
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    
    return volume.shape[axis]


def is_nifti_file(file_path: Union[str, Path]) -> bool:
    """파일이 NIfTI 파일인지 확인"""
    file_path = Path(file_path)
    return file_path.suffix.lower() in ['.nii', '.gz'] and '.nii' in str(file_path).lower()


def is_tensor_file(file_path: Union[str, Path]) -> bool:
    """파일이 Tensor 파일인지 확인"""
    file_path = Path(file_path)
    return file_path.suffix.lower() in ['.pt', '.pth']
