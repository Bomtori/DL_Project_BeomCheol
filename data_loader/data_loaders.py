from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    BaseDataLoader를 사용한 MNIST 데이터 로딩 데모
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 데이터 변환 시퀀스 정의 : 이미지를 텐서로 변환 후, 정규화 수행
        trsfm = transforms.Compose([
            transforms.ToTensor(),                      # 이미지를 pyTorch 텐서로 변환
            transforms.Normalize((0.1307,), (0.3081,))  # 평균 0.1307, 표준편차 0.3081로 정규화
        ])
        self.data_dir = data_dir
        # MNIST 데이터셋 로드 : 훈련 / 테스트 설정, 자동 다운로드, 변환 적용
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        # 부모 클래스(base_data_loader) 생성자 호출
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

"""
1. 평균 0.1307, 표준편차 0.3081로 정규화 하는 이유
    - 데이터의 스케일을 조정하고 모델 학습을 더 효율적으로 만들기 위함
    MNIST 데이터셋의 이미지들에서 계산된 값

2. 정규화 목적
    - 데이터 스케일 조정 : 데이터셋의 픽셀 값 범위를 [0, 255]에서 [0, 1]로 변환한 후,
    추가적으로 각 픽셀 값에서 데이터셋의 평균을 빼고 표준편차로 나누어 줌으로써
    데이터 분포를 조정합니다.

    - 학습의 안정성과 속도 향상 : 데이터의 평균을 0 주변으로, 표준편차를 1로 조정함으로써,
    경사 하강법을 사용할 때 수렴 속도를 향상시킬 수 있습니다. 데이터가 표준화되면, 각 차원이
    비슷한 범위를 가지게 되어, 최적화 과정에서 파라미터가 더 빠르고 안정적으로 업데이트 됩니다.
"""
