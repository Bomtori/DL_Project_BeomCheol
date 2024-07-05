"""
    'forward' 메서드는 추상 메서드로 정의되어 있어, 이 클래스를 상속받는
    모든 서브 클래스는 자신만의 'forward' 메서드를 구현해야 합니다.
    이 메서드는 모델이 입력을 받아 출력을 생성하는 로직을 담당합니다.
"""

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    모든 모델의 기본 클래스
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        포워드 패스 로직

        :return: 모델 출력
        """
        raise NotImplementedError   # 구현되지 않은 경우 예외 발생

    def __str__(self):
        """
        모델을 문자열로 출력할 때 학습 가능한 파라미터의 수를 함께 출력
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) # 학습 가능한 파라미터 필터링
        params = sum([np.prod(p.size()) for p in model_parameters])             # 파라미터의 전체 개수 계산
        return super().__str__() + '\nTrainable parameters: {}'.format(params)  # 기본 문자열 표현에 학습 가능한 파라미터 수 추가하여 반환


"""
1. 파라미터
    - 모델이 학습 과정에서 데이터로부터 배우는 가변적인 수치를 의미합니다.
    이 파라미터들은 모델의 성능에 직접적인 영향을 미치며,
    예를 들어 신경망에서는 가중치와 편향이 파라미터에 해당합니다.

    모델이 학습을 통해 이러한 파라미터를 조정함으로써, 주어진 입력 데이터에
    대한 정확한 출력을 예측하려고 시도합니다. 파라미터의 개수는 모델의 복잡성과
    크기를 나타내는 지표 중 하나로 사용될 수 있으며, 일반적으로 파라미터 수가
    많을수록 모델은 더 많은 정보를 학습할 수 있지만, 과적합의 위험도 증가할 수 있습니다.
"""