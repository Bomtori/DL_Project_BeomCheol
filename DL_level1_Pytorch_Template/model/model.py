import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        # 첫 번째 컨볼루션 레이어, 1개의 입력 채널, 10개의 출력 채널, 5x5 커널
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 두 번째 컨볼루션 레이어, 10개의 입력 채널, 20개의 출력 채널, 5x5 커널
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 드롭아웃 레이어, 2D 피처맵에 적용
        self.conv2_drop = nn.Dropout2d()
        # 첫 번째 완전 연결(FC) 레이어, 320개의 입력 피처, 50개의 출력 피처
        self.fc1 = nn.Linear(320, 50)
        # 두 번째 완전 연결(FC) 레이어, 50개의 입력 피처, 클래스 수에 해당하는 출력 피처
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # 첫 번째 컨볼루션 레이어와 최대 풀링을 거쳐 활성화 함수 ReLU 적용
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 두 번째 컨볼루션 레이어와 드롭아웃을 거친 후 최대 풀링과 ReLU 활성화 함수 적용
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 플래튼(flatten) 작업을 통해 2D 피처맵을 1D 벡터로 변환
        x = x.view(-1, 320)
        # 첫 번째 완전 연결 레이어와 ReLU 활성화 함수
        x = F.relu(self.fc1(x))
        # 훈련 중 드롭아웃 적용
        x = F.dropout(x, training=self.training)
        # 두 번째 완전 연결 레이어
        x = self.fc2(x)
        # 로그 소프트맥스를 통한 출력
        return F.log_softmax(x, dim=1)
