"""
PyTorch의 DataLoader 클래스를 상속 받아 사용자 정의 데이터 로더의 기본 클래스를 제공합니다.
데이터를 미니배치로 쪼개어 모델 학습에 제공하는 기능을 수행하며, 유효성 검사 데이터셋을 분할하는
기능도 포함합니다.
"""

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    모든 데이터 로더의 기본 클래스
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split    # 검증 세트의 분할 비율
        self.shuffle = shuffle                      # 데이터 셔플 여부

        self.batch_idx = 0
        self.n_samples = len(dataset)               # 데이터셋의 전체 샘플 수

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)   # 데이터셋을 훈련/검증 샘플러로 분할

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)  # DataLoader 초기화

    def _split_sampler(self, split):
        # 검증 세트 분할 로직
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)               #난수 생성 시드 설정
        np.random.shuffle(idx_full)     # 데이터 인덱스 셔플

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)  # 훈련 데이터 샘플러
        valid_sampler = SubsetRandomSampler(valid_idx)  # 검증 데이터 샘플러

        self.shuffle = False    # 샘플러 사용 시 셔플 불가
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        # 검증 DataLoader 생성
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

"""
1. 데이터 로더의 역할
    - 데이터 분할 : 입력받은 데이터셋을 훈련 세트와 검증 세트로 분할합니다.
    - 셔플 : 데이터의 순서를 무작위로 섞는 옵션
    - 샘플러 : 특정 규칙에 따라 데이터셋에서 샘플을 선택합니다.

2. 샘플러 설명
    - 주어진 인덱스 리스트에서 무작위로 샘플을 선택하여 데이터 로더에 공급합니다.
    이를 통해 훈련과 검증 과정에서 데이터의 다양성을 유지할 수 있으며, 모델이
    데이터의 특정 순서에 의존하지 않도록 합니다.

3. 샘플러 사용 시 셔플 불가능 이유
    - 샘플러와 셔플 옵션은 상호 배타적입니다. 샘플링을 사용할 때는 이미 데이터 인덱스를
    무작위로 섞어 샘플링하기 때문에, 중복으로 셔플이 일어나 데이터 로딩 과정이 비효율적이거나
    예상치 못한 방식으로 작동할 수 있습니다. 따라서, 샘플러를 사용할 때는 일반적으로 'shuffle'
    옵션을 'False'로 설정하여 샘플러가 정의한 순서대로 데이터를 로딩하도록 합니다.

4. 셔플, 샘플러 차이점
    - 셔플은 단순히 데이터의 순서를 랜덤화하는 반면, 샘플러는 데이터셋에서 데이터를 선택하는 
    규칙을 설정할 수 있습니다. 즉, 샘플러는 데이터셋의 특정 부분을 타깃으로 할 수 있는 반면, 
    셔플은 전체 데이터셋의 순서만을 변경합니다.
"""