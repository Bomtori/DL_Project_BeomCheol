"""
모든 트레이너 클래스의 기본 클래스로 사용됩니다.
주요 기능은 모델 훈련, 성능 모니터링, 체크포인트 저장 및 불러오기입니다.
"""

import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    모든 트레이너의 기본 클래스
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config    # 설정 정보
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])  # 로깅 설정

        self.model = model              # 모델 객체
        self.criterion = criterion      # 손실 함수
        self.metric_ftns = metric_ftns  # 평가 메트릭 함수 목록
        self.optimizer = optimizer      # 최적화 알고리즘

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']                 # 전체 훈련 에폭 수
        self.save_period = cfg_trainer['save_period']       # 체크포인트 저장 주기
        self.monitor = cfg_trainer.get('monitor', 'off')    # 모델 성능 모니터링 설정

        # 모델 성능 모니터링 및 최고 성능 모델 저장 설정
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1    #시작 에폭 설정

        self.checkpoint_dir = config.save_dir   # 체크포인트 저장 디렉토리

        # 시각화 도구 설정               
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)  # 체크포인트에서 훈련 재개

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        한 에폭 동안 훈련 로직

        :param epoch: 현재 에폭 번호
        """
        raise NotImplementedError

    def train(self):
        """
        전체 훈련 로직
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # 로그 정보 저장
            log = {'epoch': epoch}
            log.update(result)

            # 로그 정보 화면 출력
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 모델 성능 평가 및 최고 성능 모델 저장
            best = False
            if self.mnt_mode != 'off':
                try:
                    # 모델 성능 향상 여부 체크
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        체크포인트 저장

        :param epoch: 현재 에폭 번호
        :param log: 에폭마다 로그 정보
        :param save_best: True일 경우, 저장된 체크포인트를 'model_best.pth'로 이름 변경
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        저장된 체크포인트에서 훈련 재개

        :param resume_path: 재개할 체크포인트 경로
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # 체크포인트의 아키텍처 파라미터 로드
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # 체크포인트의 옵티마이저 상태 로드
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

"""
1. 로깅
    - 프로그램이 실행되는 동안 발생하는 event나 데이터를 시스템의 로그 파일이나
    다른 출력 형태로 기록하는 프로세스를 말합니다. 이 기록들은 디버깅, 소프트웨어의 상태 모니터링,
    행위 분석, 오류 추적 등 여러 목적으로 사용됩니다. 특히 복잡한 시스템이나 오랜 시간 동안 운영되는
    시스템에서 로깅은 중요한 정보를 제공하며, 문제가 발생했을 때 그 원인을 찾는 데 큰 도움을 줍니다.

2. 체크포인트의 아키텍처 파라미터 로드
    - 저장된 체크포인트 파일에서 모델의 아키텍처에 관련된 파라미터들을 불러와 현재 모델에
    적용하는 과정을 의미합니다. 예를 들어, 모델의 구조(레이어의 수, 각 레이어의 유형 등)에
    관련된 설정을 포함합니다. 이 과정을 통해 사용자는 이전에 저장한 모델의 상태를 정확히
    복원할 수 있으며, 훈련을 중단했던 지점부터 다시 시작할 수 있습니다. 만약 체크포인트와
    현재 설정이 다르면, 호환성 문제가 발생할 수 있기 때문에 이러한 경고 메세지를 로깅을 통해
    사용자에게 알립니다.

3. 옵티마이저
    - 옵티마이저는 기계 학습에서 모델의 학습 과정을 관리하는 알고리즘으로, 모델의
    가중치들을 업데이트하는 방법을 결정합니다. 이 과정은 모델이 학습 데이터로부터
    오차를 최소화하는 방향으로 진행됩니다. 대표적인 옵티마이저로는 SGD, Adam, RMSprop 등이
    있으며, 각각의 특성과 용도에 맞게 사용됩니다. 옵티마이저는 모델의 성능을 최적화하고,
    보다 빠르고 효율적으로 학습을 수행할 수 있도록 돕습니다.
"""