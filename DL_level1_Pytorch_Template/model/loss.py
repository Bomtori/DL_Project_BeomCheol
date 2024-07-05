import torch.nn.functional as F


def nll_loss(output, target):
    """
    Negative Log Likelihood 손실을 계산하는 함수

    :param output: 모델의 예측 결과 (로짓 또는 로그 확률)
    :param target: 실제 타겟 레이블
    :return: 계산된 손실 값
    """
    return F.nll_loss(output, target)
