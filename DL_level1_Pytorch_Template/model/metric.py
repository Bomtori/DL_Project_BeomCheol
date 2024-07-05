import torch


def accuracy(output, target):
    """
    정확도를 계산하는 함수

    :param output: 모델의 예측 로짓 또는 확률 (배치 크기 x 클래스 수)
    :param target: 실제 타겟 레이블 (배치 크기)
    :return: 계산된 정확도 값
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)          # 각 입력 샘플에 대해 가장 높은 확률을 가진 클래스의 인덱스를 예측
        assert pred.shape[0] == len(target)         # 예측된 결과와 타겟 레이블의 길이가 같은지 확인
        correct = 0
        correct += torch.sum(pred == target).item() # 예측이 맞은 샘플 수 계산
    return correct / len(target)                    # 정확도 계산


def top_k_acc(output, target, k=3):
    """
    top-k 정확도를 계산하는 함수

    :param output: 모델의 예측 로짓 또는 확률 (배치 크기 x 클래스 수)
    :param target: 실제 타겟 레이블 (배치 크기)
    :param k: 상위 k개의 예측을 고려할 때
    :return: 계산된 top-k 정확도 값
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]                  # 각 입력 샘플에 대해 가장 높은 확률을 가진 상위 k개 클래스의 인덱스를 예측
        assert pred.shape[0] == len(target)                     # 예측된 결과와 타겟 레이블의 길이가 같은지 확인
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()   # 상위 k개 예측 중 정답이 있는 샘플 수 계산
    return correct / len(target)                                # top-k 정확도 계산
