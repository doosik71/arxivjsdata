# Deep Distance Map Regression Network with Shape-aware Loss for Imbalanced Medical Image Segmentation

Huiyu Li, Xiabi Liu, Said Boumaraf, Xiaopeng Gong, Donghai Liao, and Xiaohong Ma (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석에서 매우 어렵고 중요한 과제인 소형 객체 분할(Small object segmentation), 특히 간 종양 분할(Liver tumor segmentation) 문제를 해결하고자 한다. 간 종양 분할에서 발생하는 주요 문제는 다음과 같다.

첫째, 종양이 전체 입력 볼륨에서 차지하는 비중이 매우 작기 때문에, 방대하고 복잡한 배경 속에서 종양을 찾아내야 한다. 특히 종양의 형태와 위치의 변화가 크고 경계가 불분명하여 분할의 난이도가 높다.
둘째, 심각한 데이터 불균형(Data imbalance) 문제이다. 이를 해결하지 않으면 학습 과정이 다수 클래스(배경)에 편향되어 소수 클래스(종양)를 완전히 무시하는 경향이 발생한다.

결과적으로 본 논문의 목표는 단순한 이진 분할 마스크(Binary segmentation mask)의 한계를 넘어, 객체의 기하학적 특성을 반영할 수 있는 Distance Map을 활용하여 불균형한 의료 영상 데이터에서도 정밀한 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이진 분할 마스크와 Distance Map 사이의 엄격한 매핑(Rigorous mapping) 관계를 이용하여, 분할 문제를 회귀(Regression) 문제로 확장하는 것이다. 주요 기여 사항은 다음과 같다.

1. **LR-Net (Light Weight Regression Network) 제안**: 이진 분할 마스크(또는 확률 맵)를 입력받아 Distance Map을 예측하는 경량 회귀 네트워크를 도입하였다. 이를 통해 네트워크가 단순한 분류를 넘어 객체의 형태와 경계 정보를 학습하도록 유도한다.
2. **Shape-aware Loss (MapDice loss) 제안**: Distance Map의 풍부한 정보를 Dice loss에 통합한 MapDice loss를 제안하였다. 이 손실 함수는 경계 지역에 더 높은 가중치를 부여하여, 특히 소형 객체의 전체적인 형태를 더 정확하게 추론하도록 강제한다.
3. **효율적인 학습 파이프라인**: LR-Net을 M-Net(Main Segmentation Network)과 분리하여 먼저 학습시킨 후, 전체 네트워크 학습 시에는 LR-Net의 파라미터를 고정(Freeze)하는 전략을 사용하여 학습의 안정성을 높였다.

## 📎 Related Works

기존의 소형 객체 분할 및 데이터 불균형 해결을 위한 접근 방식은 크게 두 가지로 나뉜다.

1. **Cascaded Training**: 이전 네트워크의 출력을 다음 네트워크의 입력으로 사용하여 불필요한 배경 정보를 제거하는 방식이다. 하지만 계산 비용이 많이 들며, 앞 단계의 네트워크에서 발생한 오분할 결과가 후속 단계로 전이되어 성능을 저하시킬 위험이 있다.
2. **Class Re-weighting**: 레이블 빈도에 반비례하는 가중치를 손실 함수에 적용하거나, Tversky index, Focal loss 기반의 Dice/Tversky loss 등을 통해 정밀도(Precision)와 재현율(Recall)의 균형을 맞추는 방식이다. 이러한 방법들은 데이터 불균형에는 효과적이지만, 객체의 기하학적 특성이나 전체적인 형태(Overall shape)에 대한 인식이 부족하다는 한계가 있다.

본 연구는 Distance Map을 직접 예측하거나 단순히 Ground-truth로 사용하는 기존 방식과 달리, 분할 마스크로부터 Distance Map을 생성하는 매핑 과정을 네트워크(LR-Net)가 학습하게 함으로써 기하학적 제약 조건을 더 강력하게 부여한다.

## 🛠️ Methodology

### 1. Norm Inverse Distance Map (NI-DM)

단순한 이진 마스크 대신, 객체의 형태와 경계 정보를 포함하는 Distance Map을 Ground-truth로 사용한다. 본 논문에서는 경계 근처의 픽셀에 더 많은 가중치를 부여하고 수치적 안정성을 위해 정규화한 **NI-DM**을 제안한다.

먼저, 이미지 공간의 한 점 $x$에서 가장 가까운 경계 $b$까지의 유클리드 거리인 Original Distance Map (O-DM) $D(x)$를 계산한다.
$$D(x) = \min_{b \in B} d(x, b)$$
여기서 $B$는 객체 경계에 있는 점들의 집합이며, $d(x, b)$는 유클리드 거리이다. 이후, 각 연결 구성 요소 $C$에 대해 다음과 같이 정규화된 역 거리 맵(NI-DM) $\phi(x)$를 정의한다.
$$\phi(x) = \frac{\max_{x \in C} (D(x)) + 1 - D(x)}{\max_{x \in C} (D(x))}$$
이 연산을 통해 거리 값은 $[0, 1]$ 범위로 정규화되며, 경계에 가까울수록 높은 값을 가지게 된다.

### 2. Network Architecture

시스템은 **M-Net (Main Segmentation Network)**과 **LR-Net (Light Weight Regression Network)**이 직렬로 연결된 구조이다.

- **M-Net**: 3D UNet 기반의 구조에 Residual connection을 추가한 형태로, 최종 출력층에서 Softmax를 통해 클래스별 확률 맵(Probability map)을 생성한다.
- **LR-Net**: M-Net의 확률 맵을 입력받아 Distance Map을 예측하는 경량 UNet이다. 인코더는 한 번의 다운샘플링만 수행하며, 디코더는 한 번의 업샘플링을 통해 원래 크기로 복원한다. 출력층에는 ReLU 활성화 함수를 사용하여 NI-DM의 특성을 반영한다.

### 3. Training and Inference Pipeline

학습 과정은 매우 전략적으로 설계되었다.

1. **LR-Net 단독 학습**: 먼저 Ground-truth 분할 마스크를 입력으로 하여 LR-Net을 독립적으로 학습시킨다. 이는 M-Net의 초기 불안정한 예측값이 LR-Net에 영향을 주어 전체 네트워크가 망가지는 것을 방지하기 위함이다.
2. **전체 네트워크 학습**: M-Net과 LR-Net을 연결한 후, **LR-Net의 파라미터를 고정(Freeze)**한 상태에서 M-Net을 학습시킨다. 이를 통해 M-Net은 LR-Net이 이미 학습한 '마스크 $\to$ Distance Map'이라는 엄격한 기하학적 제약 조건을 따르도록 강제된다.
3. **추론(Inference)**: 추론 단계에서는 LR-Net이 필요 없으며, M-Net이 직접 분할 마스크를 예측한다.

### 4. Loss Functions

- **Regression Loss**: LR-Net의 학습을 위해 예측된 Distance Map과 실제 NI-DM 사이의 차이를 측정하는 **Smooth L1 loss**를 사용한다. 이는 이상치에 강건하며 0 근처에서 미분 가능하다.
- **MapDice Loss**: M-Net이 거리 맵의 공간적 표현력을 활용하도록 하기 위해 제안된 새로운 손실 함수이다. 실제 Distance Map $\phi$를 픽셀 단위의 페널티 맵으로 사용하여, 다음과 같이 정의된다.
$$L_{MapDice} = 1 - \sum_{c=1}^{C} \frac{2 \times (p_c \times \phi_c) + \epsilon}{p_c + \phi_c + \epsilon}$$
여기서 $p_c$는 클래스 $c$에 대한 예측 확률 맵이고, $\phi_c$는 해당 클래스의 Ground-truth Distance Map이다. $\epsilon$은 분모가 0이 되는 것을 방지하는 작은 값이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MICCAI 2017 LiTS 챌린지 데이터셋(훈련 130, 테스트 70)과 자체 수집한 임상 데이터셋(137 케이스)을 사용하였다.
- **평가 지표**: Dice per Case (DC), Dice Global (DG), VOE, RVD, ASSD, MSD, RMSD 등 7가지 지표를 사용하였다.
- **구현 세부사항**: PyTorch 프레임워크, Adam 옵티마이저, NVIDIA 2080Ti GPU를 사용하였으며, Kaiming uniform 초기화를 적용하였다.

### 2. 주요 결과

- **Distance Map 종류별 비교**: O-DM, I-DM, NI-DM, SNI-DM을 비교한 결과, **NI-DM과 ReLU 활성화 함수**의 조합이 가장 높은 성능을 보였다.
- **구성 요소의 효과**:
  - M-Net 단독 사용보다 LR-Net을 추가했을 때 모든 지표에서 성능이 향상되었다.
  - Dice loss보다 MapDice loss를 사용했을 때 특히 소형 종양 분할의 정확도가 크게 향상되었다.
  - LR-Net을 미리 학습시켜 고정하는 전략이 LR-Net을 함께 학습시키는(unfrozen) 방식보다 훨씬 우수한 성능을 보였다.
- **SOTA 비교**: 임상 데이터셋 실험에서 WCE, GDS, Tversky, Focal Tversky, Exp-Log loss 등 기존의 불균형 해결 방법들보다 월등한 성능을 기록하였다. 특히 DC(Dice per Case) 지표에서 유의미한 향상을 보였다.
- **LiTS 리더보드**: LiTS 챌린지 결과 DC per case 0.679, DG 0.830 등의 경쟁력 있는 수치를 달성하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 단순히 '어떤 클래스인가'를 맞추는 분류 문제에서 벗어나, '경계로부터 얼마나 떨어져 있는가'라는 거리 정보를 회귀 문제로 풀어냄으로써 분할 성능을 높였다.

**강점 및 통찰**:

- **기하학적 제약의 도입**: Distance Map은 이진 마스크보다 훨씬 풍부한 공간 정보를 담고 있다. 이를 통해 네트워크가 객체의 전체적인 형태(Global shape)를 인식하게 함으로써, 소형 객체 분할 시 발생하는 파편화된 예측(Fragmented prediction) 문제를 효과적으로 억제하였다.
- **학습 전략의 영리함**: LR-Net을 먼저 학습시키고 고정하는 방식은, 정답 마스크와 거리 맵 사이의 '정답지'를 먼저 만든 뒤 M-Net이 그 정답지에 맞게 예측하도록 가이드하는 효과를 준다.

**한계 및 논의사항**:

- **추론 효율성**: 추론 시에는 LR-Net을 제거하므로 추가 연산 비용이 없다는 점이 장점이나, 학습 단계에서는 Distance Map을 생성하고 별도의 네트워크를 학습시켜야 하므로 학습 시간이 증가한다.
- **거리 맵의 정의**: 본 논문에서는 NI-DM이 최적이라고 결론지었으나, 다른 형태의 거리 변환(Distance Transform) 방식이 특정 장기나 다른 의료 영상 도메인에서 더 효과적일 가능성이 남아 있다.

## 📌 TL;DR

본 연구는 의료 영상의 소형 객체 분할 시 발생하는 데이터 불균형과 경계 모호성 문제를 해결하기 위해, **Distance Map 회귀 네트워크(LR-Net)**와 **형태 인식 손실 함수(MapDice loss)**를 제안하였다. 분할 마스크와 거리 맵 사이의 엄격한 매핑 관계를 학습시킨 후 이를 M-Net의 가이드로 활용함으로써, 기존의 가중치 기반 손실 함수들보다 정밀한 형태 추론이 가능함을 입증하였다. 이 방법론은 향후 정밀한 기하학적 정보가 필요한 다양한 의료 영상 분할 작업에 광범위하게 적용될 가능성이 높다.
