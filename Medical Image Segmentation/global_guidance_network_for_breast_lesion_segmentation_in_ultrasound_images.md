# Global Guidance Network for Breast Lesion Segmentation in Ultrasound Images

Cheng Xue, Lei Zhu, Huazhu Fu, Xiaowei Hu, Xiaomeng Li, Hai Zhang, Pheng-Ann Heng (2021)

## 🧩 Problem to Solve

본 연구는 초음파 영상 내 유방 병변(breast lesion)을 자동으로 분할(segmentation)하는 문제를 해결하고자 한다. 유방암의 조기 진단을 위해 정확한 병변 분할은 매우 중요하지만, 초음파 영상의 특성상 다음과 같은 기술적 난제들이 존재한다.

첫째, 초음파 영상 특유의 스펙클 노이즈(speckle artifacts)와 강한 그림자(shadows)가 존재한다. 둘째, 병변 내부의 강도 분포가 불균일(inhomogeneous intensity distributions)하며, 병변과 배경 사이의 경계가 모호하여 구분이 어렵다. 셋째, 병변의 형태가 매우 불규칙적이다.

기존의 Convolutional Neural Networks (CNNs)는 주로 국소 영역(local regions)에 집중하는 연산 특성을 가지므로, 영상 전체의 장거리 의존성(long-range dependencies)을 포착하는 능력이 부족하다. 이는 결과적으로 유방 병변 분할의 정확도를 저하시키는 원인이 된다. 따라서 본 논문의 목표는 전역적인 문맥 정보(global contextual information)를 효과적으로 학습하고 정밀한 경계선을 복원할 수 있는 딥러닝 네트워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전역 가이드 블록(Global Guidance Block, GGB)과 병변 경계 검출(Boundary Detection, BD) 모듈을 통해 CNN의 국소적 한계를 극복하는 것이다.

1. **Global Guidance Block (GGB) 도입**: 네트워크의 다양한 계층에서 추출된 특징들을 통합한 Multi-layer Integrated Feature (MLIF)를 가이드 정보로 활용한다. 이를 통해 공간(spatial) 및 채널(channel) 도메인 모두에서 장거리 비국소적 의존성(non-local dependencies)을 학습하여 전역적인 문맥 정보를 확보한다.
2. **Boundary Detection (BD) 모듈 설계**: 얕은 CNN 계층에 경계 검출 모듈을 추가하여 병변의 외곽선 맵(boundary map)을 명시적으로 학습하게 함으로써, 최종 분할 결과의 경계 품질을 향상시킨다.
3. **GG-Net 아키텍처 제안**: 위 두 모듈을 통합한 GG-Net을 통해 기존의 의료 영상 분할 방법론 및 최신 시맨틱 분할 모델보다 우수한 성능을 입증하였다.

## 📎 Related Works

기존의 유방 병변 분할 연구는 크게 네 가지 접근 방식으로 분류된다.

- **영역 기반(Region based), 변형 가능 모델(Deformable models), 그래프 기반(Graph-based), 학습 기반(Learning based) 방식**: 이러한 초기 방법론들은 주로 수작업으로 설계된 특징(hand-crafted features)에 의존하였으나, 복잡한 영상 환경에서 특징 표현 능력이 부족하여 오인식률이 높다는 한계가 있었다.
- **CNN 기반 방식**: U-Net, FCN-AlexNet 등이 도입되며 성능이 크게 향상되었다. 특히 Dilated convolution이나 Pooling 연산을 통해 수용 영역(receptive field)을 넓히거나, 중간 단계와 고수준 특징을 융합하는 방식(예: U-Net)이 사용되었다.
- **Non-local/Attention 기반 방식**: 최근에는 Non-local blocks나 Attention mechanism을 통해 전역 문맥을 포착하려는 시도가 있었다. 그러나 대부분의 기존 연구는 이러한 블록을 깊은 CNN 계층에만 배치하였다. 깊은 층은 수용 영역이 너무 커서 병변의 세부적인 부분이나 경계 디테일을 손실하는 경향이 있으며, 채널 간의 상호 의존성을 충분히 고려하지 못한다는 한계가 있다.

본 논문은 얕은 층부터 깊은 층까지의 모든 특징을 통합하여 가이드로 사용하는 GGB와 명시적인 경계 학습을 수행하는 BD 모듈을 통해 기존 방식들과 차별화를 꾀한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

GG-Net은 입력 초음파 영상을 받아 엔드-투-엔드(end-to-end) 방식으로 분할 마스크를 생성한다. 기본 구조는 CNN을 통해 다중 스케일 특징 맵을 생성하고, Atrous Spatial Pyramid Pooling (ASPP) 모듈을 사용하여 수용 영역을 확장하는 것으로 시작한다. 이후 GGB와 BD 모듈이 특징을 정제하고 경계를 보완하며, 최종적으로 분할 맵을 예측한다.

### 2. Global Guidance Block (GGB)

GGB는 공간-wise GGB와 채널-wise GGB로 구성된다. 핵심은 1~4번째 CNN 계층의 특징 맵을 동일한 크기로 리사이징한 후 결합하여 생성한 **Multi-layer Integrated Feature (MLIF)** 맵을 가이드 $G$로 사용하는 것이다.

#### (1) Spatial-wise GGB

입력 특징 맵 $X \in \mathbb{R}^{h \times w \times c}$와 가이드 맵 $G \in \mathbb{R}^{h \times w \times c}$가 주어졌을 때, 다음과 같은 절차를 거친다.

1. $X$를 세 개의 $1 \times 1$ 컨볼루션 층에 통과시켜 $\theta(x), \phi(x), \mu(x)$를 생성한다.
2. $\theta(x)$와 $\phi(x)$를 이용하여 공간적 위치 유사도 맵 $S^x$를 계산한다.
$$S^x = \text{Softmax}(X^T W_\theta^T W_\phi X)$$
3. 마찬가지로 가이드 맵 $G$로부터 위치 유사도 행렬 $S^g$를 계산한다.
$$S^g = \text{Softmax}(G^T W_\rho^T W_\eta G)$$
4. 두 유사도 행렬의 요소별 곱(element-wise multiplication)에 Softmax를 적용하여 가이드된 유사도 행렬 $S^M$을 만들고, 이를 $\mu(x)$에 곱해 최종 특징 $Y$를 산출한다.
$$Y = \mu(x) \text{Softmax}(S^x \cdot S^g) + X$$

#### (2) Channel-wise GGB

공간적 학습에서 무시된 채널 간의 상관관계를 학습한다.

1. 입력 $Y$로부터 채널 유사도 맵 $S^Z \in \mathbb{R}^{c \times c}$를 계산한다.
2. 가이드 맵 $G$에 Squeeze-and-Excitation (SE) 블록을 적용하여 정보량이 많은 채널을 강조한 $\hat{G}$를 생성한다. 이때 채널 통계량 $\beta$는 다음과 같이 계산된다.
$$\beta_k = \frac{1}{h \times w} \sum_{i=1}^{h} \sum_{j=1}^{w} G(i,j,k)$$
3. $\hat{G}$로부터 채널 유사도 맵 $S^{\hat{G}}$를 생성하고, $S^Z$와 $S^{\hat{G}}$를 결합하여 가이드된 채널 유사도 맵 $S^Q$를 도출한다.
4. $S^Q$를 입력 $Y$에 곱하여 최종 정제된 특징 $Z$를 얻는다.

### 3. Breast Lesion Boundary Detection (BD) Module

경계 품질을 높이기 위해 얕은 CNN 계층에 BD 모듈을 배치한다.

- **동작 원리**: 특징 맵 $F^{(i)}$에 $1 \times 1$ 컨볼루션을 적용해 1채널 맵 $F^\phi(i)$를 만든 뒤, Maxpooling 연산을 통해 1픽셀 시프트(shift)시킨 결과와 원래 맵의 차이를 계산하여 경계 맵 $E$를 생성한다.
- **손실 함수**: 전체 손실 함수 $L_{total}$은 각 계층의 분할 손실($L_{seg}^i$), 경계 손실($L_{boundary}^i$), 그리고 최종 출력의 분할 손실($L_{f\_seg}$)의 합으로 정의된다.
$$L_{total} = \sum_{i=1}^{N_{layer}} (\lambda_1 L_{seg}^i + \lambda_2 L_{boundary}^i) + L_{f\_seg}$$
여기서 $L_{seg}$는 Dice loss와 Binary Cross-Entropy (BCE) loss의 결합이며, $L_{boundary}$는 예측 경계 맵 $D^i$와 실제 경계(Canny operator로 추출) 사이의 평균 제곱 오차(MSE)를 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 공개 데이터셋인 BUSI(780장)와 자체 수집한 데이터셋(632장)을 사용하였다.
- **평가 지표**: Dice coefficient, Jaccard index, Recall, Precision, Accuracy, Hausdorff distance (HD), Average boundary distance (ABD) 등 7가지 지표를 사용하였다.
- **비교 대상**: U-Net, U-Net++, FPN, DeepLabv3+, AG-Unet, DAF 등 최신 모델들과 비교하였다.

### 2. 주요 결과

- **자체 수집 데이터셋**: GG-Net은 모든 지표에서 SOTA(State-of-the-art) 성능을 달성하였다. 특히 Dice 성능이 타 모델 대비 월등히 높았으며, HD와 ABD 값은 낮게 유지되어 경계 복원 능력이 뛰어남을 입증하였다.
- **BUSI 데이터셋**: 공개 데이터셋에서도 타 모델보다 우수한 성능을 보였으며, 특히 병변이 없는 'Normal' 케이스를 포함하여 학습시킨 경우에도 가장 강건한 성능을 유지하였다.
- **전립선 분할 응용**: 본 네트워크를 전립선 초음파 영상 분할 작업에 적용한 결과, U-Net 및 DAF 등 기존 모델보다 높은 Jaccard 및 Dice 점수를 기록하여 범용적인 유효성을 입증하였다.

### 3. 절제 연구 (Ablation Study)

- GGB가 없는 기본 모델보다 GGB가 추가되었을 때 성능이 향상되었으며, 공간-wise와 채널-wise GGB를 모두 사용할 때 최적의 성능이 나타났다.
- BD 모듈의 경계 지도 학습을 제거했을 때($L_{boundary}$ 제거) 분할 정확도가 하락하는 것을 확인하여, 명시적인 경계 학습의 중요성을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

본 연구는 CNN의 고질적인 문제인 '국소적 시야'를 전역 가이드 맵(MLIF)을 통한 Non-local 학습으로 해결하였다. 특히 단순히 깊은 층에 Non-local 블록을 넣는 것이 아니라, 얕은 층의 디테일한 정보와 깊은 층의 추상적 정보를 통합하여 가이드로 사용함으로써, 전역 문맥을 잡으면서도 세부 경계를 놓치지 않는 구조를 설계한 점이 돋보인다.

### 2. 한계점 및 비판적 해석

- **실패 사례**: 병변의 크기가 매우 크거나 내부 강도 분포가 극도로 복잡한 경우, 또는 경계가 지나치게 모호한 경우에는 여전히 오분류(False Positive)가 발생하거나 일부 영역을 누락하는 경향이 있다.
- **데이터 의존성**: 자체 수집 데이터셋의 성능이 BUSI 데이터셋보다 높게 나타났는데, 이는 저자들이 언급했듯 데이터 품질의 차이에서 기인한 것으로 보인다. 실제 임상 환경의 매우 낮은 품질 영상에서도 동일한 성능이 유지될지는 추가 검증이 필요하다.
- **계산 복잡도**: Non-local 연산과 다중 계층 특징 통합으로 인해 파라미터 수가 55M에 달하며, 이는 실시간 진단 시스템에 적용할 때 최적화 이슈가 될 수 있다.

## 📌 TL;DR

본 논문은 초음파 영상의 유방 병변 분할을 위해 **전역 가이드 블록(GGB)**과 **경계 검출(BD) 모듈**을 탑재한 **GG-Net**을 제안한다. MLIF를 활용해 공간 및 채널 도메인의 장거리 의존성을 학습하고, 명시적인 경계 지도 학습을 통해 분할 정밀도를 높였다. 실험 결과, 유방 병변 및 전립선 분할 작업에서 기존 SOTA 모델들을 능가하는 성능을 보였으며, 이는 향후 초음파 기반 의료 영상 분석의 정확도를 높이는 데 중요한 기여를 할 것으로 보인다.
