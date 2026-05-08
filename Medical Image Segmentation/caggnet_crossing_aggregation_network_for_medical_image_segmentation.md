# CAggNet: Crossing Aggregation Network for Medical Image Segmentation

Xu Cao, Yanghao Lin (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분석에서 발생하는 정밀한 시맨틱 세그멘테이션(Semantic Segmentation)의 어려움이다. 의료 영상은 일반 영상에 비해 학습 데이터와 라벨이 부족하며, 해상도가 낮거나 경계선이 모호한(blurred boundary) 특성을 가지고 있어 미세한 세부 정보를 포착하는 것이 매우 어렵다.

기존의 U-Net 구조는 인코더와 디코더 사이의 단순한 선형 스킵 연결(Linear Skip Connection)을 사용하여 저수준 특징 맵과 고수준 특징 맵을 결합한다. 그러나 이러한 방식은 시맨틱 정보의 직접적인 융합만으로는 경계가 모호하거나 형태가 복잡한 객체를 정밀하게 분리하는 데 한계가 있다. 따라서 본 연구의 목표는 다양한 스케일의 특징을 상호작용적으로 융합하여 의료 영상 내 객체의 인식 능력을 높이고, 보다 정확하고 효율적인 세그멘테이션을 수행하는 $\text{CAggNet}$ 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 단순한 스킵 연결을 대체하여 다층적인 다운샘플링 및 업샘플링 레이어를 통한 '교차 집계(Crossing Aggregation)' 구조를 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Crossing Aggregation Module (CAM) 제안**: $\text{U-Net}$ 구조 내에서 정보를 밀집(dense)하고 반복적으로 통합하기 위해, 서로 다른 스케일의 특징 맵을 융합하는 새로운 모듈을 설계하였다.
2. **Weighted Aggregation Module (WAM) 도입**: 네트워크의 최종 단계에서 다중 스케일의 출력 정보를 채널 어텐션(Channel Attention) 메커니즘을 통해 효율적으로 통합하는 가중치 집계 방식을 제안하였다.
3. **Focal Loss의 적용**: 의료 영상 특유의 전경-배경 클래스 불균형(Class Imbalance) 문제를 해결하기 위해 $\text{Focal Loss}$를 도입하여 모델의 강건성을 입증하였다.

## 📎 Related Works

논문에서는 $\text{FCN}$에서 시작하여 $\text{U-Net}$, $\text{ResNets}$로 이어지는 세그멘테이션 연구의 흐름을 설명한다. $\text{U-Net}$은 스킵 연결을 통해 다운샘플링 과정에서 손실된 정보를 복원하며 픽셀 단위의 정밀한 로컬라이제이션을 가능하게 했다. 이후 $\text{Ternaus-Net}$, $\text{W-Net}$, $\text{Res U-net}$, $\text{Dense U-net}$, $\text{Attention U-Net}$ 등 다양한 변형 모델들이 등장하였으나, 이들은 대부분 선형적인 인코더-디코더 백본을 유지한다는 공통점이 있다.

최근에는 인코더와 디코더 사이의 시맨틱 갭(Semantic Gap)을 줄이기 위해 $\text{DLA (Deep Layer Aggregation)}$, $\text{UNet++}$, $\text{UNet3+}$와 같은 중첩 구조(Nested Structure)가 연구되었다. $\text{DLA}$와 $\text{UNet++}$는 $\text{DenseNets}$의 아이디어를 흡수하여 스킵 경로에서 더 풍부한 정보를 집계함으로써 정밀한 특징 추출을 가능하게 한다. $\text{CAggNet}$은 이러한 중첩 구조의 장점을 취하면서도, $\text{UNet++}$가 가진 학습의 어려움(복잡한 훈련 과정 및 가지치기 필요성)을 해결하기 위해 그래프 기반의 풀 스케일 레이어 집계 방식을 채택하여 차별점을 두었다.

## 🛠️ Methodology

$\text{CAggNet}$은 전반적으로 인코더-디코더 구조를 따르며, 핵심은 $\text{CAM}$과 $\text{WAM}$이라는 두 가지 서브 구조의 결합에 있다.

### 1. Crossing Aggregation Module (CAM)

$\text{CAM}$은 그래프 기반의 풀 스케일 레이어 집계 방식으로, 세 가지 서로 다른 스케일의 특징 맵(상위 레벨 $\text{stage } N-1$, 현재 레벨 $\text{stage } N$, 하위 레벨 $\text{stage } N+1$)을 입력으로 받는다.

- **작동 절차**:
    1. $\text{stage } N$ 특징 맵, $\text{stage } N-1$의 다운샘플링 결과, $\text{stage } N+1$의 업샘플링 결과를 결합($\text{Concatenate}$)한다.
    2. 결합된 특징 맵을 두 개의 $3 \times 3$ 합성곱 층과 $\text{ReLU}$ 활성화 함수에 통과시킨다.
    3. 최종 출력물에 입력 단계의 $\text{stage } N-1$ 특징 맵을 더해 잔차 연결(Residual Connection)을 형성한다.

- **수식 설명**:
    결합 단계의 텐서 $Z$는 다음과 같이 정의된다.
    $$Z = \text{Concat}(X_{i,j-1}, DS(X_{i-1,j}), US(X_{i+1,j}))$$
    최종 출력 $X_{i,j}$는 다음과 같이 계산된다.
    $$X_{i,j} = X_{i,j-1} + \sigma_2(W_2 \sigma_1(W_1 Z + b_1) + b_2)$$
    여기서 $DS$와 $US$는 각각 다운샘플링과 업샘플링을, $\sigma$는 $\text{ReLU}$ 활성화 함수를, $W$와 $b$는 합성곱 층의 가중치와 편향을 의미한다.

### 2. Weighted Aggregation Module (WAM)

$\text{WAM}$은 다중 스케일의 출력 정보 중 가치 있는 특징을 강조하여 최종 예측 맵의 복원력을 높인다.

- **작동 절차**:
    1. 전역 평균 풀링($\text{Global Average Pooling}$)을 통해 특징 맵에서 1차원 어텐션 벡터를 생성한다.
    2. 두 개의 $1 \times 1$ 합성곱 층과 $\text{ReLU}$ 및 $\text{Sigmoid}$ 함수를 거쳐 채널 어텐션 가중치를 생성한다.
    3. 입력 특징 맵과 생성된 어텐션 벡터를 요소별 곱(Element-wise product) 연산하여 최적화된 특징 맵을 도출한다.
    4. 이렇게 처리된 각 층의 출력물들은 아래에서 위로 업샘플링 및 결합 과정을 통해 최종적으로 융합된다.

### 3. Focal Loss

의료 영상의 전경-배경 클래스 불균형 문제를 해결하기 위해 $\text{Focal Loss}$를 사용한다. 이는 쉬운 예제(easy negatives)의 비중을 낮추고 어려운 예제(hard positives)에 집중하게 하여 학습의 효율을 높인다.

- **수식**:
    $$\text{FocalLoss}(P_t) = -\alpha_t(1 - P_t)^\gamma \log(P_t)$$
    여기서 $P_t$는 예측 확률이며, $\alpha$는 클래스 균형을 맞추는 하이퍼파라미터, $\gamma$는 손실을 재조정하는 포커싱 파라미터이다. 본 논문에서는 $\gamma=2, \alpha=0.25$ 설정에서 최적의 성능을 보였음을 명시하였다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **CELL**: 2018 Data Science Bowl의 핵 검출 데이터셋 (총 670장, $256 \times 256$ 리사이즈).
  - **GLAND**: MICCAI 2015 Gland Segmentation Challenge 데이터셋 (총 165장, $512 \times 512$ 리사이즈).
- **평가 지표**: $\text{mIoU (mean Intersection-over-Union)}$ 및 $\text{F1-score}$를 사용하였다.
- **비교 대상(Baselines)**: $\text{FCN}$, $\text{U-Net}$, $\text{UNet++}$와 비교하였으며, 모든 베이스라인은 $\text{BCE (Binary Cross Entropy)}$ 손실 함수를 사용하여 학습되었다.

### 정량적 결과

- **성능 비교**: $\text{CAggNet}$은 두 데이터셋 모두에서 모든 베이스라인 모델보다 일관되게 높은 성능을 보였다.
  - **CELL 데이터셋**: $\text{IoU}$ $0.8537$, $\text{F1-score}$ $0.9216$을 달성하여 $\text{UNet++}$($\text{IoU}$ $0.8489$)를 상회하였다.
  - **GLAND 데이터셋**: $\text{IoU}$ $0.7922$, $\text{F1-score}$ $0.8845$를 달성하여 $\text{UNet++}$($\text{IoU}$ $0.7919$)보다 우수하였다.
- **손실 함수 연구**: $\text{GLAND}$ 데이터셋 실험 결과, $\text{BCE Loss}$보다 $\text{Focal Loss}$($\alpha=0.25, \gamma=2$)를 적용했을 때 $\text{IoU}$ $0.8063$, $\text{F1-score}$ $0.8927$로 성능이 크게 향상됨을 확인하였다.
- **최종 결과 (Ablation Study)**: $\text{U-Net}$ 대비 $\text{F1-score}$ 평균 $0.94\%$, $\text{IoU}$ 평균 $1.56\%$의 이득을 얻었으며, $\text{UNet++}$ 대비로도 $\text{F1-score}$ $0.70\%$, $\text{IoU}$ $1.18\%$ 향상된 결과를 보였다.

## 🧠 Insights & Discussion

$\text{CAggNet}$의 강점은 $\text{CAM}$을 통해 다양한 해상도 수준의 특징을 그래프 구조로 조밀하게 연결함으로써, 기존 $\text{U-Net}$의 단순한 결합 방식보다 풍부한 멀티 스케일 특징을 추출할 수 있다는 점이다. 특히 $\text{ResNets}$와 유사한 잔차 연결 구조를 도입하여 네트워크 깊이가 깊어짐에도 불구하고 파라미터 손실을 방지하고 학습의 안정성을 유지한 점이 돋보인다.

또한, 의료 영상의 고질적인 문제인 클래스 불균형을 $\text{Focal Loss}$를 통해 효과적으로 해결하였으며, $\text{WAM}$의 채널 어텐션 메커니즘이 객체의 세부 디테일을 복원하는 데 실질적인 도움을 주었음을 실험적으로 증명하였다.

다만, 본 논문에서 제시한 실험은 2D 영상 데이터셋에 한정되어 있다. 의료 영상 분석의 핵심인 3D CT나 MRI 영상, 혹은 비디오 프레임과 같은 고차원 데이터셋에서의 성능 검증은 이루어지지 않았으며, 이는 향후 연구 과제로 남아 있다.

## 📌 TL;DR

본 논문은 의료 영상의 모호한 경계와 데이터 부족 문제를 해결하기 위해, 다중 스케일 특징을 상호작용적으로 융합하는 **Crossing Aggregation Network (CAggNet)**를 제안한다. 교차 집계 모듈($\text{CAM}$)과 가중치 집계 모듈($\text{WAM}$), 그리고 $\text{Focal Loss}$를 결합하여 $\text{U-Net}$ 및 $\text{UNet++}$보다 정밀한 세그멘테이션 성능을 달성하였다. 이 연구는 정밀한 객체 분리가 필요한 의료 인공지능 진단 시스템의 성능 향상에 기여할 가능성이 높다.
