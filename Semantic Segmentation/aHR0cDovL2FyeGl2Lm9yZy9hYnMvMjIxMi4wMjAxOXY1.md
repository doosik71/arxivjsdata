# SASFORMER: TRANSFORMERS FOR SPARSELY ANNOTATED SEMANTIC SEGMENTATION

Hui Su, Yue Ye, Wei Hua, Lechao Cheng, Mingli Song (2023)

## 🧩 Problem to Solve

시맨틱 세그멘테이션(Semantic Segmentation)은 이미지의 모든 픽셀을 식별해야 하는 작업으로, 이를 위해 모든 픽셀에 대해 정밀한 라벨을 생성하는 pixel-by-pixel annotation 작업에는 막대한 시간과 비용이 소요된다. 이를 해결하기 위해 객체의 일부에만 라벨을 지정하는 Sparsely Annotated Semantic Segmentation (SASS) 연구가 진행되어 왔으며, 여기에는 point-wise 및 scribble-wise supervision 등이 포함된다.

SASS의 핵심 문제는 정밀한 객체 경계 정보가 부족하기 때문에, 라벨이 지정되지 않은 나머지 픽셀들의 클래스를 정확하게 추정하는 것이 매우 어렵다는 점이다. 기존의 접근 방식들은 이를 해결하기 위해 복잡한 다단계 학습 전략(multi-stage training strategy)이나 보조 네트워크(auxiliary networks)를 사용하였으며, 이는 학습 과정을 복잡하게 만들고 많은 시간을 소요하게 하는 한계가 있다. 본 논문의 목표는 Vision Transformer의 내재적인 전역 의존성(global dependencies)을 활용하여, 복잡한 프레임워크 없이 단순하면서도 효율적인 단일 단계(single-stage) SASS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 self-attention 메커니즘이 픽셀 간의 효과적인 관계를 캡처할 수 있다는 점에 착안하여, 이를 통해 라벨이 있는 픽셀의 정보를 라벨이 없는 픽셀로 전파(propagation)하는 것이다.

구체적으로, 계층적 패치 어텐션 맵(hierarchical patch attention maps)을 생성하고 이를 네트워크 예측값과 결합하여 라벨 간의 상관관계가 있는 영역을 도출한다. 또한, 상관관계 결과의 특징과 네트워크 예측값 사이의 일관성을 보장하기 위한 새로운 Affinity Loss를 도입하여, 동일한 객체에 속하는 영역들이 저차원 및 고차원 특징 공간 모두에서 유사성을 갖도록 강제한다.

## 📎 Related Works

### 1. Sparsely Annotated Semantic Segmentation (SASS)

기존의 SASS 연구들은 최소한의 라벨링으로 성능을 높이기 위해 다양한 방법을 시도하였다.

- **정규화 손실(Regularization losses):** Dense CRF나 kernel cut을 이용해 저차원 정보의 클러스터링을 유도한다.
- **다중 작업 보조(Multitask auxiliary):** 경계 검출(boundary detection)과 같은 보조 작업을 추가하여 성능을 보완한다.
- **일관성 학습(Consistency learning) 및 의사 라벨(Pseudo-label):** 여러 네트워크를 사용하거나 다단계 학습 과정을 통해 점진적으로 추론 범위를 넓힌다.

이러한 방식들은 공통적으로 학습 과정이 복잡하고 시간이 많이 걸린다는 단점이 있다.

### 2. Transformer

최근 Vision Transformer는 이미지 분류 및 세그멘테이션 등 다양한 분야에서 성과를 거두고 있다. 특히 Transformer의 long-term dependency 메커니즘은 약지도 학습(weakly supervised) 문제에서 이미지 카테고리와 지역 특징 간의 관계를 캡처하는 데 유용하다. 하지만 기존 연구들은 주로 class token과 patch token 간의 관계에 집중했을 뿐, 서로 다른 patch token 간의 상관관계를 최적화하고 활용하려는 시도는 부족했다. SASFormer는 이 지점에 집중하여 패치 간의 부정확한 상관관계를 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

SASFormer는 SegFormer를 백본(backbone)으로 사용하며, 전체적인 흐름은 다음과 같다. 입력 이미지를 패치 단위로 나누어 계층적 Transformer 인코더에 통과시켜 멀티 레벨 특징을 추출하고, 이를 디코더를 통해 세그멘테이션 확률 맵 $P$로 변환한다. 학습 단계에서는 인코더의 어텐션 맵을 활용해 라벨 정보를 전파하고, 이를 기반으로 예측값과 전파된 값 사이의 일관성을 맞추는 Affinity Loss를 적용한다.

### 패치 어텐션 맵 생성 (Patch Attention Map Generation)

인코더 블록의 각 Transformer 레이어에서 효율적인 self-attention을 통해 패치 어텐션 맵 $A_{l,n}$을 생성한다.

$$A_{l,n} = \text{softmax}\left(\frac{Q_{l,n} K_{l,n}^T}{\sqrt{D}}\right)$$

여기서 $Q_{l,n}$과 $K_{l,n}$은 각각 $l$번째 인코더 블록의 $n$번째 레이어의 query와 key 표현이며, $D$는 임베딩 차원이다. 각 블록 내의 여러 레이어에서 생성된 어텐션 맵들을 평균 내어 최종 패치 어텐션 맵 $A_l$을 구한다.

$$A_l = \frac{1}{N} \sum_{n=1}^{N} A_{l,n}$$

이 어텐션 맵은 저차원(색상, 질감)부터 고차원(시맨틱 일관성)까지의 픽셀 간 의존성을 캡처한다.

### Affinity Loss Function

라벨이 있는 픽셀의 정보를 없는 픽셀로 전달하기 위해, 네트워크의 예측 확률 $P$를 어텐션 맵 $A_l$과 행렬 곱셈하여 전파된 세그멘테이션 확률 $Y_l$을 생성한다.

$$Y_l = A_l \otimes P'_l$$

여기서 $P'_l$은 예측 확률 $P$를 어텐션 맵의 해상도에 맞게 보간(interpolate)한 값이다. 이후 $Y_l$과 $P_l$을 소프트맥스(softmax)를 통해 정규화하여 $Y^*_l$과 $P^*_l$을 얻는다.

Affinity Loss $L_{aff}$는 이 두 확률 분포 사이의 $L_1$ 거리를 최소화함으로써, 전파된 정보와 네트워크의 예측이 일치하도록 유도한다.

$$L_{aff} = \frac{1}{L} \sum_{l=1}^{L} \|Y^*_l - P^*_l\|_1$$

최종 손실 함수는 라벨링된 영역에 대한 표준 Cross Entropy Loss $L_{seg}$와 $L_{aff}$의 가중 합으로 정의된다.

$$L = L_{seg} + \alpha * L_{aff}$$

## 📊 Results

### 실험 설정

- **데이터셋:** PASCAL VOC 2012 (Point-wise 및 Scribble-wise annotation), Cityscapes, ADE20k.
- **지표:** mIoU (mean Intersection over Union).
- **백본:** SegFormer-B4 (ImageNet 사전 학습 가중치 사용).
- **하이퍼파라미터:** 학습 80k epoch, 초기 학습률 0.001, SGD 옵티마이저 사용. $\alpha$ 값은 Scribble 설정에서 1.2, Point 설정에서 0.2로 설정하였다.

### 주요 결과

1. **PASCAL VOC 2012:**
   - **Point-wise:** Baseline(64.21%) 및 TEL(69.33%) 대비 높은 **73.13% mIoU**를 달성하였다.
   - **Scribble-wise:** Baseline(73.37%) 및 TEL(78.53%)을 앞서는 **79.49% mIoU**를 기록하며 SOTA 성능을 보였다.
2. **Cityscapes & ADE20k:** 라벨 희소성(10%, 20%, 50%)에 관계없이 기존 방법론들보다 우수한 성능을 보였으며, 특히 라벨이 적을수록 SASFormer의 성능 향상 폭이 뚜렷했다.

### Ablation Study

- **멀티 스케일 효과:** 단일 인코더 블록의 어텐션 맵만 사용하는 것보다, 모든 블록($L_1$부터 $L_4$까지)의 Affinity Loss를 모두 합산했을 때 가장 높은 성능(79.49%)이 나타났다.
- **거리 측정 방식:** L1 distance가 KL divergence, Cross Entropy, L2 distance보다 가장 좋은 성능을 보였다.
- **백본 유연성:** Segmentor, SETR 등 다른 Transformer 기반 네트워크에 적용했을 때도 공통적으로 성능 향상이 확인되어, 제안 방법론의 범용성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 Vision Transformer의 self-attention 맵이 단순히 특징 추출 도구가 아니라, 라벨 정보를 전파하는 매개체로 활용될 수 있음을 보여주었다. 시각화 결과, 낮은 레벨의 어텐션 맵은 색상과 질감의 유사성을 캡처하고, 높은 레벨로 갈수록 시맨틱한 일관성을 캡처하는 특성을 보였다. 특히 Affinity Loss를 적용함으로써, 기존 Transformer가 배경 영역에 주의를 분산시키던 문제(distractions from irrelevant background)를 효과적으로 억제하고 객체 내부의 결합력을 높인 점이 고무적이다.

### 한계 및 논의

본 연구는 단일 단계 학습으로 효율성을 극대화했으나, 여전히 $\alpha$와 같은 하이퍼파라미터 설정이 성능에 영향을 미친다는 점이 명시되어 있다. 또한, 매우 극단적으로 적은 양의 라벨이 제공될 때 Transformer의 어텐션 맵이 얼마나 신뢰할 수 있는 전파 경로를 형성할 수 있을지에 대한 정밀한 분석은 부족한 편이다.

## 📌 TL;DR

SASFormer는 SegFormer를 기반으로 하여, 희소한 라벨(Point, Scribble)만으로도 고성능 세그멘테이션을 수행하는 단일 단계 프레임워크이다. Transformer의 패치 어텐션 맵을 이용해 라벨 정보를 미라벨 영역으로 전파하고, 전파된 결과와 예측값 사이의 일관성을 맞추는 **Affinity Loss**를 도입하여 복잡한 다단계 학습 없이도 SOTA 성능을 달성하였다. 이 연구는 Transformer의 내재적 특징을 활용해 약지도 학습의 복잡도를 획기적으로 낮추었다는 점에서 향후 효율적인 세그멘테이션 연구에 중요한 이정표가 될 가능성이 높다.
