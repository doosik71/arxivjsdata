# Learning to Segment Object Candidates via Recursive Neural Networks

Tianshui Chen, Liang Lin, Xian Wu, Nong Xiao, and Xiaonan Luo (2018)

## 🧩 Problem to Solve

본 논문은 객체 탐지(Object Detection) 및 인식(Recognition) 시스템의 전처리 단계에서 매우 중요한 역할을 하는 **Object Proposal Generation**(객체 후보 영역 생성) 문제를 다룬다.

기존의 객체 탐지 시스템은 이미지 내의 모든 위치와 크기에 대해 전수 조사를 수행하는 것이 계산 비용 측면에서 매우 비효율적이기 때문에, 객체가 존재할 가능성이 높은 소수의 후보 영역(Proposals)을 먼저 생성하는 방식을 사용한다. 좋은 Object Proposal 방법론은 다양한 크기와 위치의 객체를 모두 찾아내어 높은 재현율(Recall)을 확보하는 동시에, 객체의 경계(Boundary)를 정확하게 보존해야 한다.

하지만 기존의 Bottom-up 방식의 영역 그룹화(Region Grouping) 방법들은 주로 수동으로 설계된 **고정된 유사도 측정 지표(Fixed Similarity Measures)**를 사용하여 영역을 병합한다. 이러한 방식은 복잡한 환경에서 최적의 성능을 내기 어렵고, 정교한 튜닝이 필요하다는 한계가 있다. 따라서 본 논문의 목표는 재귀 신경망(Recursive Neural Networks, ReNN)을 통해 영역 병합 유사도와 Objectness(객체성) 측정 지표를 적응적으로 학습하여, 정확도와 효율성이 모두 높은 객체 후보 영역 생성 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같이 요약할 수 있다.

1. **ReNN 기반의 계층적 영역 그룹화 아키텍처 제안**: 영역 간의 유사도 측정과 Objectness 예측을 재귀적으로 학습하는 딥러닝 구조를 설계하여, 끝단간(End-to-End) 학습이 가능하게 하였다.
2. **학습 가능한 유사도 및 Objectness 지표**: 수동 설계된 지표 대신, 구조적 손실 함수(Structured Loss)를 통해 영역 병합 과정과 객체성 예측을 공동 최적화(Joint Optimization)한다.
3. **Randomized Merging Algorithm 도입**: 그리디(Greedy)하게 가장 높은 점수의 영역만 병합할 때 발생하는 지역 최적점(Local Minima) 및 오병합 문제를 해결하기 위해, 상위 $k$개의 후보 중 확률적으로 병합 대상을 선택하는 무작위 병합 알고리즘을 제안하여 Recall을 크게 향상시켰다.

## 📎 Related Works

논문에서는 Object Proposal 생성 방법을 크게 두 가지 범주로 나눈다.

1. **Window-based Scoring**: 슬라이딩 윈도우를 통해 각 윈도우에 Objectness 점수를 부여하는 방식이다. BING이나 EdgeBox 등이 이에 해당하며 계산 효율성이 높지만, IoU(Intersection over Union) 임계값이 높아질수록 재현율이 급격히 떨어지는 경향이 있어 정밀한 위치 지정(Localization)에 한계가 있다.
2. **Region Grouping**: 초기 과분할(Over-segmentation) 상태에서 영역들을 계층적으로 병합하는 방식이다. Selective Search(SS), MCG 등이 대표적이며, 윈도우 기반 방식보다 위치 정확도가 높다. 그러나 대부분 고정된 유사도 지표를 사용하므로 복잡한 케이스에서 성능이 저하된다.

본 연구는 이러한 기존 Region Grouping 방식의 한계를 극복하기 위해 유사도 측정 지표 자체를 학습 가능하게 만들었다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인

전체 시스템은 다음과 같은 순서로 진행된다.

- **Over-segmentation**: Graph-based 방법을 통해 이미지를 $N$개의 초기 영역으로 분할한다.
- **Local Feature Extraction**: VGG16-net 기반의 Fast R-CNN 아키텍처를 사용하여 각 영역의 ROI Pooling을 통해 고정 길이의 특징 벡터 $v_i$를 추출한다.
- **Recursive Region Grouping**: ReNN을 통해 영역들을 계층적으로 병합하며 Object Proposal을 생성한다.

### 2. Recursive Neural Networks (ReNN) 구조

ReNN은 다음 네 가지 모듈로 구성된다.

- **Semantic Mapper ($F_s$)**: 지역 특징 $v_i$를 세만틱 공간의 특징 $x_i$로 매핑한다.
  $$x_i = F_s(v_i; \theta_s) = \sigma(W_s v_i + b_s)$$
- **Feature Combiner ($F_c$)**: 인접한 두 자식 노드의 세만틱 특징을 입력받아 부모 노드의 세만틱 특징을 생성한다.
  $$x_{i,j} = F_c([x_i, x_j]; \theta_c) = \sigma(W_c [x_i, x_j] + b_c)$$
- **Merging Scorer ($F_m$)**: 두 영역이 병합되어야 할 확신도를 점수로 계산한다.
  $$s_{i,j} = F_m(x_{i,j}; \theta_m) = W_m x_{i,j} + b_m$$
- **Objectness Scorer ($F_o$)**: 해당 영역이 객체를 포함하고 있을 확률을 예측한다. 두 개의 Fully Connected 레이어와 Softmax $\phi(\cdot)$를 사용한다.
  $$o_i = F_o(x_i; \theta_o) = \phi(W_{o1} \sigma(W_{o0} x_i + b_{o0}) + b_{o1})$$

### 3. Randomized Merging Algorithm

그리디 병합은 한 번의 잘못된 병합이 발생하면 해당 객체를 영영 찾지 못하는 치명적인 문제가 있다. 이를 해결하기 위해 상위 $k$개의 병합 후보쌍 $\{(r_{it}, r_{jt})\}_{t=1}^k$을 선정하고, 다음의 다항 분포(Multinomial Distribution)에 따라 하나를 무작위로 선택하여 병합한다.

$$\rho_{it,jt} = \frac{\exp(s_{it,jt})}{\sum_{t=1}^k \exp(s_{it,jt})}$$

이 과정을 $K$번 반복 수행하여 생성된 후보군들의 다양성을 확보함으로써 Recall을 높인다.

### 4. Optimization 및 손실 함수

전체 손실 함수 $L$은 다음과 같이 정의된다.
$$L = L_m + \lambda L_o + \frac{\eta}{2} ||\theta||_2^2$$

- **Merging Loss ($L_m$)**: 마진 손실(Margin Loss) 형태를 취하며, 서로 다른 클래스의 영역이 같은 클래스의 영역보다 먼저 병합되는 경우 페널티를 부여한다. 정답 트리(Correct Tree)의 점수가 오답 트리보다 높도록 학습시킨다.
- **Objectness Loss ($L_o$)**: 예측된 Objectness 점수와 실제 정답(IoU $\ge 0.5$는 Positive, $\le 0.2$는 Negative) 간의 Cross-entropy 손실을 계산한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: PASCAL VOC 2007(학습 및 테스트), VOC 2012(검증), ImageNet 2015(일반화 성능 테스트).
- **평가 지표**: IoU 임계값에 따른 Recall, Average Recall (AR), 그리고 Fast R-CNN을 결합한 mAP.
- **비교 대상**: BING, EdgeBox, Selective Search, RPN, MCG2015 등 최신 방법론.

### 2. 주요 결과

- **정량적 성능**: PASCAL VOC 2007 및 2012에서 대부분의 지표(AR, Recall @ 0.8)가 기존 방법론보다 우수하게 나타났다. 특히 IoU 임계값이 높은(정밀한) 상황에서 강점을 보였다.
- **객체 탐지 성능**: Fast R-CNN 프레임워크에 적용했을 때, CaffeNet 기반 mAP 58.6%, VGG-Net 기반 mAP 69.0%를 기록하여 비교 대상 중 가장 높은 성능을 보였다.
- **일반화 능력**: PASCAL VOC로만 학습한 모델을 ImageNet 2015에 적용했을 때도 성능 저하가 적어, 범용적인 객체성(Generic Objectness)을 잘 학습했음을 입증하였다.
- **효율성**: 이미지당 약 4.2초가 소요되며, 이는 매우 정밀하지만 매우 느린 MCG2015(30초 이상)보다 7배 이상 빠르면서도 성능은 더 높다.

## 🧠 Insights & Discussion

**강점 및 성과**
본 논문은 고정된 유사도 지표를 사용하는 기존의 Bottom-up 방식에서 벗어나, 신경망을 통해 유사도와 객체성을 직접 학습시켰다는 점이 매우 고무적이다. 특히 Randomized Merging은 단순한 아이디어임에도 불구하고, 그리디 탐색의 취약점인 '초기 오병합' 문제를 효과적으로 완화하여 Recall을 유의미하게 끌어올렸다.

**한계 및 분석**
실험 결과, 아주 작은 크기의 객체(면적 5k 픽셀 미만)에 대해서는 RPN보다 성능이 다소 낮게 나타났다. 이는 본 방법론이 초기 Over-segmentation 결과에 크게 의존하기 때문으로 분석된다. 작은 객체의 경우 초기 분할 단계에서 경계가 뭉개질 가능성이 높아, 이후의 ReNN 학습만으로는 이를 완벽히 복구하기 어렵다는 한계가 있다.

**비판적 해석**
본 연구는 Proposal 생성 단계에서의 학습에 집중하였으나, 결국 Fast R-CNN과 같은 탐지기와 결합되었을 때의 성능이 최종 지표가 된다. ReNN 구조가 복잡함에도 불구하고 추론 속도가 합리적인 수준으로 유지된 점은 긍정적이나, 무작위 병합을 $K$번 반복하는 과정이 실제 실시간 시스템에 적용될 때 어느 정도의 오버헤드가 될지에 대한 더 깊은 분석이 필요해 보인다.

## 📌 TL;DR

본 논문은 **Recursive Neural Networks(ReNN)**를 사용하여 객체 후보 영역을 생성하는 새로운 프레임워크를 제안한다. 핵심은 **영역 병합 유사도와 객체성 점수를 데이터로부터 직접 학습**하고, **무작위 병합 알고리즘**을 통해 그리디 탐색의 한계를 극복하여 높은 재현율(Recall)과 정밀한 경계 보존 능력을 확보한 것이다. 실험적으로 PASCAL VOC와 ImageNet에서 기존 SOTA 방법들보다 우수한 성능과 효율성을 입증하였으며, 이는 향후 고정밀 객체 탐지 시스템의 전처리 단계에서 중요한 역할을 할 가능성이 높다.
