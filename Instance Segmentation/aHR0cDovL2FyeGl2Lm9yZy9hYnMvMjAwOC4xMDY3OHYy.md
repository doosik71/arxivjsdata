# Probabilistic Deep Learning for Instance Segmentation

Josef Lorenz Rumberger, Lisa Mais, and Dagmar Kainmueller (2020)

## 🧩 Problem to Solve

본 논문은 **Proposal-free Instance Segmentation** 모델에서 모델 내재적 불확실성(model-inherent uncertainty)을 추정하는 방법을 제안한다. 

일반적인 딥러닝 모델은 점 추정치(point estimates)를 제공하므로 예측의 신뢰도를 알 수 없다. Probabilistic Deep Learning은 예측값의 분포를 통해 불확실성을 정량화할 수 있으며, 이는 의료 영상과 같은 안전 필수 환경(safety-critical environment)에서의 품질 평가나 능동 학습(active learning)에서 라벨링 효율을 높이는 데 매우 중요하다.

특히, 기존의 Proposal-based 방법(예: Mask R-CNN)은 바운딩 박스(bounding box)를 먼저 제안해야 하므로, 생물 의료 영상에서 자주 나타나는 가늘고 긴 곡선 구조의 객체를 분리하는 데 한계가 있다. 반면, Proposal-free 방법은 이러한 한계를 극복하여 최신 성능을 보여주고 있으나, 여전히 불확실성 추정 기능이 결여되어 있다는 문제가 있다. 따라서 본 논문의 목표는 Proposal-free Instance Segmentation에 적용 가능한 일반적인 불확실성 추정 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Bayesian Approximate CNN의 통합**: Proposal-free Instance Segmentation 모델에 Concrete Dropout 모델을 결합하여, 모델의 불확실성을 추정할 수 있는 최초의 시도를 하였다.
2. **불확실성 평가 지표의 확장**: 본래 Semantic Segmentation을 위해 제안된 불확실성 평가 지표를 Instance Segmentation 작업에 맞게 수정하여 정량적으로 분석하였다.
3. **가이드 기반 교정(Guided Proofreading) 가능성 제시**: 추정된 불확실성 지도를 사용하여 예측 오류가 발생한 지점을 효율적으로 찾아내고 수정할 수 있음을 시뮬레이션을 통해 입증하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계
- **Proposal-based Methods**: Mask R-CNN과 같은 모델은 객체 후보 영역을 먼저 제안한다. 하지만 생물학적 이미지의 가늘고 긴 구조물은 바운딩 박스 내에 다른 객체가 포함될 가능성이 높아 segmentation 성능이 저하된다.
- **Proposal-free Methods**: Watershed energy map, affinity-graph, metric space 등을 학습하여 인스턴스를 분리한다. 성능은 우수하지만, 최종 출력이 이진 맵(binary maps) 형태이므로 신뢰도나 불확실성 정보를 제공하지 않는다.
- **Probabilistic Deep Learning**: Dropout sampling 등을 통해 불확실성을 추정하려는 시도가 있었으나, 대개 Proposal-based 모델에 국한되었거나 end-to-end 학습이 불가능한 구조였다.

### 본 연구의 차별점
본 연구는 모델의 파라미터 분포를 학습하는 **Concrete Dropout**을 사용하여, 별도의 proposal 과정 없이도 픽셀 수준의 불확실성을 추정하고 이를 인스턴스 수준의 불확실성으로 확장하였다.

## 🛠️ Methodology

### 1. Metric Learning 및 손실 함수
본 모델은 각 픽셀을 임베딩 공간의 벡터로 투영하는 Metric Learning 방식을 사용한다. 동일 인스턴스의 임베딩은 가깝게, 서로 다른 인스턴스는 멀게 배치하기 위해 **Discriminative Loss**를 사용하며, 이는 다음 세 가지 항의 합으로 구성된다.

- **Variance Term ($L_{var}$)**: 임베딩 $e_{c,i}$를 해당 인스턴스의 중심 $\mu_c$로 끌어당긴다.
  $$L_{var} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i=1}^{N_c} ||\mu_c - e_{c,i}||^2$$
- **Distance Term ($L_{dist}$)**: 서로 다른 인스턴스의 중심들 사이의 거리가 최소 $2\delta_d$ 이상이 되도록 밀어낸다.
  $$L_{dist} = \frac{1}{C(C-1)} \sum_{c_A=1}^{C} \sum_{c_B=1, c_A \neq c_B}^{C} [2\delta_d - ||\mu_{c_A} - \mu_{c_B}||^2]_+^2$$
- **Regularization Term ($L_{reg}$)**: 임베딩 중심들을 원점으로 끌어당겨 정규화한다.
  $$L_{reg} = \frac{1}{C} \sum_{c=1}^{C} ||\mu_c||^2$$

최종 손실 함수는 $L_{disc} = L_{var} + L_{dist} + 0.001 \cdot L_{reg}$이며, 여기에 배경/전경/중첩 영역을 구분하는 3-클래스 교차 엔트로피 손실(Cross-Entropy Loss)이 추가된다.

### 2. Concrete Dropout Model
모델 불확실성(Epistemic Uncertainty)을 추정하기 위해 Concrete Dropout을 적용한다. 이는 Dropout 비율 $p$를 고정하지 않고 데이터로부터 학습 가능하게 만든 모델이다. 학습 목표는 다음과 같은 정규화 항 $L_{concrete}$를 포함한다.

$$L_{concrete} = \frac{1}{N} \cdot \left( \sum_{l=1}^{L} \frac{\iota^2}{(1-p_l)^2} ||M_l||^2 - \zeta F_l H(p_l) \right)$$

여기서 $M_l$은 $l$번째 층의 가중치 행렬, $p_l$은 Dropout 확률, $H(p_l)$은 베르누이 분포의 엔트로피이다. 이 손실 함수는 가중치 크기에 따른 불확실성의 왜곡을 방지하고, 학습 데이터가 적을수록 더 높은 불확실성을 가지도록 유도한다.

### 3. 후처리 및 추론 과정 (Inference)
1. **Draws 생성**: 하나의 입력 이미지를 Concrete Dropout U-Net에 8번 통과시켜 8개의 서로 다른 임베딩 맵을 얻는다.
2. **인스턴스 분리**: 각 맵에 대해 Mean-shift clustering을 적용하여 개별 인스턴스 맵들을 생성한다.
3. **Agglomeration (응집)**: 
   - 가장 많은 인스턴스를 포함한 맵을 기준(base)으로 설정한다.
   - 다른 맵들과의 IoU를 계산하고, **Hungarian matching algorithm**을 통해 최적의 매핑을 찾아 인스턴스들을 병합한다.
4. **확률 맵 생성**: 모든 draw에 대해 병합된 결과물을 합산하고 draw 횟수로 나누어, 각 픽셀이 특정 인스턴스에 속할 확률을 계산한다.

### 4. 불확실성 평가 지표: PAvPU
Semantic Segmentation의 지표를 수정하여 **Patch Accuracy vs. Patch Uncertainty (PAvPU)**를 제안한다. $4 \times 4$ 패치를 사용하여 다음 두 조건부 확률을 계산한다.
- $p(\text{accurate}|\text{certain})$: 모델이 확실하다고 판단했을 때 실제로 정확할 확률.
- $p(\text{uncertain}|\text{inaccurate})$: 모델이 틀렸을 때 실제로 불확실하다고 판단했을 확률.

이 두 지표를 결합하여 다음과 같이 정의한다.
$$\text{PAvPU} = \frac{n_{ac} + n_{iu}}{n_{ac} + n_{au} + n_{ic} + n_{iu}}$$
(여기서 $n_{ac}$: 정확하고 확실함, $n_{iu}$: 부정확하고 불확실함 등)

## 📊 Results

### 실험 설정
- **데이터셋**: BBBC010 C. elegans (선충) 데이터셋.
- **아키텍처**: 5-level U-Net.
- **비교 대상**: Semi-conv Ops, SON, Harmonic Embeddings, PatchPerPix 등.

### 정량적 결과
- **정확도**: Concrete Dropout 모델은 baseline 모델보다 약간 더 높은 성능을 보였으며, 특히 $\text{AP}_{.5}$ 지표에서는 최신 모델인 PatchPerPix와 경쟁하거나 더 나은 결과를 달성하였다.
- **불확실성 분석**: 불확실성 지도가 객체 수준의 오류(False Merge, False Split)가 발생하는 위치와 밀접하게 연관되어 있음을 확인하였다.
- **시뮬레이션 실험**: 불확실성이 가장 높은 상위 $N$개 패치를 찾아 ground truth로 수정했을 때, 적은 수의 수정($N=20$)만으로도 $\text{avAP}$가 $0.770$에서 $0.791$로 유의미하게 상승하였다. 이는 불확실성 추정치가 사람이 수동으로 오류를 교정하는 과정을 효과적으로 가이드할 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 Proposal-free Instance Segmentation에 베이지안 근사 기법을 성공적으로 도입하여, 단순한 분할 성능 향상을 넘어 **'예측의 신뢰도'**라는 중요한 정보를 제공하게 되었다. 특히, 시뮬레이션을 통해 실제 워크플로우(Guided Proofreading)에서의 활용 가능성을 증명한 점이 고무적이다.

### 한계 및 향후 과제
- **데이터 불확실성(Aleatoric Uncertainty)의 부재**: 현재 모델은 모델 파라미터의 불확실성(Epistemic)만 다루고 있다. 관측 노이즈나 라벨의 모호성으로 인한 데이터 불확실성을 함께 모델링하는 손실 함수 개발이 필요하다.
- **국소적 근사의 한계**: Concrete Dropout은 전체 사후 분포(full posterior)의 국소적 근사치만을 제공한다. 이를 해결하기 위해 서로 다른 로컬 옵티마를 탐색하는 **Probabilistic Model Ensemble** 기법을 도입하면 더 정교한 불확실성 추정이 가능할 것이다.
- **하이퍼파라미터 의존성**: Mean-shift의 bandwidth가 고정되어 있어 일부 케이스에서 False Merge/Split이 발생한다. 인스턴스별 최적 bandwidth를 학습하는 방향으로 확장이 필요하다.

## 📌 TL;DR

본 논문은 **Concrete Dropout**을 **Proposal-free Instance Segmentation**에 결합하여, 각 인스턴스 분할 결과에 대한 **모델 내재적 불확실성을 정량화**하는 방법론을 제안하였다. 제안된 방법은 기존 SOTA 모델과 경쟁 가능한 성능을 보이면서도, 분할 오류(Merge/Split)가 발생한 지점을 정확히 짚어내는 불확실성 지도를 제공한다. 이는 향후 의료 영상 분석에서 전문가의 효율적인 교정(Proofreading)을 돕거나, 능동 학습을 통해 라벨링 비용을 절감하는 데 핵심적인 역할을 할 것으로 기대된다.