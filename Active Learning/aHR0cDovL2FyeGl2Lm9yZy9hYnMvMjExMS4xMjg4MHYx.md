# Active Learning at the ImageNet Scale

Zeyad Ali Sami Emam, Hong-Min Chu, Ping-Yeh Chiang, Wojciech Czaja, Richard Leapman, Micah Goldblum, Tom Goldstein (2021)

## 🧩 Problem to Solve

본 논문은 대규모 데이터셋인 ImageNet 규모에서 Active Learning(AL)을 적용할 때 발생하는 성능 저하 문제를 해결하고자 한다. 일반적으로 AL은 레이블이 없는 데이터 중 가장 정보량이 많은 샘플을 선택하여 레이블링 비용을 줄이고 모델 성능을 최적화하는 것을 목표로 한다.

하지만 기존의 AL 알고리즘들은 CIFAR-10과 같은 소규모 데이터셋에서는 효과적이었으나, ImageNet과 같이 클래스 수가 많고(1,000개) 데이터 특성이 매우 이질적인 대규모 데이터셋에서는 오히려 Random Sampling보다 성능이 떨어지는 현상이 발생한다. 저자들은 이러한 성능 저하의 주요 원인이 AL 알고리즘이 특정 클래스에 치우쳐 샘플을 선택하는 '샘플링 불균형(Sampling Imbalance)' 문제에 있음을 지적하며, 이를 해결하여 대규모 환경에서도 유효한 AL 전략을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AL의 대규모 확장 시 발생하는 클래스 불균형 문제를 식별하고, 이를 해결하기 위한 **Balanced Selection (BASE)** 알고리즘을 제안한 것이다. 

BASE의 중심 아이디어는 단순히 모델이 불확실해하는 샘플을 뽑는 것이 아니라, 특성 공간(Feature Space)에서 결정 경계(Decision Boundary)에 가까운 샘플들을 선택하되, 모든 클래스에서 균등한 수의 샘플이 선택되도록 강제하는 것이다. 이를 통해 정보량이 많은 샘플을 확보함과 동시에 클래스 분포의 균형을 유지하여, 대규모 데이터셋에서도 Random Sampling 이상의 성능을 달성할 수 있음을 입증하였다.

## 📎 Related Works

기존의 AL 알고리즘은 크게 두 가지 범주로 나뉜다. 첫째는 모델의 예측 불확실성을 측정하는 **Uncertainty-based sampling** (예: Entropy, Least Confidence, Margin sampling)이며, 둘째는 특성 공간에서 데이터의 대표성을 고려하는 **Density-based sampling** (예: Coreset, BADGE)이다.

저자들은 이러한 기존 방식들이 다음과 같은 한계가 있음을 언급한다:
1. **확장성 부족**: Coreset이나 BADGE와 같은 방식은 계산 복잡도가 매우 높아 ImageNet 규모의 데이터셋(약 120만 장)에 적용하기 어렵다.
2. **클래스 불균형 무시**: 대부분의 알고리즘이 데이터셋이 균형 잡혀 있다는 가정하에 설계되어, ImageNet과 같은 복잡한 데이터셋에서는 특정 클래스만 과도하게 선택하는 경향이 있다.
3. **SSP와의 상호작용**: 최근 Self-supervised Pretraining (SSP)이 모델 성능을 크게 향상시키고 있으나, SSP가 적용된 초기 가중치 상태에서 AL이 추가적인 이득을 줄 수 있는지에 대한 연구가 부족했다.

## 🛠️ Methodology

### 전체 파이프라인
본 논문은 SSP를 통해 사전 학습된 백본 네트워크를 사용하며, 이후 AL 루프를 통해 선택된 데이터로 분류기(Classifier)를 학습시킨다. 특히 백본을 고정하고 선형 분류기만 학습시키는 **Linear Evaluation** 설정과 전체 네트워크를 미세 조정하는 **End-to-end Finetuning** 설정을 모두 고려한다.

### Margin Selection (MASE)
BASE의 기초가 되는 MASE는 결정 경계에 가장 가까운 샘플을 선택하는 방식이다. 결정 경계까지의 거리인 Distance to Decision Boundary (DDB)를 다음과 같이 정의한다:

$$DDB(x) = \min_{\epsilon} \|\epsilon\|^2 \text{ s.t. } f(x+\epsilon) \neq f(x)$$

입력 공간에서 이를 계산하는 것은 비용이 매우 크므로, 저자들은 특성 공간(Feature Space)에서 이 거리를 추정한다. 이는 특성 벡터를 선형 결정 경계의 법선 벡터(Normal Vector)에 투영하는 방식으로 효율적으로 구현된다.

### Balanced Selection (BASE)
MASE가 전체 샘플 중 상위 $b$개를 뽑는다면, BASE는 클래스별로 균형을 맞추기 위해 **Class-Specific Distance to Decision Boundary (DCSDB)**를 도입한다.

$$DCSDB(x, c) = 
\begin{cases} 
\min_{\epsilon} \|\epsilon\|^2 \text{ s.t. } f(x+\epsilon) = c & \text{if } f(x) \neq c \\
\min_{\epsilon} \|\epsilon\|^2 \text{ s.t. } f(x+\epsilon) \neq c & \text{if } f(x) = c 
\end{cases}$$

**학습 및 선택 절차:**
1. 각 클래스 $c \in \{1, \dots, C\}$에 대하여, $DCSDB(x, c)$ 값이 가장 작은(즉, 경계에 가장 가까운) 샘플을 $b/C$개씩 선택한다.
2. 선택된 샘플들에 대해 레이블을 부여하고 학습 데이터셋 $\mathcal{D}_L$에 추가한다.
3. 이 과정을 정해진 예산이 소진될 때까지 반복한다.

BASE의 시간 복잡도는 $O(C \cdot (d' + \log b) \cdot |\mathcal{D}_U|)$이며, 여기서 $d'$는 특성 공간의 차원, $C$는 클래스 수, $|\mathcal{D}_U|$는 레이블이 없는 데이터의 수이다. 이는 다른 효율적인 베이스라인들과 대등한 수준의 속도를 가진다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-10, Imbalanced CIFAR-10, ImageNet.
- **모델**: ResNet-18, ResNet-50, ViT.
- **지표**: Top-1 및 Top-5 Accuracy.
- **베이스라인**: Random Sampler, Coreset, BADGE, Confidence, Margin, VAAL 등.

### 주요 결과
1. **ImageNet에서의 성능**: Linear Evaluation 설정에서 대부분의 기존 AL 알고리즘이 Random Sampling보다 낮은 성능을 보였다. 이는 AL이 유도한 클래스 불균형 때문이다. 반면, **BASE는 Random Sampling을 일관되게 능가**하였다.
2. **데이터 효율성**: state-of-the-art인 EsViT (SSP) 모델을 사용할 때, BASE를 적용하면 전체 데이터의 **71%만 사용하고도 동일한 Top-1 정확도**를 달성했으며, **55%의 데이터만으로도 동일한 Top-5 정확도**를 달성하였다.
3. **클래스 불균형 완화**: 분포 분석 결과, BASE는 Random Sampling 이후 가장 균형 잡힌 클래스 분포를 유지하는 전략임이 확인되었다.
4. **소규모 데이터셋 검증**: Imbalanced CIFAR-10 실험에서도 BASE는 클래스 균형을 잘 유지하며 높은 정확도를 기록하여, 제안 방법론의 일반성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 AL 알고리즘을 평가할 때 단순히 소규모 데이터셋에서의 성능만 믿어서는 안 된다는 중요한 통찰을 제공한다. ImageNet과 같은 대규모 데이터셋에서는 클래스 간의 이질성이 크기 때문에, 모델의 불확실성만으로 샘플링하면 특정 클래스에 편향된 샘플링이 일어나고, 이는 결국 전체 일반화 성능의 저하로 이어진다.

또한, SSP(Self-supervised Pretraining)가 매우 강력한 성능 향상을 제공하기 때문에, AL이 의미를 가지려면 Random Sampling + SSP 조합보다 더 나은 성능을 보여야 한다. 실험 결과, BASE는 SSP와 결합되었을 때 추가적인 성능 이득을 제공함으로써 AL이 대규모 설정에서도 여전히 가치가 있음을 보여주었다.

한계점으로는, AL 실험이 하이퍼파라미터와 초기 샘플 집합($s_0$)의 설정에 매우 민감하다는 점이 언급되었다. 특히 네트워크를 매 라운드마다 완전히 수렴(Saturation)할 때까지 학습시켜야 공정한 비교가 가능하므로 계산 비용이 매우 높다는 현실적인 어려움이 있다.

## 📌 TL;DR

이 논문은 대규모 이미지 데이터셋(ImageNet)에서 기존 Active Learning 알고리즘들이 클래스 불균형 문제로 인해 Random Sampling보다 성능이 떨어지는 현상을 분석하고, 이를 해결하기 위해 클래스별로 균등하게 결정 경계 샘플을 추출하는 **BASE (Balanced Selection)** 알고리즘을 제안하였다. BASE는 계산 효율성이 높으면서도 데이터 효율성을 극대화하여, SSP와 결합 시 훨씬 적은 양의 레이블(Top-5 기준 55%)만으로도 SOTA 성능에 도달할 수 있음을 입증하였다. 이는 향후 대규모 세그멘테이션이나 디텍션 작업의 레이블링 비용 절감 연구에 중요한 기초가 될 것으로 보인다.