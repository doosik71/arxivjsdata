# MAXIMUM-ENTROPY ADVERSARIAL AUDIO AUGMENTATION FOR KEYWORD SPOTTING

Zuzhao Ye, Gregory Ciccarelli, Brian Kulis (2024)

## 🧩 Problem to Solve

본 논문은 오디오 데이터, 특히 Keyword Spotting (KWS, 특정 키워드 검출) 작업에서 딥러닝 모델의 성능을 향상시키기 위한 효율적인 데이터 증강(Data Augmentation) 방법을 제안한다. 딥러닝 모델은 일반적으로 방대한 양의 레이블링된 데이터를 필요로 하지만, 실제 환경에서는 언어, 지역, 디바이스 종류, 키워드 설정에 따라 데이터 희소성(Data Scarcity) 문제가 빈번하게 발생한다.

기존의 오디오 증강 기법인 SpecAugment는 시간 및 주파수 마스킹을 통해 효과를 보였으나, 마스킹 비율과 같은 하이퍼파라미터를 매우 정밀하게 튜닝해야 하며, 잘못 설정할 경우 오히려 모델 성능을 저하시키는 단점이 있다. 또한, GAN(Generative Adversarial Networks) 기반의 증강 방식은 데이터 생성 능력이 뛰어나지만, 학습 과정이 복잡하고 계산 비용이 매우 높다는 한계가 있다. 따라서 본 연구의 목표는 구현이 간단하고 계산 효율적이면서도, 적은 데이터 상황에서 모델의 강건성과 성능을 높일 수 있는 새로운 오디오 증강 전략을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Adversarial Learning의 개념을 차용하되, 공격에 대한 방어(Robustness)가 아닌 데이터 증강을 목적으로 사용하는 것이다. 특히, 모델 출력의 Entropy를 최대화하는 방향으로 입력 데이터를 변형하여 새로운 데이터를 생성하는 **Adversarial Training with Entropy (ATE)** 방법을 제안한다.

핵심 직관은 모델이 예측하기 가장 어려워하는(즉, Entropy가 가장 높은) 지점의 데이터를 생성하여 학습에 활용함으로써, 모델이 결정 경계(Decision Boundary) 근처의 데이터를 더 잘 학습하게 유도하는 것이다. 이는 복잡한 최적화 과정이나 별도의 생성 모델 학습 없이, 단순히 입력 데이터에 대한 Entropy의 그라디언트(Gradient)를 계산하는 것만으로 구현 가능하다.

## 📎 Related Works

오디오 데이터 증강은 크게 Raw Audio 도메인(Time shifting, Pitch scaling 등)과 Spectrogram 도메인으로 나뉜다. Spectrogram 도메인에서는 SpecAugment가 사실상의 표준(Gold Standard)으로 사용되고 있으며, 최근에는 레이블 믹싱을 적용한 SpecMix 등이 제안되었다.

Adversarial Training 분야에서는 모델을 공격하여 취약점을 찾고 이를 학습에 반영하는 FGSM(Fast Gradient Sign Method)과 같은 기법들이 연구되었다. 또한, Maximum Entropy를 활용한 데이터 증강 연구(ME-ADA)가 존재하지만, 이는 복잡한 Min-max 최적화 과정을 거치며 학습 초기 단계에서만 적용되어 학습 효율성을 크게 떨어뜨리는 문제가 있다. 본 논문은 이러한 복잡한 최적화를 배제하고, 단순한 Gradient Ascent를 통해 실시간으로 증강 데이터를 생성함으로써 계산 효율성과 성능의 균형을 맞추고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
ATE의 기본 흐름은 입력 데이터 $x$를 네트워크에 통과시켜 얻은 Softmax 출력의 Entropy를 계산하고, 이 Entropy를 최대화하는 방향으로 입력 $x$를 미세하게 조정하여 증강 데이터 $x^{aug}$를 생성하는 것이다.

### 상세 구성 요소 및 절차
1. **Entropy 계산**:
   네트워크의 최종 출력층에서 Softmax를 통해 클래스 확률 분포 $p$를 얻는다. 이 때, 이진 분류(Binary Classification)의 경우 Entropy $E$는 다음과 같이 정의된다.
   $$E = -[p \log p + (1-p) \log(1-p)]$$
   다중 클래스 분류(Multiclass Classification)의 경우, $N$개의 클래스에 대해 다음과 같이 계산한다.
   $$E = -\sum_{i=1}^{N} p_i \log p_i$$

2. **증강 데이터 생성**:
   입력 $x$에 대한 Entropy의 그라디언트 $\nabla_x E$를 계산하고, Gradient Ascent를 적용하여 $E$를 증가시키는 방향으로 $x$를 이동시킨다. 이때, 급격한 변화를 막기 위해 Clipping 연산을 적용한다.
   $$x^{aug} = x + \text{clip}(\nabla_x E(x, y, \theta), -\epsilon, \epsilon)$$
   여기서 $\epsilon$은 clipping 임계값이며, 본 실험에서는 학습 데이터의 표준편차(one standard deviation)로 설정되었다.

3. **학습 절차 (Algorithm 1)**:
   - 배치 데이터 $(x, y)$를 샘플링한다.
   - 설정된 확률 $P_{aug}$ (본 논문에서는 0.5)에 따라 증강 여부를 결정한다.
   - 증강을 수행하는 경우, 위 식을 통해 $x^{aug}$를 생성하고 이를 이용해 모델 파라미터 $\theta$를 업데이트한다: $\theta \leftarrow \theta - \alpha \nabla_{\theta} \ell(x^{aug}, y, \theta)$.
   - 증강을 수행하지 않는 경우, 원래의 $x$를 사용하여 업데이트한다.

이 과정은 증강 데이터를 생성하기 위한 역전파(Back-propagation) 한 번과, 실제 가중치 업데이트를 위한 역전파 한 번, 총 두 번의 패스가 필요하므로 계산 시간이 약 두 배 증가한다.

## 📊 Results

### 실험 설정
- **데이터셋**:
    - Amazon 내부 데이터: "Computer", "Amazon", "Echo" 키워드 (인도 지역, 각각 10k 및 100k 샘플).
    - 공개 데이터셋: ESC-50 (2,000개), UrbanSound8K (8,000개), Speech Commands v2 (105,000개).
- **전처리**: 64차원 log Mel-filterbank energy (LFBE) 특징 추출.
- **모델**: 5개의 Convolutional layer와 3개의 Fully Connected layer로 구성된 CNN (약 2M 파라미터).
- **지표**: Amazon 데이터는 고정된 FRR(False Reject Rate)에서의 FAR(False Accept Rate)을 측정하였고, 공개 데이터셋은 Accuracy를 측정하였다.

### 주요 결과
1. **Amazon 데이터셋 (Table 1)**:
   - ATE를 단독으로 사용하는 것보다, ATE를 먼저 적용하고 그 결과물에 SpecAugment를 적용하는 **A+S** 조합이 모든 데이터셋에서 가장 우수한 성능을 보였다.
   - 특히 데이터 양이 적은 10k 데이터셋에서 효과가 극대화되었으며, NoAug 대비 최대 47.2% (Computer 키워드)의 상대적 성능 향상을 달성하였다.

2. **공개 데이터셋 (Table 2)**:
   - **강건성(Robustness)**: ESC-50, SCV2 데이터셋에서 SpecAugment와 SpecMix는 오히려 Accuracy를 떨어뜨리는 경향을 보였으나, ATE는 일관되게 성능을 향상시키거나 유지하여 더 강건함을 입증하였다.
   - **효율성(Efficacy)**: A+S 조합이 전반적으로 가장 높은 성능을 냈으며, 특히 ESC-50에서 최적의 결과를 보였다.
   - **계산 효율성(Training Efficiency)**: SCV2 데이터셋 기준, ATE는 ME-ADA보다 훨씬 빠른 학습 시간을 기록하였다. ME-ADA는 Min-max 최적화 과정에서 데이터셋 크기가 배수로 증가하기 때문에 학습 시간이 매우 길어지는 반면, ATE는 적절한 시간 증가만으로 유사하거나 더 나은 성능을 냈다.

## 🧠 Insights & Discussion

본 논문은 단순한 Gradient Ascent 기반의 Entropy 최대화 전략이 오디오 증강에서 매우 효과적일 수 있음을 보여주었다. 특히 주목할 점은 **SpecAugment와의 시너지 효과**이다. SpecAugment는 정적인 마스킹을 수행하는 반면, ATE는 모델의 현재 상태에 기반한 동적인 Adversarial 샘플을 생성하므로, 두 기법을 결합했을 때 상호 보완적인 증강 효과가 나타나는 것으로 해석된다.

또한, 기존의 Adversarial Training이 주로 '공격에 대한 방어'를 위해 Loss를 최대화하는 방향으로 이루어져 오히려 일반적인 정확도를 떨어뜨리는 경우가 많았던 것과 달리, 본 연구는 **Entropy를 최대화**함으로써 모델이 불확실한 영역을 학습하게 하여 일반화 성능을 높였다는 점이 핵심적인 차별점이다.

한계점으로는 ATE가 SpecMix와 같은 레이블 믹싱 기법과는 시너지가 나지 않았다는 점이 언급되었다. 이는 레이블 믹싱의 특성과 Entropy 최대화 방향성이 서로 충돌할 가능성이 있음을 시사하며, 이에 대한 추가적인 연구가 필요함을 보여준다.

## 📌 TL;DR

본 논문은 모델 출력의 Entropy를 최대화하는 방향으로 입력 데이터를 변형하는 **Adversarial Training with Entropy (ATE)**라는 단순하고 효율적인 오디오 증강 기법을 제안한다. 이 방법은 특히 데이터가 부족한 Keyword Spotting 환경에서 강력한 성능을 발휘하며, 기존의 SpecAugment와 결합했을 때 최적의 성능을 낸다. 복잡한 GAN이나 Min-max 최적화 없이 그라디언트 계산만으로 구현 가능하여 실용성이 매우 높으며, 향후 다양한 음성 및 오디오 인식 작업에 적용될 가능성이 크다.