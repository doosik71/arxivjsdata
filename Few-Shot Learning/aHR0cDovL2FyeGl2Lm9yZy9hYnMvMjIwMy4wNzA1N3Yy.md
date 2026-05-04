# Self-Promoted Supervision for Few-Shot Transformer

Bowen Dong, Pan Zhou, Shuicheng Yan, and Wangmeng Zuo (2022)

## 🧩 Problem to Solve

본 논문은 Vision Transformer(ViT)가 Few-Shot Learning(FSL) 환경에서 Convolutional Neural Networks(CNN)에 비해 현저히 낮은 성능을 보이는 문제에 집중한다. 일반적으로 ViT는 대규모 데이터셋에서 강력한 성능을 발휘하지만, 학습 데이터가 매우 제한적인 Few-Shot 설정에서는 CNN이 가진 Inductive Bias(귀납적 편향)의 부재로 인해 어려움을 겪는다.

특히 저자들은 ViT가 데이터가 부족한 상황에서 Patch Token 간의 의존성(Token Dependency)을 학습하는 속도가 느리며, 이로 인해 저품질의 토큰 의존성을 학습하게 되어 결과적으로 일반화 성능이 떨어진다는 점을 실험적으로 발견하였다. 따라서 본 연구의 목표는 ViT가 적은 양의 데이터만으로도 효율적으로 토큰 의존성을 학습하고, 새로운 클래스(Novel Classes)에 대해 높은 일반화 성능을 가질 수 있도록 하는 훈련 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-promoted sUpervisioN (SUN)**이라는 프레임워크를 통해 ViT에 조밀한 위치 기반 감독(Dense Location-specific Supervision)을 제공하는 것이다.

중심적인 직관은 ViT의 글로벌 시맨틱 학습뿐만 아니라, 개별 패치 토큰 수준에서 어떤 토큰이 유사하고 어떤 토큰이 다른지에 대한 가이드를 제공함으로써 토큰 의존성 학습을 가속화하는 것이다. 이를 위해 동일한 아키텍처를 가진 Teacher ViT를 사용하여 패치 수준의 Pseudo Label을 생성하고, 이를 Student ViT의 학습에 활용하는 '자기 촉진(Self-promoted)' 방식을 채택하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Vision Transformer (ViT):** ViT는 Global Self-attention을 통해 장거리 의존성을 캡처할 수 있으나, CNN과 달리 Inductive Bias가 부족하여 ImageNet이나 JFT-300M과 같은 초거대 데이터셋이 필수적이다. 소규모 데이터셋에서는 성능이 급격히 저하되는 경향이 있다.
2. **Few-Shot Classification:** 기존 방식은 크게 최적화 기반(Optimization-based), 메모리 기반(Memory-based), 거리 기반(Metric-based) 방법으로 나뉜다. 최근 일부 연구에서 Transformer 레이어를 분류기로 사용하는 시도가 있었으나, 이는 백본(Backbone) 자체의 Few-Shot 능력을 개선하는 것과는 차이가 있다.

### 기존 접근 방식과의 차별점

기존의 ViT 성능 향상 시도는 주로 CNN의 지식을 증류(Distillation)하거나 CNN 모듈을 직접 삽입하여 CNN-alike Inductive Bias를 강제로 부여하는 방식이었다. 반면, SUN은 ViT 고유의 구조를 유지하면서, 패치 수준의 조밀한 감독(Dense Supervision)을 통해 ViT가 스스로 효율적인 토큰 의존성을 학습하도록 유도한다는 점에서 차별화된다.

## 🛠️ Methodology

SUN 프레임워크는 **Meta-Training**과 **Meta-Tuning**의 두 단계로 구성된다.

### 1. Meta-Training Phase

이 단계의 목적은 새로운 클래스에 빠르게 적응할 수 있는 메타 학습자 $f$(ViT 백본)를 학습시키는 것이다.

**가. 위치 기반 감독(Location-specific Supervision) 생성**
먼저, 동일한 아키텍처를 가진 Teacher 모델 $f_g$ (ViT $f_0$와 분류기 $g_0$의 조합)를 베이스 데이터셋 $\mathcal{D}_{base}$로 사전 학습시킨다. 이후 입력 이미지 $x_i$에 대해 각 패치 토큰 $z_j$의 분류 신뢰도 점수 $\hat{s}_{ij}$를 계산하여 Pseudo Label을 생성한다.
$$\hat{s}_i = [\hat{s}_{i1}, \hat{s}_{i2}, \dots, \hat{s}_{iK}] = f_g(x_i) \in \mathbb{R}^{c \times K}$$

**나. Background Patch Filtration (BGF)**
Teacher 모델이 배경 패치를 강제로 특정 시맨틱 클래스로 할당하여 잘못된 감독 정보를 제공하는 것을 방지하기 위해 BGF를 도입한다. 신뢰도 점수가 가장 낮은 하위 $p\%$의 패치를 배경 클래스로 정의하고, 이를 새로운 독립 클래스(Background Class)로 할당하여 $s_{ij} \in \mathbb{R}^{c+1}$의 형태로 확장한다.

**다. Spatial-Consistent Augmentation (SCA)**
데이터 다양성을 확보하면서 Pseudo Label의 정확도를 유지하기 위해 SCA 전략을 사용한다.

- **Spatial-only augmentation:** Random crop, flip, rotation 등을 적용하여 $\tilde{x}_i$를 생성하고, 이를 Teacher 모델에 입력하여 Pseudo Label $s_{ij}$를 생성한다.
- **Non-spatial augmentation:** Color jitter, Gaussian blur 등을 $\tilde{x}_i$에 추가 적용하여 $\bar{x}_i$를 생성하고, 이를 Student 모델 $f$에 입력한다.
이를 통해 Student는 다양성이 높은 데이터를 학습하면서도, 비교적 덜 왜곡된 $\tilde{x}_i$로부터 생성된 정확한 로컬 가이드를 받을 수 있다.

**라. 손실 함수 (Loss Function)**
최종 학습 목표는 글로벌 시맨틱 손실과 로컬 패치 손실의 합을 최소화하는 것이다.
$$\mathcal{L}_{SUN} = H(g_{global}(z_{global}), y_i) + \lambda \sum_{j=1}^K H(g_{local}(z_j), s_{ij})$$
여기서 $z_{global}$은 모든 패치 토큰의 Global Average Pooling(GAP) 결과이며, $H$는 Cross Entropy Loss이다. $\lambda$는 로컬 손실의 가중치(본 논문에서는 0.5 사용)이다.

### 2. Meta-Tuning Phase

Meta-Training이 완료된 $f$를 기반으로, Meta-Baseline 방식을 따라 새로운 태스크에 적응시킨다.

- **프로토타입 계산:** 서포트 셋(Support set) $\mathcal{S}$에서 각 클래스 $k$의 특징 벡터 평균을 계산하여 프로토타입 $w_k$를 생성한다.
- **분류:** 쿼리 이미지 $x$의 특징 벡터와 프로토타입 간의 코사인 유사도를 계산하여 클래스를 예측한다.
$$p_k = \frac{\exp(\gamma \cdot \cos(GAP(f(x)), w_k))}{\sum_{k'} \exp(\gamma \cdot \cos(GAP(f(x)), w_{k'}))}$$
이후 Cross Entropy Loss를 사용하여 $f$를 미세 조정(Fine-tuning)한다.

## 📊 Results

### 실험 설정

- **데이터셋:** miniImageNet, tieredImageNet, CIFAR-FS.
- **백본 모델:** LV-ViT, Swin Transformer, Visformer, NesT (모두 ResNet-12와 유사한 $\sim 12.5\text{M}$ 파라미터 규모로 조정).
- **지표:** 5-way 1-shot 및 5-way 5-shot 분류 정확도 (95% 신뢰구간 포함).

### 주요 결과

1. **ViT 종류별 성능 향상:** SUN 프레임워크를 적용했을 때, 모든 종류의 ViT에서 Meta-Baseline 대비 비약적인 성능 향상이 관찰되었다. 특히 5-way 1-shot 설정에서 Meta-Baseline 대비 10.3%~20.2%의 성능 향상을 보였다.
2. **CNN SoTA와의 비교:** SUN을 적용한 ViT(특히 Visformer, NesT)는 기존의 강력한 CNN 기반 Few-Shot 모델들(RE-Net 등)과 대등하거나 오히려 능가하는 성능을 기록하였다. 특히 tieredImageNet과 CIFAR-FS에서는 새로운 SoTA(State-of-the-Art) 성능을 달성하였다.
3. **토큰 의존성 학습 시각화:** Attention Map 시각화 결과, SUN을 적용한 모델이 Vanilla ViT나 CNN Distillation 모델보다 더 많은 시맨틱 토큰을 정확하게 캡처함을 확인하였다. 이는 SUN이 토큰 간의 의존성 학습을 효과적으로 가속화했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석

- **로컬 시맨틱의 중요성:** 본 연구는 단순히 글로벌 라벨만으로 학습하는 것보다, 패치 수준의 세밀한 감독을 제공하는 것이 ViT의 일반화 성능을 크게 향상시킨다는 점을 입증하였다. 이는 모델이 엉뚱한 배경 패턴(Skewed patterns)에 의존하지 않고 실제 객체의 핵심 특징을 학습하도록 유도한다.
- **Inductive Bias의 대체:** CNN의 구조적 편향을 직접 주입하는 대신, 데이터 기반의 조밀한 감독을 통해 ViT가 스스로 최적의 의존성을 학습하게 함으로써 ViT의 잠재력을 최대한 끌어올렸다.

### 한계 및 논의사항

- **Teacher 모델 의존성:** SUN의 성능은 Teacher 모델이 생성하는 Pseudo Label의 품질에 의존한다. 비록 BGF와 SCA로 이를 보완하였으나, Teacher 모델 자체가 완전히 잘못된 예측을 할 경우 학습에 부정적인 영향을 줄 가능성이 있다.
- **계산 비용:** Meta-Training 단계에서 Teacher 모델을 사전 학습시키고 두 가지 증강 경로(SCA)를 통해 학습해야 하므로, 일반적인 학습보다 연산 시간이 증가한다.

## 📌 TL;DR

본 논문은 ViT가 Few-Shot Learning에서 CNN보다 성능이 낮은 원인이 **Inductive Bias 부족으로 인한 느린 토큰 의존성 학습**에 있음을 밝히고, 이를 해결하기 위해 **Self-promoted sUpervisioN (SUN)** 프레임워크를 제안한다. SUN은 동일 구조의 Teacher ViT를 통해 패치 수준의 Pseudo Label을 생성하고, 이를 통해 Student ViT에 조밀한 위치 기반 감독을 제공한다. 실험 결과, SUN은 ViT 기반 Few-Shot 분류에서 기존의 CNN SoTA 모델들을 능가하는 성능을 보여주었으며, 이는 향후 데이터가 부족한 환경에서 ViT를 활용하는 효율적인 학습 방법론으로서 중요한 가치를 지닌다.
