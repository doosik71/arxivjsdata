# Continual Learning Improves Zero-Shot Action Recognition

Shreyank N Gowda, Davide Moltisanti, and Laura Sevilla-Lara (2024)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Action Recognition (ZSAR)에서 발생하는 일반화 성능 저하와 **Catastrophic Forgetting**(치명적 망각) 문제를 해결하고자 한다. ZSAR의 핵심 목표는 학습 과정에서 본 적 없는 새로운 클래스(unseen classes)를 인식하는 것인데, 이를 위해 모델을 미세 조정(fine-tuning)하는 과정에서 이전에 학습한 지식이나 사전 학습(pre-training) 단계에서 얻은 일반화 능력이 손실되는 현상이 발생한다.

특히, 일반적인 Zero-Shot Learning (ZSL) 설정보다 더 까다로운 **Generalized Zero-Shot Learning (GZSL)** 설정에서는 모델이 새로운 클래스를 배우면서도 기존의 seen 클래스에 대한 지식을 동시에 유지해야 한다. 저자들은 Continual Learning (CL)의 목표인 '이전 지식을 잊지 않고 새로운 작업을 학습하는 것'이 ZSL의 '일반화 능력 향상'이라는 목표와 매우 밀접하게 정렬되어 있다는 점에 주목하여, 비디오 행동 인식 분야에 CL 패러다임을 최초로 도입하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Generative Iterative Learning (GIL)**이라는 새로운 프레임워크를 제안한 것이다. GIL의 중심 아이디어는 CL의 **Generative Replay** 기법을 ZSAR에 적용하여, 과거 클래스의 합성된 특징(synthesized features)과 새로운 클래스의 실제 특징(real features)을 결합해 모델을 점진적으로 학습시키는 것이다.

GIL의 주요 설계 직관은 다음과 같다:

1. **Replay Memory의 도입**: 과거 클래스의 대표성 있는 특징(Prototype)과 분포(Noise)를 저장하여 모델이 이전 지식을 유지하게 한다.
2. **점진적 학습(Incremental Learning)**: 새로운 데이터를 한꺼번에 학습시키지 않고 소량의 배치(10%씩)로 나누어 점진적으로 도입함으로써 모델의 급격한 가중치 변화를 막고 일반화 성능을 유지한다.
3. **합성 특징의 활용**: 실제 데이터 대신 생성 모델을 통해 만들어진 합성 특징을 사용하여 학습 데이터의 균형을 맞추고, 특징 공간을 더 조밀하게(compact) 만들어 학습 효율을 높인다.

## 📎 Related Works

**1. Zero-Shot Action Recognition (ZSAR)**
기존 연구들은 주로 비디오 특징과 시맨틱 레이블 사이의 공유 임베딩 공간을 구축하거나, GAN을 이용해 unseen 클래스의 시각적 특징을 생성하여 분류기를 학습시키는 방식에 집중했다. 최근에는 CLIP과 같은 Vision-Language 기초 모델을 활용하여 제로샷 성능을 높이는 시도가 많았으나, 이러한 방식들은 대부분 정적인 특징 조작이나 대조 학습에 치중되어 있으며, 학습 과정에서의 망각 문제나 점진적 지식 습득 과정을 고려하지 않았다.

**2. Continual Learning (CL)**
CL은 새로운 데이터를 학습할 때 발생하는 Catastrophic Forgetting을 억제하는 것이 주 목표이다. Weight Regularization(예: EWC)이나 Memory Augmentation 같은 기법들이 제안되었으며, 특히 Generative Replay는 과거 데이터를 생성하여 현재 데이터와 섞어 학습함으로써 망각을 방지한다.

**3. 차별점**
기존의 Continual Zero-Shot Learning (CZSL) 연구들은 주로 이미지 도메인에 국한되어 있었으며, 비디오 행동 인식 분야에서 CL을 적용한 사례는 본 논문이 처음이다. 또한, GIL은 개별 인스턴스 수준의 특징이 아닌 **Class Prototype**과 **Noise** 기반의 생성을 통해 효율성과 안정성을 확보했다는 점에서 기존 생성 기반 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

GIL은 크게 세 단계(**Initialization $\rightarrow$ Incremental Learning $\rightarrow$ Update**)의 반복적인 사이클로 구성된다.

#### 1. 주요 구성 요소

- **Foundation Model ($M$)**: 비디오 인코더(기본적으로 X-Florence 사용). 백본은 고정하고 마지막 두 개의 Fully-Connected (FC) 레이어만 학습시킨다.
- **Replay Memory**:
  - **Buffer**: 각 클래스의 Prototype ($\mu$)과 Noise ($\sigma$)를 저장한다.
  - **Semantic-to-Visual Encoder ($E$)**: CVAE 기반으로, 시맨틱 임베딩을 입력받아 해당 클래스의 $\mu$와 $\sigma$를 예측한다.
  - **Feature Generator ($F$)**: $\mu$와 $\sigma$를 입력받아 $M$이 생성하는 특징과 유사한 시각적 특징을 생성하는 GAN 기반 네트워크이다.

#### 2. 학습 절차 및 방정식

**A. Initialization Stage**
사전 학습 데이터셋 $\mathcal{P}$를 사용하여 각 클래스의 프로토타입 $\mu$와 표준편차 $\sigma$를 계산하여 버퍼에 저장한다. 이후 CVAE($E$)와 생성기($F$)를 학습시킨다. $F$의 학습을 위한 목적 함수는 다음과 같다:
$$L_D = \mathbb{E}_{(x,a) \sim p(x_S \times a_S)} [G(x, H(x))] - \mathbb{E}_{z \sim p_z} \mathbb{E}_{a \sim p_a} [G(F(a, z), a)] - \alpha \mathbb{E}_{z \sim p_z} \mathbb{E}_{a \sim p_a} (\|\nabla_{\hat{x}} G(F(a, z))\|^2 - 1)^2$$
최종적으로 분류 손실 $L_{CLS}$와 상호 정보량 손실 $L_{MI}$를 결합하여 최적화한다:
$$\min_F \min_H \max_G L_D + \lambda_1 L_{CLS}(F) + \lambda_2 L_{MI}(F)$$

**B. Incremental Learning Stage**
비디오 모델 $M$의 마지막 두 레이어를 미세 조정한다. 이때 학습 데이터는 다음 두 가지의 혼합으로 구성된다:

- **합성 특징 ($\hat{f}$)**: 버퍼에 저장된 과거 클래스들에 대해 $F$를 통해 생성한 특징.
- **실제 특징 ($f$)**: 현재 단계에서 도입된 새로운 클래스들의 실제 비디오 특징.
이 과정에서 새로운 클래스는 전체의 10%씩 점진적으로 추가된다.

**C. Update Stage**
새로 학습한 클래스의 프로토타입과 노이즈를 버퍼에 추가하고, CVAE($E$)를 업데이트하여 새로운 시맨틱 임베딩으로부터 이들을 생성할 수 있게 한다. $F$는 고정된 상태를 유지한다.

### 추론 및 테스트 (Testing)

1. Unseen 클래스의 시맨틱 임베딩을 $E \rightarrow F$ 순으로 통과시켜 합성 특징을 생성한다.
2. 생성된 특징을 사용하여 $M$을 최종 미세 조정한다 (ZSL의 경우 unseen만, GZSL의 경우 seen $\cup$ unseen 모두 사용).
3. 테스트 비디오가 입력되면 $M$을 통해 임베딩을 추출하고, 임베딩 공간에서 **Nearest Neighbor (K-NN, K=1)** 검색을 통해 가장 가까운 클래스로 분류한다.

## 📊 Results

### 실험 설정

- **데이터셋**: HMDB-51, UCF-101, Kinetics-600 (Kinetics-200 split).
- **백본 모델**: X-Florence, X-CLIP, Vita-CLIP 등 다양한 모델을 적용하여 모델 불가지론적(model-agnostic) 특성을 검증했다.
- **지표**: Top-1 Accuracy, Harmonic Mean (GZSL의 경우 seen/unseen 정확도의 조화 평균).

### 주요 결과

1. **Catastrophic Forgetting 억제**: GIL을 사용하지 않았을 때 사전 학습 데이터(Kinetics)에 대한 성능이 급격히 떨어지는 반면, GIL을 적용하면 성능 하락이 크게 줄어들었다 (최대 24.7% 향상).
2. **ZSL 성능**: UCF-101과 HMDB-51에서 기존 SOTA 모델들을 능가하는 성능을 보였으며, 특히 Florence 백본 사용 시 UCF-101에서 $79.4\%$의 높은 정확도를 기록했다.
3. **GZSL 성능**: 가장 도전적인 GZSL 설정에서도 기존 방법론 대비 최대 $19.7\%$의 성능 향상을 이루어내며 새로운 SOTA를 달성했다.

### Ablation Study 및 분석

- **Replay Memory의 영향**: 메모리가 없을 때보다 프로토타입과 노이즈를 함께 저장했을 때 성능이 가장 높았으며, 결과의 편차(std)가 줄어들어 강건함이 증가했다.
- **합성 데이터 vs 실제 데이터**: 사전 학습 클래스에 대해 실제 데이터를 사용하는 것보다 **합성 데이터(Synthetic Only)**를 사용하는 것이 성능이 더 좋았다. t-SNE 분석 결과, 합성 특징이 실제 특징보다 더 조밀한(compact) 클러스터를 형성하여 모델 학습에 유리함이 확인되었다.
- **샘플링 비율**: 새로운 데이터를 1%~10% 정도로 천천히 도입할 때 최적의 성능을 보였으며, 50% 이상 한꺼번에 학습시키면 성능이 저하되었다. 이는 '너무 빠른 학습'이 일반화 능력을 해친다는 것을 시사한다.

## 🧠 Insights & Discussion

본 연구는 ZSAR에서 발생하는 망각 문제가 단순한 데이터 부족의 문제가 아니라, 학습 과정에서의 최적화 경로와 일반화 능력의 손실에서 기인한다는 점을 밝혀냈다.

**강점**:

- CL의 Generative Replay를 비디오 도메인에 성공적으로 이식하여, 기존의 정적인 ZSL 방식보다 훨씬 유연하고 강력한 일반화 성능을 확보했다.
- 특정 백본에 의존하지 않고 다양한 기초 모델에 적용 가능한 프레임워크임을 입증했다.
- 합성 데이터가 실제 데이터보다 더 정제된(compact) 표현을 제공한다는 통찰을 통해, 왜 생성 기반 접근법이 ZSL에서 효과적인지를 시각적으로 증명했다.

**한계 및 논의**:

- 합성 특징에 의존하므로, 생성 모델이 만들어낸 특징에 편향(bias)이 있을 경우 실제 데이터와의 괴리가 발생할 가능성이 있다.
- 학습 데이터와 테스트 데이터의 시점이 매우 다르거나 뷰포인트(viewpoint) 변화가 극심한 데이터셋에서의 효과는 명확히 검증되지 않았다.
- 생성기 $F$를 고정하는 것이 최적화에 유리하다는 결과가 나왔으나, 더 복잡한 데이터셋에서는 생성기의 동적 업데이트가 필요할 수도 있다.

## 📌 TL;DR

본 논문은 비디오 행동 인식의 Zero-Shot 설정에서 발생하는 지식 망각 문제를 해결하기 위해 **Continual Learning(CL)** 기반의 **Generative Iterative Learning (GIL)** 프레임워크를 제안한다. GIL은 Replay Memory를 통해 과거 클래스의 프로토타입을 보존하고, 생성 모델로 만든 합성 특징과 새로운 실제 특징을 섞어 점진적으로 모델을 학습시킴으로써 일반화 성능을 극대화한다. 실험 결과, UCF-101, HMDB-51 등 주요 벤치마크에서 ZSL 및 GZSL 모두 새로운 SOTA를 달성했으며, 이는 향후 Few-shot이나 Semi-supervised 비디오 학습 연구에도 중요한 기초가 될 것으로 기대된다.
