# AN INTEGRAL PROJECTION-BASED SEMANTIC AUTOENCODER FOR ZERO-SHOT LEARNING

William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot (2023)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL) 환경에서 발생하는 주요 문제들을 해결하고자 한다. ZSL의 핵심 목표는 학습 단계에서 본 적 없는 클래스(unseen classes)를 예측하거나 분류하는 것이다.

기존의 접근 방식들은 크게 두 가지 방향으로 나뉘지만 각각 명확한 한계를 가진다. 첫째, Embedding-based 방법론들은 시각적 특징 공간(visual feature space)을 시각적-의미적 공간(semantic space)으로 투영하는 함수를 학습한다. 그러나 이러한 방식은 학습 데이터와 테스트 데이터 간의 분포 차이로 발생하는 Domain Shift 문제에 취약하며, 고차원 데이터에서 특정 샘플이 많은 클래스의 근접 이웃으로 몰리는 Hubness 문제에 노출된다. 둘째, Generative-based 방법론들은 GAN이나 VAE를 통해 가상 데이터를 생성하여 문제를 지도 학습으로 전환하려 하지만, GAN의 학습 불안정성(divergence)이나 VAE의 결과물 흐릿함(blurriness) 같은 문제가 존재하며 모델의 가역성(invertibility)이 부족하여 추론과 재구성에 한계가 있다.

따라서 본 논문의 목표는 Domain Shift와 Hubness 문제를 완화하면서도, 안정적인 생성 능력을 갖춘 분석적 솔루션 기반의 Semantic Autoencoder 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 특징 공간($X$)과 의미적 공간($S$)을 단순 결합(concatenation)하여 입력 공간을 확장하고, 이를 잠재 표현 공간으로 투영한 뒤 다시 원래의 확장된 공간으로 재구성하는 **Integral Projection-based Semantic Autoencoder (IP-SAE)**를 설계한 것이다.

중심적인 직관은 입력 공간을 시각-의미 결합 공간으로 확장함으로써 잠재 표현 공간의 표현력을 높이고, 이를 통해 도메인에 불변하는(domain-invariant) 매니폴드를 학습하는 것이다. 또한, 반복적인 학습 과정 대신 Sylvester 방정식을 통한 분석적 해(analytical solution)를 도출함으로써 하이퍼파라미터 튜닝을 최소화하고 결과의 재현성을 높였다.

## 📎 Related Works

논문은 관련 연구를 크게 두 가지 범주로 분류하여 설명한다.

1. **Embedding space-based methods**: 시각적 특징과 의미적 특징 간의 호환성(compatibility)을 학습하거나 잠재 공간으로 투영하는 방식이다. ALE, DeViSE 등이 이에 해당하며, 시각적 매니폴드와 의미적 공간을 정렬하려 시도한다. 하지만 이러한 방법들은 여러 모달리티를 엔드투엔드로 통합하여 최적화하는 메커니즘이 부족하며, 앞서 언급한 Domain Shift와 Hubness 문제에 취약하다.
2. **Generative-based methods**: 의미적 표현으로부터 가상의 시각적 특징을 생성하여 분류기를 학습시키는 방식이다. f-VAEGAN-D2나 TF-VAEGAN 같은 모델들이 대표적이다. 이들은 Domain Shift를 완화할 수 있다는 장점이 있지만, GAN 계열은 학습이 불안정하고 VAE 계열은 생성된 샘플의 품질이 낮을 수 있다는 한계가 있다.

IP-SAE는 기존의 SAE(Semantic Autoencoder) 구조를 계승하면서도, 입력 공간을 확장하여 전사 함수(surjective function)의 성질을 확보함으로써 생성 모델의 안정성과 변별력을 동시에 확보했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

IP-SAE는 인코더(Encoder)와 디코더(Decoder)로 구성된 오토인코더 구조를 가진다. 인코더는 확장된 시각-의미 결합 공간을 잠재 표현 공간으로 투영하고, 디코더는 이를 다시 원래의 확장된 공간으로 재구성한다.

### 상세 방법론 및 방정식

**1. 입력 공간의 확장**
기존의 시각적 특징 $X$ 대신, 의미적 표현 $S$를 결합한 확장된 공간 $X'$를 정의한다.
$$X' = X \oplus S$$

**2. 최적화 목표 및 손실 함수**
모델은 시각적 공간 $X$와 의미적 공간 $S$ 사이의 손실을 최소화하는 가중치 $W$를 찾는 것을 목표로 한다.
$$\text{minimize}_W \|X - W^T W X\|_F^2 \quad \text{s.t. } WX = S$$
여기서 $F$는 Frobenius norm을 의미한다. 실제 구현을 위해 위 제약 조건을 완화(relax)하여 다음과 같은 소프트 제약 조건의 손실 함수를 사용한다.
$$\text{minimize}_W \|X - W^T S\|_F^2 + \lambda \|WX - S\|_F^2$$
이 식에서 $\lambda$는 디코더와 인코더 손실 간의 중요도를 조절하는 가중치 계수이다.

**3. 분석적 해 (Analytical Solution)**
위 식은 이차 형식(quadratic form)이므로 미분을 통해 전역 최적해를 구할 수 있으며, 이는 다음과 같은 **Sylvester 방정식** 형태로 정식화된다.
$$AW + WB = C$$
여기서 $A = SS^T$, $B = \lambda XX^T$, $C = (1 + \lambda)SX^T$ 이며, $A$와 $B$는 양의 준정부호(positive semi-defined) 행렬이다. 이 방정식은 Bartels-Stewart 알고리즘을 통해 효율적으로 해결할 수 있다.

**4. 정규화 및 추론**
Ridge regression의 개념을 도입하여 $\lambda$를 통해 정규화를 수행한다. 이는 특이 행렬(singular matrix) 문제를 해결하고, 시각적 공간의 노이즈를 줄이며, 도메인 간의 전이 가능성을 최적화하는 역할을 한다.

**5. ZSL 설정별 절차**

- **Standard ZSL**: 학습된 $W$를 이용해 unseen 클래스의 확장된 공간 프로토타입을 투영하고, 실제 시각적 공간과의 코사인 유사도를 측정하여 가장 유사한 클래스로 분류한다.
- **Generalized ZSL (GZSL)**: seen 클래스와 unseen 클래스가 동시에 존재하는 환경이다. $\lambda$ 정규화를 통해 seen bias(학습 데이터에 치우치는 현상)를 완화하고 도메인 shift 문제를 해결한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SUN, CUB-200-2011, AwA1, AwA2 총 4개의 벤치마크 데이터셋을 사용하였다.
- **백본**: 시각적 특징 추출을 위해 ResNet101을 사용하였다.
- **지표**: 클래스별 평균 Top-1 정확도(Average per-class accuracy), GZSL의 경우 Harmonic Mean ($H$), 그리고 생성 모델의 성능을 정밀하게 평가하기 위해 Precision과 Recall을 추가로 측정하였다.

### 정량적 결과

- **Standard ZSL**: 모든 데이터셋에서 기존 SOTA(State-of-the-art) 방법론들을 큰 차이로 압도하였다. 특히 AwA2에서 $94.4\%$, SUN에서 $94.4\%$ 등의 매우 높은 정확도를 기록하였다.
- **Generalized ZSL**: Seen 클래스에 대해서는 기존 모델들과 유사한 성능을 보였으나, Unseen 클래스에 대해서는 CUB와 SUN 데이터셋에서 유의미한 성능 향상을 보였다.

### 정성적 분석 및 시각화

- **t-SNE 분석**: 원래의 시각적 공간(original space)에서는 클래스 간 중첩이 심했으나, IP-SAE의 확장 공간(enhanced space)에서는 클래스 간 분리가 뚜렷하며 의미적 공간 주변으로 데이터가 잘 응집되어 Hubness 문제가 해결되었음을 확인하였다.
- **Confusion Matrix**: 확장 공간을 사용했을 때 주대각 성분이 훨씬 뚜렷하게 나타나, 모델이 unseen 클래스를 정확하게 식별하고 있음을 보여주었다.
- **Precision/Recall**: 확장 공간에서의 Recall이 Precision보다 높게 나타났는데, 이는 모델이 unseen 클래스의 분포를 충분히 커버하는 고품질의 샘플을 생성하고 있음을 시사한다.

## 🧠 Insights & Discussion

### 생성 능력의 수학적 근거

저자들은 인코더의 전사 함수(surjective function) 성질을 강조한다. 확장된 입력 공간을 사용함으로써 인코더는 모든 의미적 정보를 캡처할 수 있게 되며, 이는 디코더가 시각적 공간을 복원할 때 필요한 충분한 정보를 보존하게 만든다. 즉, 인코딩 단계에서 정보 보존과 디코딩 단계의 생성 능력 사이의 적절한 트레이드-오프(trade-off)를 $\lambda$를 통해 조절하는 것이 핵심이다.

### $\lambda$ 정규화의 역할

$\lambda$는 단순한 가중치가 아니라 Ridge 정규화로서 작동하여, 유사도 행렬의 조건수(condition number)를 개선하고 디코딩 과정의 노이즈를 줄인다. 이는 모델이 시각적 공간과 의미적 공간의 공유 주성분(shared principal components)을 더 잘 학습하도록 유도한다.

### 평가 지표에 대한 비판적 견해

본 논문은 기존 ZSL 연구들이 '평균 클래스 정확도'만을 보고하는 것에 대해 비판적이다. 정확도는 샘플 크기에 편향될 수 있으며 매니폴드의 공간적 배치를 정확히 반영하지 못한다. 따라서 생성 기반 ZSL에서는 지식 전이가 실제 분포를 얼마나 잘 커버하는지(Recall)와 생성된 프로토타입의 품질이 얼마나 실제와 유사한지(Precision)를 함께 측정해야 한다고 주장한다.

## 📌 TL;DR

본 논문은 시각적-의미적 공간을 결합한 확장 입력 공간을 사용하는 **IP-SAE** 모델을 제안하여 Zero-Shot Learning의 고질적인 문제인 Domain Shift와 Hubness 문제를 해결하였다. 특히 반복적인 학습 대신 Sylvester 방정식의 분석적 해를 사용하여 학습 안정성과 재현성을 확보하였으며, 4개의 벤치마크 데이터셋에서 SOTA를 경신하는 성능을 보였다. 이 연구는 다중 모달리티 공간의 확장과 정규화가 생성 기반 ZSL의 성능을 어떻게 향상시키는지 수학적으로 증명하였으며, 향후 레이블 의존도를 낮춘 대규모 인식 시스템 구축에 기여할 가능성이 크다.
