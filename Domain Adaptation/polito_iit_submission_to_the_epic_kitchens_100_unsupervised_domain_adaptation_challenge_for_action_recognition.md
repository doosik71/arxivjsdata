# PoliTO-IIT Submission to the EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition

Chiara Plizzari, Mirco Planamente, Emanuele Alberti, Barbara Caputo (2021)

## 🧩 Problem to Solve

본 논문은 EPIC-KITCHENS-100 데이터셋을 활용한 액션 인식(Action Recognition) 분야의 Unsupervised Domain Adaptation(UDA) 챌린지 해결 방안을 다룬다.

해결하고자 하는 핵심 문제는 소스 도메인과 타겟 도메인 사이에 존재하는 도메인 시프트(Domain Shift)이다. 특히 1인칭 시점(First-person) 비디오에서는 RGB 영상, 광학 흐름(Optical Flow), 오디오(Audio) 등 여러 모달리티를 사용하는데, 각 모달리티마다 도메인 시프트의 성격이 다르다는 점이 문제의 핵심이다. 예를 들어, RGB 영상은 환경 변화에 민감한 반면, Optical Flow는 외형보다는 움직임에 집중하므로 환경 변화에 상대적으로 강건하다. 또한 오디오 정보는 사용하는 도구의 재질(예: 플라스틱 도마 vs 나무 도마)에 따라 소리가 달라지므로 시각 정보와는 다른 차원의 시프트가 발생한다.

따라서 본 연구의 목표는 이러한 다중 모달리티의 특성을 효과적으로 결합하고, 레이블이 없는 타겟 데이터를 활용하여 도메인 간의 간극을 줄임으로써 타겟 도메인에서의 액션 인식 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 Domain Generalization(DG) 단계에서 시작하여 Unsupervised Domain Adaptation(UDA)을 거쳐, 최종적으로 여러 모델의 앙상블을 통한 일관성(Consistency)을 확보하는 단계적 접근 방식이다.

1. **RNA-Net의 확장**: 기존의 Audio-Visual 도메인 정렬 기법인 Relative Norm Alignment(RNA)를 Optical Flow 모달리티까지 확장하여, 타겟 데이터 없이도 다양한 소스 도메인에서 일반화된 특성을 추출할 수 있도록 설계하였다.
2. **UDA 프레임워크 통합**: RNA의 비지도 학습 특성을 이용하여 소스와 타겟 데이터를 모두 학습에 반영하는 UDA 구조로 확장하였으며, 여기에 $TA^3N$과 같은 최신 UDA 알고리즘을 결합하였다.
3. **앙상블 일관성 손실 함수 제안**: 서로 다른 백본 네트워크를 사용할 때 발생하는 예측 불확실성을 줄이기 위해, 모델 간의 특징 놈(Feature Norm)을 맞추는 Temporal Hard Norm Alignment(T-HNA)와 예측 결과의 일관성을 강제하는 Min-Entropy Consistency(MEC) 손실 함수를 도입하였다.

## 📎 Related Works

본 논문에서는 다음과 같은 관련 연구들을 참고하고 이를 확장하였다.

- **RNA-Net [11]**: 오디오와 비주얼 모달리티 간의 특징 놈(Feature Norm) 불균형이 특정 도메인으로의 편향을 야기한다는 점에 착안하여, 이를 정렬함으로써 도메인 불가지론적(Domain-agnostic) 특징을 추출하는 DG 기법이다.
- **$TA^3N$ [2]**: Temporal Adversarial Adaptation Network($TA^2N$)를 기반으로 도메인 주의 집중(Domain Attention) 메커니즘과 최소 엔트로피 정규화(Minimum Entropy Regularization)를 추가하여 비디오 도메인 적응을 수행하는 UDA 기법이다.
- **DANN [5]**: Gradient Reversal Layer(GRL)를 사용하여 도메인 분류기를 속임으로써 도메인 불변 특징을 학습하는 기본적인 적대적 학습 방식이다.

기존 방식들이 단일 모달리티나 특정 UDA 기법에 의존했다면, 본 연구는 DG $\rightarrow$ UDA $\rightarrow$ Ensemble Consistency로 이어지는 파이프라인을 통해 다중 소스 데이터의 이점과 다중 백본의 다양성을 모두 활용했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Domain Generalization (DG)

타겟 데이터에 접근하기 전, 여러 소스 도메인에서 공통된 특징을 학습하기 위해 Relative Norm Alignment(RNA)를 사용한다. RNA 손실 함수 $L_{RNA}$는 오디오와 비주얼 특징 놈의 평균 거리 차이를 최소화하여 모달리티 간의 불균형을 해소한다.

$$L_{RNA} = \left( \frac{\mathbb{E}[h(X^v)]}{\mathbb{E}[h(X^a)]} - 1 \right)^2$$

여기서 $h(x_i^m) = (\|\cdot\|_2 \circ f_m)(x_i^m)$는 $m$번째 모달리티의 특징 $f_m$에 대한 $L_2$-norm을 의미하며, $\mathbb{E}[\cdot]$는 해당 모달리티 샘플들에 대한 평균값이다. 본 연구에서는 이를 Optical Flow 모달리티까지 확장하여 적용하였다.

### 2. Unsupervised Domain Adaptation (UDA)

UDA 설정에서는 소스 데이터와 레이블이 없는 타겟 데이터를 모두 사용하여 학습한다. 이를 위해 $L_{RNA}$를 다음과 같이 재정의한다.

$$L_{RNA} = L_{s}^{RNA} + L_{t}^{RNA}$$

여기서 $L_{s}^{RNA}$는 소스 데이터에, $L_{t}^{RNA}$는 타겟 데이터에 적용된 RNA 손실이다. 또한, $TA^3N$ 기법을 통합하여 TRM(Temporal Relation Module)의 다중 스케일 특징을 정렬하고, 도메인 주의 집중 메커니즘과 최소 엔트로피 정규화를 통해 분류기를 정교화한다.

### 3. Ensemble UDA Losses

다양한 백본(I3D, BN-Inception, ResNet50 등)을 앙상블할 때, 각 모델이 독립적으로 적응하여 발생하는 예측 값의 불일치를 해결하기 위해 두 가지 일관성 손실 함수를 사용한다.

- **Temporal Hard Norm Alignment (T-HNA)**: 서로 다른 백본 네트워크 $b$에서 추출된 특징 놈을 특정 상수 값 $R$로 정렬하여 각 모델의 기여도를 재균형화한다.
    $$L_{T-HNA} = \sum_{b} (\mathbb{E}[h_t(X^b)] - R)^2$$
- **Min-Entropy Consensus (MEC)**: 여러 모델이 타겟 데이터에 대해 일관된 예측을 하도록 유도한다.
    $$L_{MEC} = -\frac{1}{m} \sum_{i=1}^{m} \frac{1}{b} \max_{y \in Y} \sum_{b} \log p_b(y|x_t^i)$$
    여기서 $m$은 배치 크기, $p_b$는 $b$번째 백본의 예측 확률이다.

### 4. 시스템 구조 및 구현

- **백본 네트워크**: RGB 및 Flow 스트림에는 I3D(Kinetics 사전학습)를, 오디오 스트림에는 BN-Inception(ImageNet 사전학습)을 사용하였다. 또한 ResNet50 기반의 TSN/TSM 모델도 활용하였다.
- **융합 전략**: RNA-Net에서는 최종 예측 점수를 평균 내는 Late Fusion을 사용하였고, 다른 설정에서는 TRM을 통해 프레임 임베딩을 집계하는 Mid-fusion 전략을 채택하였다.

## 📊 Results

### 1. 챌린지 결과 (Leaderboard)

본 연구의 제출물('plnet')은 EPIC-KITCHENS-100 UDA 챌린지에서 다음과 같은 성과를 거두었다.

- **Verb (동사)**: 1위 달성
- **Noun (명사) & Action (동작)**: 3위 달성
- **Top-5 Accuracy**: 모든 카테고리에서 1위 달성

### 2. Ablation Study (앙상블 손실 함수 효과)

공식 검증 세트(Validation Set)에서 앙상블 손실 함수의 기여도를 분석한 결과, 단순 앙상블보다 T-HNA와 MEC를 추가했을 때 Top-1 정확도가 모든 카테고리에서 최대 2% 향상됨을 확인하였다.

### 3. DG 성능 분석

타겟 데이터 없이 소스 데이터만 사용한 DG 설정에서의 결과는 다음과 같다.

- RNA-Net이 Baseline(Source Only) 대비 Top-1에서 최대 3%, Top-5에서 최대 10% 향상된 성능을 보였다.
- RNA-Net은 타겟 데이터에 접근하지 않은 상태의 $TA^3N$보다 우수한 성능을 보였으며, $TA^3N$과 결합했을 때 추가적인 성능 향상이 나타나 두 기법의 상보적 관계를 입증하였다.

## 🧠 Insights & Discussion

본 논문은 다중 모달리티를 사용하는 액션 인식 환경에서 도메인 시프트를 해결하기 위해 매우 체계적인 접근 방식을 취하였다. 특히 단순히 하나의 UDA 알고리즘에 의존하지 않고, **DG $\rightarrow$ UDA $\rightarrow$ Ensemble Consistency**라는 3단계 전략을 통해 성능을 단계적으로 끌어올린 점이 인상적이다.

RNA-Net을 통해 모달리티 간 특징 놈을 정렬하는 것이 도메인 불가지론적 특징 추출에 효과적임을 보였으며, 이는 특히 다중 소스 데이터가 존재할 때 강력한 힘을 발휘한다. 또한, 앙상블 시 발생하는 모델 간의 불일치 문제를 T-HNA와 MEC라는 일관성 손실 함수로 해결하려 한 시도는 실무적인 관점에서 매우 유용한 접근이다.

다만, 본 보고서는 챌린지 제출을 위한 기술 보고서 형식이므로, 제안한 방법론이 EPIC-KITCHENS-100 외의 다른 데이터셋에서도 동일하게 일반화될 수 있는지에 대한 광범위한 실험적 검증은 명시되지 않았다. 또한, 다양한 백본을 앙상블함으로써 발생하는 연산 복잡도 증가 문제에 대한 논의가 부족하다는 점이 한계로 보인다.

## 📌 TL;DR

본 연구는 EPIC-KITCHENS-100 UDA 챌린지에서 **Relative Norm Alignment(RNA)**를 확장한 DG 기법, **$TA^3N$** 기반의 UDA, 그리고 **T-HNA 및 MEC** 일관성 손실 함수를 결합한 3단계 파이프라인을 제안하였다. 이를 통해 RGB, Flow, Audio 모달리티 간의 도메인 시프트를 효과적으로 극복하였으며, 결과적으로 'Verb' 부문 1위를 포함해 최상위권의 성적을 거두었다. 이 연구는 다중 모달리티 기반의 비지도 도메인 적응 및 모델 앙상블 전략 수립에 중요한 가이드라인을 제공한다.
