# Video Unsupervised Domain Adaptation with Deep Learning: A Comprehensive Survey

Yuecong Xu, Haozhi Cao, Lihua Xie, Xiaoli Li, Zhenghua Chen, Jianfei Yang (2024)

## 🧩 Problem to Solve

본 논문은 비디오 분석 작업, 특히 행동 인식(Action Recognition) 분야에서 발생하는 **Domain Shift** 문제를 해결하기 위한 Video Unsupervised Domain Adaptation (VUDA) 기술의 최신 동향을 분석한다. 딥러닝 기반의 비디오 모델은 대규모 공개 데이터셋(Source Domain)에서 학습되지만, 이를 실제 환경(Target Domain)에 적용했을 때 데이터 분포의 차이로 인해 성능이 급격히 저하되는 문제가 발생한다.

비디오 데이터의 경우, 매 프레임에 대한 레이블을 지정하는 비용(Annotation Cost)이 매우 높기 때문에, 타겟 도메인의 레이블 없이 학습하는 Unsupervised Domain Adaptation (UDA) 방식이 매우 실용적이다. 본 논문의 목표는 소스 도메인의 레이블된 데이터와 타겟 도메인의 레이블되지 않은 데이터를 활용하여, 도메인 간의 간극을 줄이고 모델의 일반화 능력(Generalizability)과 이식성(Portability)을 향상시키는 VUDA 방법론들을 체계적으로 정리하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 기반의 VUDA 연구를 체계적으로 분류하고 분석한 최초의 종합 서베이 논문이라는 점에 있다. 주요 기여 사항은 다음과 같다.

- **VUDA의 체계적 분류**: VUDA를 크게 **Closed-set VUDA**와 **Non-closed-set VUDA**로 구분하고, 각 설정에 따른 세부 방법론을 분류하였다.
- **Closed-set 방법론의 5가지 범주화**: 소스-타겟 도메인을 정렬하는 방식에 따라 Adversarial-based, Discrepancy-based, Semantic-based, Reconstruction-based, Composite 방법으로 세분화하여 분석하였다.
- **Non-closed-set 시나리오 확장**: 레이블 공간의 제약(Partial/Open-set), 소스 데이터 접근성(Source-free/Black-box), 타겟 데이터 가용성(Zero-shot/VDG) 등 실제 환경에서 발생 가능한 다양한 제약 조건을 정의하고 관련 연구를 정리하였다.
- **벤치마크 데이터셋 및 백본 분석**: VUDA 연구에 사용되는 다양한 데이터셋을 정리하고, I3D, TRN 등 사용된 Backbone 네트워크가 성능에 미치는 영향을 분석하였다.

## 📎 Related Works

기존의 Domain Adaptation (DA) 및 UDA 연구들은 주로 이미지 데이터(Image-UDA)에 집중되어 왔다. 하지만 비디오 데이터는 이미지와 달리 **Temporal Feature(시간적 특징)**와 광학 흐름(Optical Flow), 오디오(Audio)와 같은 **Multi-modality** 특성을 가지고 있다.

단순히 이미지 UDA 모델의 2D CNN을 3D CNN으로 교체하는 방식(Vanilla substitution)으로는 비디오의 복잡한 도메인 시프트를 해결하기 어렵다. 따라서 본 논문은 이미지 UDA와의 차이점을 명확히 하며, 비디오 특유의 시간적-공간적 정렬(Spatio-temporal alignment)이 필수적임을 강조한다. 또한, 전이 학습(Transfer Learning)의 특수한 사례로서 DA를 다룬 기존 서베이들과 달리, 오직 '비디오' 작업에 특화된 UDA를 심도 있게 다룬다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. VUDA의 정의 및 이론적 배경
VUDA의 목표는 레이블된 소스 도메인 $\mathcal{D}_s$에서 학습된 모델을 레이블되지 않은 타겟 도메인 $\mathcal{D}_t$에 적응시켜, 타겟 도메인의 경험적 위험(Empirical Target Risk) $\epsilon_t$를 최소화하는 것이다. 도메인 적응 이론에 따르면 $\epsilon_t$는 다음 세 가지 항목의 합으로 상한선(Upper-bound)이 결정된다.
1. 소스-타겟 도메인 모두에서 이상적인 공동 가설(Ideal joint hypothesis)의 결합 오류
2. 소스 도메인에서의 경험적 오류(Empirical source-domain error)
3. 두 도메인 간의 거리(Divergence)

대부분의 VUDA 방법론은 세 번째 항인 **도메인 간 거리**를 줄이는 데 집중한다.

### 2. Closed-set VUDA 방법론
Closed-set VUDA는 소스-타겟 도메인이 동일한 레이블 공간을 공유하며, 학습 시 두 데이터에 모두 접근 가능하다는 가정을 가진다.

- **Adversarial-based**: 도메인 판별자(Domain Discriminator)를 도입하여, 판별자가 데이터의 출처를 구분하지 못하도록 특징 추출기를 학습시킨다. 이는 특징 공간에서 두 도메인의 분포를 일치시키는 방식이다.
- **Discrepancy-based (Metric-based)**: MMD(Maximum Mean Discrepancy)나 CORAL과 같은 지표를 사용하여 두 도메인 간의 통계적 거리를 직접 계산하고 이를 최소화한다.
- **Semantic-based**: 두 도메인이 공유하는 의미적 특성(Spatio-temporal association, Feature clustering, Modality correspondence)을 활용한다. 예를 들어, Contrastive Learning을 통해 동일 클래스의 특징들이 도메인에 상관없이 가깝게 위치하도록 유도한다.
- **Reconstruction-based**: Encoder-Decoder 구조를 사용하여 데이터를 재구성하는 목표(Reconstruction objective)를 통해 도메인 불변 특징(Domain-invariant features)을 추출한다. VAE(Variational AutoEncoder) 등이 활용된다.
- **Composite**: 위 방법론들의 장점을 결합한 형태이다. 예를 들어, Adversarial loss와 Semantic loss를 동시에 최적화하여 안정성과 성능을 모두 확보한다.

### 3. Non-closed-set VUDA 시나리오
실제 환경의 제약을 반영하여 다음과 같은 확장 시나리오를 다룬다.

- **Label Space Constraint**:
    - **Partial-set (PVDA)**: 타겟 레이블 공간이 소스의 부분집합인 경우 ($\mathcal{Y}_t \subset \mathcal{Y}_s$). 소스에만 존재하는 Outlier 클래스로 인한 Negative Transfer를 막는 것이 핵심이다.
    - **Open-set (OSVDA)**: 타겟에 소스에 없는 새로운 클래스가 존재하는 경우 ($\mathcal{Y}_s \subset \mathcal{Y}_t$).
- **Source Data Assumption**:
    - **Source-free (SFVDA)**: 개인정보 보호 등의 이유로 소스 데이터 없이 학습된 모델만 제공되는 경우.
    - **Black-box (BVDA)**: 소스 모델의 파라미터조차 알 수 없고 API 형태로만 예측값을 얻을 수 있는 경우.
- **Target Data Assumption**:
    - **Zero-shot / Video Domain Generalization (VDG)**: 적응 단계에서 타겟 데이터를 전혀 볼 수 없는 경우.
- **Cross-domain Tasks**: 단순 행동 인식을 넘어 비디오 시맨틱 세그멘테이션, VQA(Video Quality Assessment) 등으로 확장한다.

## 📊 Results

### 1. 데이터셋 분석
본 논문은 UCF-Olympic, UCF-HMDB 등을 포함한 다양한 벤치마크 데이터셋을 정리하였다. 특히, 가상 세계와 현실 세계를 잇는 Kinetics-Gameplay나, 저조도 환경을 다루는 HMDB-ARID와 같이 **Large Domain Shift**가 발생하는 데이터셋의 중요성을 강조한다.

### 2. 백본 네트워크(Backbone)의 영향
실험 결과, VUDA 성능은 사용된 백본 네트워크에 큰 영향을 받는다.
- **I3D**: 가장 널리 사용되는 백본으로, 구현이 쉽고 Kinetics-400 사전 학습 모델의 성능이 뛰어나 generalizability가 높다.
- **TRN**: 시간적 관계 추론(Temporal relation reasoning)에 강점이 있어 효율적이지만, VDG 시나리오에서는 일반화 성능이 떨어지는 경향이 있다.
- **결론**: 단순히 모델의 복잡도를 높이는 것보다, 시간적 특징을 효과적으로 추출하고 정렬할 수 있는 구조(TSM, TRN 등)를 사용하는 것이 VUDA 성능 향상에 결정적이다.

## 🧠 Insights & Discussion

### 강점 및 성과
최근의 VUDA 연구는 단순히 이미지 UDA를 모방하는 수준을 넘어, 비디오의 **Multi-modality(RGB, Optical Flow, Audio)**와 **Temporal Dynamics**를 명시적으로 정렬하는 방향으로 진화하였다. 또한, Source-free나 Zero-shot과 같은 현실적인 제약 조건을 다루는 연구들이 등장하며 실용성이 높아졌다.

### 한계 및 미래 방향
- **시나리오의 확장**: Multi-target VUDA나 Active Learning을 결합한 Active VUDA 연구가 여전히 부족하다.
- **도메인 시프트의 종류**: 현재 대부분의 연구는 Covariate Shift(입력 분포의 변화)에만 집중하고 있으며, Label Shift나 Conditional Shift(개념 표류)에 대한 대응이 필요하다.
- **모달리티의 확장**: RGB 외에 Human Skeleton, Depth, Lidar 데이터 등을 활용한 VUDA 연구가 필요하며, Transformer 및 GNN 기반의 백본 도입이 가속화되어야 한다.
- **이론적 기반 부족**: 현재의 VUDA 방법론들은 대부분 경험적(Empirical) 결과에 의존하고 있다. 비디오 특성에 맞는 도메인 적응 이론(Domain Adaptation Theory)이 확립되어야 모델의 설명 가능성(Explainability)이 확보될 수 있다.
- **LLVM의 활용**: CLIP과 같은 Large Language-Vision Models를 활용하여 Zero-shot 성능을 극대화하는 방향이 유망하지만, 비디오의 세밀한 동작(Action)을 구분하는 능력은 여전히 개선 대상이다.

## 📌 TL;DR

본 논문은 딥러닝 기반의 **비디오 비지도 도메인 적응(VUDA)** 연구를 종합적으로 분석한 서베이 논문이다. VUDA를 Closed-set과 Non-closed-set으로 구분하고, 각각의 정렬 방법론(Adversarial, Discrepancy, Semantic, Reconstruction, Composite)과 제약 시나리오(Partial, Source-free, Zero-shot 등)를 체계적으로 분류하였다. 비디오 특유의 시간적 특징과 다중 모달리티 정렬의 중요성을 강조하며, 향후 연구 방향으로 이론적 기반 확립, LLVM 활용, 다양한 센서 데이터 확장 등을 제시한다. 이 연구는 비디오 모델을 실제 환경에 적용하려는 연구자들에게 포괄적인 가이드라인을 제공한다.