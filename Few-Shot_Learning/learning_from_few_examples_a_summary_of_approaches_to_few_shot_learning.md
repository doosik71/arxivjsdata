# LEARNING FROM FEW EXAMPLES: A SUMMARY OF APPROACHES TO FEW-SHOT LEARNING

Archit Parnami and Minwoo Lee

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 대량의 학습 데이터와 높은 계산 자원을 요구하지만, 실제 세계에서는 데이터 수집, 전처리 및 레이블링의 어려움, 프라이버시 문제 등으로 인해 소량의 데이터만 사용 가능한 "Few-Shot Learning (FSL)" 문제를 다룹니다. 특히, 소수의 학습 예제만으로 모델이 새로운 태스크에 효과적으로 일반화하도록 학습하는 방법을 체계적으로 요약하고 분석하는 것을 목표로 합니다. 인간이 몇 가지 예시만으로 새로운 개념을 빠르게 학습하는 능력에 비해, 기존 기계 학습 모델은 이러한 능력에서 한계를 보입니다.

## ✨ Key Contributions

이 설문 논문은 Few-Shot Learning (FSL) 분야의 다양한 접근 방식을 체계적으로 분류하고 요약하여 다음의 핵심 기여를 제공합니다:

- **FSL 접근 방식의 포괄적인 분류:** FSL 방법을 메타 학습 기반(Meta-Learning-based FSL)과 비메타 학습 기반(Non-Meta-Learning-based FSL)으로 크게 분류하고, 메타 학습 기반 FSL을 다시 Metric-based, Optimization-based, Model-based 접근 방식으로 세분화합니다.
- **하이브리드 FSL 시나리오 분석:** Cross-Modal, Semi-Supervised, Generalized, Generative, Cross-Domain, Transductive, Unsupervised, Zero-Shot Learning 등 다양한 FSL 문제 변형과 이에 대한 하이브리드 접근 방식을 소개합니다.
- **최신 SOTA(State-Of-The-Art) 방법론 요약:** 각 범주 내에서 2020년 1월까지 제안된 대표적인 FSL 알고리즘들을 상세히 설명합니다.
- **진행 상황 및 과제 논의:** FSL 분야의 현재까지의 발전 추세와 함께 실제 적용 시 발생할 수 있는 주요 도전 과제 및 미해결 문제들을 제시합니다.

## 📎 Related Works

본 논문은 FSL 분야를 요약하려는 이전 연구들로부터 영감을 받았습니다. 주요 관련 연구들은 다음과 같습니다:

- **Wang et al. [4]:** FSL의 핵심 문제를 신뢰할 수 없는 경험적 위험 최소화(empirical risk minimizer)로 정의하며 문제의 어려움을 강조했습니다.
- **Chen et al. [5]:** 여러 대표적인 few-shot 분류 알고리즘에 대한 비교 분석을 제공했습니다.
- **Weng [6]:** FSL에 대한 메타 학습 접근 방식을 논의했습니다. 본 논문은 Weng의 작업을 확장하여 비메타 학습 및 하이브리드 메타 학습 접근 방식을 추가로 다루고, 최신 메타 학습 방법론을 포함합니다.

## 🛠️ Methodology

본 논문은 Few-Shot Learning (FSL)의 접근 방식을 크게 두 가지 범주로 나누어 설명합니다. 대부분의 접근 방식은 이미지 분류 문제 해결을 염두에 두고 개발되었으나, 회귀, 객체 탐지, 세분화 등 다른 문제에도 적용 가능합니다.

### 1. 메타 학습 기반 FSL (Meta-Learning-based Few-Shot Learning)

메타 학습(Meta-Learning)은 여러 태스크($T$)에 걸쳐 "학습하는 방법"($f(D^{\text{train}}_i, x; \theta)$)을 학습하는 것을 목표로 하며, 새로운 Few-Shot 태스크에 빠르게 적응할 수 있는 사전 지식($\theta$)을 습득합니다. 학습은 에피소드 방식으로 진행됩니다 (Algorithm 1 참조).

- **1.1. Metric-based Meta-Learning (거리 기반 메타 학습)**

  - **핵심 아이디어:** 데이터 샘플 간의 거리를 학습하여 유사한 샘플은 가깝게, 다른 샘플은 멀게 임베딩 공간에 매핑합니다. 쿼리 이미지($\hat{x}$)를 가장 가까운 지원 클래스($v_c$)에 할당하여 분류합니다.
  - **방법:** 임베딩 함수 $g_{\theta_1}$ (신경망) 또는 임베딩 함수와 거리 함수 $d_{\theta_2}$ (다른 신경망)를 모두 학습합니다.
  - **예시:**
    - **Siamese Networks [20]:** 동일한 CNN 두 개를 사용하여 두 이미지의 유사도 점수를 출력합니다. L1 거리를 사용합니다.
    - **Matching Networks [13]:** 어텐션 커널 $a(\hat{x}, x_k)$을 사용하여 쿼리($\hat{x}$)와 지원($x_k$) 임베딩 간의 코사인 유사도를 측정하고, 이를 기반으로 가중치를 부여하여 예측합니다 (Eq. 7, 8).
    - **Prototypical Networks [21]:** 각 클래스의 프로토타입($v_c$)을 해당 클래스 지원 이미지 임베딩의 평균으로 정의하고 (Eq. 9), 쿼리 임베딩과 프로토타입 간의 유클리드 거리에 소프트맥스를 적용하여 예측합니다 (Eq. 10).
    - **Relation Networks [22]:** 유사도 측정 대신 별도의 CNN($d_{\theta_2}$)을 학습하여 쿼리 임베딩과 클래스 프로토타입 임베딩의 연결($\oplus$)에 대한 관계 점수($r_c$)를 출력합니다.
    - **TADAM [16]:** 학습 가능한 소프트맥스 온도 $\lambda$와 태스크 임베딩 네트워크(Task Embedding Network, TEN)를 사용하여 태스크 적응형 임베딩 $g_{\theta_1}(x, \Gamma)$을 생성합니다.
    - **TapNet [23]:** 클래스별 참조 벡터 $\Phi$와 태스크 종속 투영 공간 $M$을 학습하여 태스크 임베딩된 특징과 참조 간의 불일치를 줄입니다.
    - **CTM [24]:** Category Traversal Module (CTM)을 사용하여 지원 및 쿼리 세트 이미지에 대한 문맥 임베딩(contextual embeddings)을 생성하고, 현재 태스크에 적합한 특징을 학습합니다.
    - **Attention-based methods [32, 33]:** 어텐션 모듈을 통합하여 더 차별적인 특징 임베딩을 학습합니다.

- **1.2. Optimization-based Meta-Learning (최적화 기반 메타 학습)**

  - **핵심 아이디어:** 소수의 훈련 데이터로도 빠르게 최적화될 수 있는 좋은 초기 매개변수($\theta$)를 메타 학습합니다. 이는 학습자(learner) 모델과 메타 학습자(meta-learner) 모델의 두 단계로 나뉩니다.
  - **예시:**
    - **LSTM Meta-Learner [35]:** LSTM을 메타 학습자로 사용하여 학습자 모델의 매개변수($\theta$)를 몇 단계 만에 업데이트하는 방법을 학습합니다 (Eq. 17, Algorithm 2).
    - **Model-Agnostic Meta-Learning (MAML) [14]:** 단일 모델의 초기 매개변수 $\theta$를 학습하여 새로운 태스크에 대해 몇 번의 경사 하강 단계만으로 빠르게 최적화될 수 있도록 합니다 (Algorithm 3).
      - **MAML 변형:** Proto-MAML [39], TAML [40], MAML++ [41], HSML [42], CAVIA [43].
    - **Meta-Transfer Learning (MTL) [37]:** 사전 훈련된 DNN을 특징 추출기($\Theta$)로 사용하고, 마지막 레이어 분류기 매개변수 $\theta$와 스케일 및 시프트 매개변수($\phi_{S_1}, \phi_{S_2}$)만 메타 학습하여 심층 네트워크의 과적합 문제를 완화합니다.
    - **Latent Embedding Optimization (LEO) [38]:** 모델 매개변수의 저차원 잠재 임베딩을 학습하고, 이 잠재 공간에서 최적화 기반 메타 학습을 수행합니다.

- **1.3. Model-based Meta-Learning (모델 기반 메타 학습)**
  - **핵심 아이디어:** 빠르고 효과적인 학습을 위해 특별히 설계된 모델 아키텍처를 사용합니다. 외부 메모리나 "고속 가중치(fast-weights)" 같은 메커니즘을 통합합니다.
  - **예시:**
    - **Memory Augmented Neural Networks (MANN) [44]:** 수정된 Neural Turing Machine (NTM)을 사용하여 새로운 데이터를 메모리에 빠르게 동화하고, 이 데이터를 활용하여 소수의 샘플 후에도 정확한 예측을 합니다.
    - **Memory Matching Networks (MM-Net) [46]:** Key-Value Memory Networks의 메모리 모듈을 Matching Networks에 통합하여 지원 세트 전체를 메모리 슬롯에 인코딩하고 일반화합니다.
    - **Meta Networks (MetaNet) [48]:** 고속 가중치를 사용하여 태스크에 걸쳐 빠른 일반화를 목표로 하는 메타 학습 모델입니다.
    - **Conditionally Shifted Neurons (CSNs) [49]:** 제한된 태스크 경험을 기반으로 빠르게 채워지는 메모리 모듈에서 가져온 태스크별 시프트로 활성화 값을 수정합니다.
    - **SNAIL [50]:** 메타 학습을 시퀀스-투-시퀀스 문제로 공식화하고, 시퀀스 예측을 위해 Temporal Convolutions와 Causal Attention 레이어를 교차하여 사용합니다.

### 2. 비메타 학습 기반 FSL (Non-Meta-Learning based Few-Shot Learning)

메타 학습 외의 전략을 사용하여 소량 데이터 환경에서의 학습을 돕습니다.

- **2.1. Transfer Learning (전이 학습)**

  - **핵심 아이디어:** 대량의 데이터를 가진 관련 태스크에서 사전 훈련된 딥 네트워크의 지식을 전이하여, 소수의 새 클래스에 대해 미세 조정합니다.
  - **방법:**
    - **사전 훈련된 네트워크의 임베딩을 사용한 거리 기반 분류:** SimpleShot [68]은 사전 훈련된 네트워크에서 특징 임베딩을 얻고, 정규화 후 유클리드 거리를 사용한 k-NN 분류를 수행합니다.
    - **사전 훈련된 네트워크의 임베딩을 사용한 새 분류기 훈련:** Tian et al. [70]은 사전 훈련된 네트워크에서 표현을 얻고, 이를 L2 정규화한 후 각 few-shot 태스크에 대한 새로운 분류기를 훈련합니다.
    - **사전 훈련된 네트워크의 임베딩을 사용한 전이적 추론 (Transductive Inference):** Dhillon et al. [71]은 레이블링된 지원 샘플뿐만 아니라 레이블링되지 않은 쿼리 샘플도 활용하여 사전 훈련된 네트워크를 미세 조정합니다 (Eq. 18). Ziko et al. [72]은 라플라시안 정규화된 전이적 추론을 제안합니다 (Eq. 19).

- **2.2. Miscellaneous (기타: 오토인코더)**
  - **MoVAE (Mixture of Variational Autoencoders) [73]:** 각 클래스에 대해 VAE를 훈련하고, 재구성 손실을 측정하여 분류를 수행하는 one-shot 학습 방법입니다.

### 3. 하이브리드 접근 방식 (Hybrid Approaches)

FSL 문제의 다양한 변형을 다룹니다.

- **Cross-Modal FSL:** 이미지 데이터의 한계를 완화하기 위해 텍스트와 같은 다른 양식의 의미론적 데이터를 활용합니다 (예: AM3 [54]).
- **Semi-Supervised FSL:** 제한된 레이블링된 데이터와 함께 충분한 레이블링되지 않은 데이터를 활용하여 few-shot 분류기의 성능을 향상시킵니다 (예: Ren et al. [17], Sun et al. [55]).
- **Generalized FSL:** 메타 훈련 클래스(seen)와 새로운 클래스(unseen) 모두에서 쿼리를 공동으로 분류하는 것을 목표로 합니다 (예: Gidaris & Komodakis [56], Ye et al. [57]).
- **Generative FSL:** 제한된 샘플로부터 더 많은 합성 샘플을 생성하여 데이터를 증강하는 방법을 학습합니다 (예: "hallucinator" [59]).
- **Cross Domain FSL:** 훈련 및 테스트 태스크가 다른 도메인에 속할 때 발생하는 도메인 시프트 문제에 대처합니다 (예: Tseng et al. [60]).
- **Transductive FSL:** 쿼리 세트의 레이블링되지 않은 예제에 포함된 정보를 전체적으로 활용하여 예측을 수행합니다 (예: Transductive Propagation Networks [62]).
- **Unsupervised FSL:** 지원 세트의 예제도 레이블링되지 않은 시나리오에서, 클러스터링을 통해 쿼리를 할당합니다 (예: Huang et al. [64]).
- **Zero-Shot Learning (ZSL):** 태스크에 대한 훈련 예제가 전혀 없는 경우를 다루며, 시각-보조 양식 정렬에 주로 의존합니다 (예: Xian et al. [65]).

## 📊 Results

Few-Shot Learning 분야는 2016년 Matching Networks의 43% 정확도에서 시작하여, 2020년 1월 기준 miniImageNet [13] 5-way 1-shot 분류 태스크에서 80% 이상의 정확도를 달성하며 상당한 발전을 이루었습니다 (Figure 20, Table 9 참조).

- **정확도 향상:** 다양한 최적화, 거리, 모델 기반 및 하이브리드 접근 방식들이 지속적으로 정확도를 높여왔습니다.
- **다양한 접근 방식의 경합:** Metric-based, Optimization-based, Hybrid Meta-Learning은 물론 Non-Meta-Learning (예: SimpleShot) 접근 방식까지 모두 높은 성능을 보여주며, 특정 한 가지 방법이 최고라고 단정하기 어려운 상황입니다.
- **miniImageNet 5-way 1-shot 기준:** SimpleShot [68]과 LST [55], AM3 [54], CAN [32] 등의 하이브리드/비메타 학습 방식들이 상위권에 랭크되어, 복잡한 메타 학습 방식에 필적하거나 능가하는 성능을 보여주기도 합니다.

## 🧠 Insights & Discussion

Few-Shot Learning은 데이터 부족, 수집 비용, 프라이버시 문제 등 실제 환경의 제약을 극복하려는 중요한 연구 분야입니다. 인간처럼 소수의 예제만으로 새로운 정보를 학습하고 일반화하는 능력에 초점을 맞춥니다.

**주요 시사점:**

- **메타 학습의 효과:** 메타 학습은 "학습하는 방법을 학습"함으로써 새로운 태스크에 대한 빠른 적응과 일반화 능력을 크게 향상시켰습니다.
- **전이 학습의 재발견:** 사전 훈련된 네트워크를 활용한 단순한 전이 학습 기반 방법들(예: SimpleShot)이 복잡한 메타 학습 방법들과 대등하거나 더 나은 성능을 보여주며, 강력한 특징 임베딩의 중요성을 강조합니다.
- **하이브리드 접근의 잠재력:** FSL의 다양한 실제 시나리오(예: 교차 도메인, 준지도 학습)에 대한 하이브리드 접근 방식은 FSL의 적용 범위를 넓히는 데 기여하고 있습니다.

**한계 및 미해결 과제:**

- **경직된 훈련 방식:** 대부분의 메타 학습 방법은 M-way K-shot 에피소드 훈련 패러다임에 의존합니다. 이는 실제 배포 시 태스크의 클래스 수(M)와 예제 수(K)를 미리 알 수 없거나, K보다 적은 예제가 주어질 경우 성능 저하를 초래할 수 있어 유연성이 떨어집니다.
- **단일 태스크 분포 제약:** 훈련 및 테스트 태스크가 동일한 분포($p(T)$)에서 샘플링된다는 가정이 일반적입니다. 이는 모델이 훈련된 도메인과 다른 도메인의 새로운 태스크에 일반화되지 못하는 "Cross Domain FSL" 문제를 야기합니다.
- **기존 클래스와 신규 클래스의 공동 분류:** 현재 FSL 분류기는 훈련 종료 후 지원 세트에 있는 새로운 클래스만 분류할 수 있으며, 훈련에 사용된 기존(seen) 클래스를 인식하지 못합니다. 실제 환경에서는 기존 클래스와 새로운(unseen) 클래스를 함께 분류하는 "Generalized Few-Shot Learning" 능력이 필요합니다.
- **이미지 외 도메인으로의 확장:** FSL 연구는 주로 이미지 분류에 집중되어 왔습니다. 오디오, 무선 신호 등 다른 데이터 도메인은 대규모 데이터셋이 부족하고 데이터의 균일성을 확보하기 어려워 FSL 방법 적용에 어려움이 있습니다.

## 📌 TL;DR

Few-Shot Learning (FSL)은 딥러닝의 데이터 부족 문제를 해결하기 위해 소수의 예제로 새로운 태스크에 일반화하는 방법을 연구합니다. 본 논문은 FSL 접근 방식을 메타 학습 기반 (거리, 최적화, 모델 기반)과 비메타 학습 기반 (전이 학습, 오토인코더)으로 분류하고, 교차 모달/도메인, 준지도, 일반화된 FSL 등 하이브리드 시나리오를 포괄적으로 요약했습니다. miniImageNet 벤치마크에서 지속적인 성능 향상을 보였지만, 경직된 훈련 패러다임, 단일 태스크 분포 제약, 기존/신규 클래스 공동 분류의 어려움, 이미지 외 도메인으로의 확장 등이 여전히 중요한 과제로 남아있습니다.
