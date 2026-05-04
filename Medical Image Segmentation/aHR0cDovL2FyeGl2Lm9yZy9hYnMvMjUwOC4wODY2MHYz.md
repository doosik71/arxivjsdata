# Unified and Semantically Grounded Domain Adaptation for Medical Image Segmentation

Xin Wang, Yin Guo, Jiamin Xia, Kaiyu Zhang, Niranjan Balu, Mahmud Mossa-Basha, Linda Shapiro, Chun Yuan (2026)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델은 하드웨어, 촬영 프로토콜, 환자군 및 질병의 양상에 따른 도메인 시프트(Domain Shift) 문제로 인해 성능이 심각하게 저하되는 문제가 발생한다. 이를 해결하기 위해 레이블이 있는 소스 도메인에서 레이블이 없는 타겟 도메인으로 지식을 전달하는 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 연구가 활발히 진행되어 왔다.

기존의 UDA 접근 방식은 크게 두 가지 설정으로 나뉜다. 첫째는 소스 데이터에 접근 가능한 **Source-accessible** 설정으로, 주로 소스와 타겟 간의 특징 정렬(Alignment)을 통해 적응을 유도한다. 둘째는 소스 데이터 없이 사전 학습된 모델만 사용하는 **Source-free** 설정으로, 주로 의사 레이블링(Pseudo-labeling)이나 네트워크 증류(Network Distillation)와 같은 암시적 기법에 의존한다.

본 논문은 이러한 두 설정의 방법론적 괴리가 의료 영상의 핵심인 '해부학적 지식(Anatomical Knowledge)'에 대한 명시적이고 구조적인 구축이 부족하기 때문에 발생한다고 지적한다. 결과적으로 기존 방식들은 적응된 특징이 유효한 해부학적 구조를 캡처하고 있는지 보장하는 설명 가능한 메커니즘이 부족하며, 이는 생리학적으로 불가능하거나 파편화된 분할 결과로 이어진다. 따라서 본 연구의 목표는 소스 데이터 접근 여부와 상관없이 일관되게 적용 가능하며, 해부학적으로 근거가 있는(Semantically Grounded) 통합된 도메인 적응 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간이 새로운 영상 조건에 적응하는 방식, 즉 '전형적인 해부학적 형태를 기억하고 이를 개별 특성에 맞게 변형하는 과정'을 모방하는 것이다. 이를 위해 다음과 같은 핵심 설계를 제안한다.

1.  **해부학적 정규성을 위한 공유 매니폴드(Shared Latent Manifold):** 모든 이미지에 공유되는 도메인 불가지론적(Domain-agnostic) 확률 매니폴드를 구축한다. 이 공간은 소수의 전형적인 해부학적 표현(Basis distributions)의 가중치 조합으로 구성되어, 모델이 전역적인 해부학적 규칙성을 학습하도록 한다.
2.  **해부학적 구조와 기하학적 변형의 분리(Disentanglement):** 이미지의 구조적 콘텐츠를 전형적인 해부학적 템플릿(Canonical anatomical template, $z$)과 개별 특이적 공간 변형(Spatial deformation, $\phi$)으로 명시적으로 분리한다.
3.  **창발적 적응(Emergent Adaptation):** 명시적인 교차 도메인 정렬 전략(Explicit cross-domain alignment) 없이, 모델 아키텍처 자체의 설계(공유 매니폴드)를 통해 도메인 적응이 자연스럽게 이루어지도록 한다.

## 📎 Related Works

### 기존 UDA 접근 방식 및 한계
- **Source-accessible UDA:** 적대적 학습(Adversarial training), 준지도 학습, 통계적 정렬 등을 사용하여 도메인 불변 표현을 학습한다. 그러나 이러한 방법들은 고차원 특징 공간에서 정렬을 수행하므로 계산 비용이 높고, 해부학적 제약 조건이 없어 의미적 모호성(Semantic ambiguity)이 발생하기 쉽다.
- **Source-free UDA:** 의사 레이블링, 엔트로피 최소화, 증류 기법 등을 사용한다. 이러한 방식은 소스 모델의 출력값에 포함된 노이즈에 취약하며, 해부학적 충실도(Anatomical fidelity)를 유지하기 어렵다는 한계가 있다.

### 기존 VAE 기반 연구와의 차별점
기존의 Variational Autoencoders(VAEs) 기반 연구들이 해부학적 구조와 외형을 분리하려 시도했으나, 대부분은 공통 해부학 구조를 가진 쌍을 이룬(Paired) 다중 모달리티 이미지에 의존했다. 반면, 본 논문은 서로 다른 피험자와 공간 위치에서 촬영된 이미지들을 다루는 도메인 적응 설정에서 작동하며, 공유된 잠재 기반(Shared bases)을 통해 도메인 불변의 구조적 특징을 학습한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 확률 모델
본 프레임워크는 이미지를 전형적인 해부학 템플릿 $z$, 공간 변형 $\phi$, 그리고 스타일 $s$로 분리하여 모델링한다. 공간 변형 $\phi$는 Stationary Velocity Field(SVF) $v$를 통해 $\phi = \exp(v)$로 매개변수화된다.

전체 생성 모델의 결합 확률 분포는 다음과 같이 정의된다:
$$p(x, y, w, z, v, s) = p(w)p(s)p(v)p(z|w)p(x|z, v, s)p(y|z, v)$$

여기서 $w$는 템플릿 $z$를 결정하는 저차원 벡터이며, $s$는 스타일 정보를 담고 있다. 학습을 위해 변분 추론(Variational Inference)을 사용하며, 목적 함수는 다음과 같은 Evidence Lower Bound(ELBO)를 최대화하는 것이다:
$$\text{ELBO} = \mathbb{E}_{q} [\log p(x|z, v, s)]_{L_{recon}} + \mathbb{E}_{q} [\log p(y|z, v)]_{L_{seg}} - \mathbb{E}_{q} [D_{KL}(q(z|w) \| p(z|w))]_{L_{tem}} - \mathbb{E}_{q} [D_{KL}(q(v|x, z) \| p(v))]_{L_{vel}} - \dots$$

### 세부 구성 요소 및 학습 절차

#### 1. 공유 기반을 이용한 의미적 인코딩 (Semantically Grounded Encoding)
템플릿 $z$는 자유롭게 학습되는 것이 아니라, 공유된 $M$개의 학습 가능한 기반 분포(Basis distributions) $\{q^m(z^l)\}_{m=1}^M$의 로그 선형 혼합(Log-linear mixture)으로 표현된다:
$$q(z^l|w) \propto \prod_{m=1}^M [q^m(z^l)]^{w_m}$$
여기서 가중치 $w$는 확률 심플렉스(Probability simplex) $\Delta$ 상에 존재하도록 제약된다. 이를 통해 모델은 전형적인 해부학적 원형(Prototypes)을 기억하고 이를 조합하여 다양한 형태를 생성할 수 있다.

#### 2. 매니폴드 구조화 제약 (Manifold Structuring)
매니폴드가 의미 있게 구성되도록 두 가지 추가 손실 함수를 도입한다:
- **Usage Loss ($L_{usage}$):** 각 기반 분포가 고르게 사용되도록 하여 모드 붕괴(Mode collapse)를 방지하고 표현력을 극대화한다.
- **Structural Loss ($L_{struct}$):** 잠재 공간의 거리와 실제 해부학적 구조의 유사성 사이의 관계를 일치시킨다. 이때 심플렉스의 비유클리드 기하학을 반영하기 위해 Fisher-Rao Metric $D_{FR}$을 사용한다:
  $$D_{FR}[w \| w'] = 2 \arccos \left( \sum_{m=1}^M \sqrt{w_m w'_m} \right)$$

#### 3. 통합 적응 패러다임 (Unified Paradigm)
본 프레임워크는 동일한 아키텍처를 사용하되, 학습 단계만 다르게 설정하여 두 가지 환경을 모두 지원한다.

- **Source-accessible:** 소스와 타겟 데이터를 동시에 사용하여 단일 단계로 학습한다.
- **Source-free:** 2단계 학습을 거친다.
    - **Stage 1 (Source-only):** 소스 데이터를 통해 해부학적 기반 분포 $q^m(z^l)$과 분할 디코더를 학습하여 '해부학적 기억'을 형성한다.
    - **Stage 2 (Target-only):** 기반 분포와 분할 디코더를 고정한 채, 타겟 데이터의 이미지-매니폴드 매핑을 재조정(Recalibrate)하여 적응한다.

## 📊 Results

### 실험 설정
- **데이터셋:** MS-CMRSeg (심장 MRI, bSSFP $\rightarrow$ LGE 도메인 시프트), AMOS22 (복부 CT/MRI, MRI $\rightarrow$ CT 도메인 시프트).
- **지표:** Dice Similarity Coefficient (DSC) $\uparrow$, Average Symmetric Surface Distance (ASSD) $\downarrow$.
- **비교 대상:** ADVENT, VarDA, DARUNet (Source-accessible), Tent, AdaMI, ProtoContra (Source-free).

### 주요 결과
1.  **정량적 성능:** 두 데이터셋의 모든 설정에서 기존 SOTA 방법들을 상회하는 성능을 기록했다. 특히 Source-free 설정에서 기존 방식들보다 압도적인 성능 향상을 보였으며, Source-accessible 모델과의 성능 격차를 획기적으로 줄였다.
2.  **정성적 분석:** 타겟 도메인의 영상 품질이 낮거나 노이즈가 심한 경우에도, 제안 방법은 해부학적 템플릿과 변형의 분리 구조 덕분에 파편화되지 않은, 생리학적으로 타당한(Anatomically plausible) 분할 결과를 생성했다.
3.  **해석 가능성:** 
    - **매니폴드 탐색(Traversal):** 가중치 $w$를 보간(Interpolation)했을 때 해부학적 형태가 부드럽게 변하는 것을 확인하여 latent space가 의미 있게 구조화되었음을 입증했다.
    - **t-SNE 시각화:** 명시적인 정렬 손실 없이도 소스와 타겟의 $w$ 벡터들이 잠재 공간에서 잘 겹쳐져 있음을 보여, '창발적 적응'이 성공적으로 이루어졌음을 증명했다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구의 가장 큰 강점은 도메인 적응을 '특징 정렬'의 관점이 아닌 '해부학적 지식의 공유' 관점에서 접근했다는 점이다. 공유된 기반 분포를 통해 구축된 매니폴드는 일종의 '해부학적 메모리' 역할을 하며, 이를 통해 모델은 타겟 도메인의 이미지를 보았을 때 기억 속에 있는 전형적인 구조를 꺼내어 적절히 변형하는 방식으로 적응한다. 이는 인간의 시각적 인지 과정을 모방한 설계로, 높은 강건성과 해석 가능성을 동시에 확보했다.

### 한계 및 향후 과제
- **차원 확장:** 현재 2D 기반으로 구현되어 있어 3D 볼륨 데이터에 적용 시 메모리 및 계산 비용 증가 문제가 예상된다.
- **슬라이스 간 일관성:** 슬라이스를 독립적으로 처리하므로 볼륨 재구성 시 슬라이스 간 연속성이 부족할 수 있다.
- **계산 비용:** 기반 분포와 등록 모듈(Registration module)의 도입으로 단순한 Encoder-Decoder 구조보다는 연산량이 많다.

## 📌 TL;DR

본 논문은 의료 영상 분할을 위해 해부학적 구조(Canonical anatomy)와 개별 기하학(Individual geometry)을 분리하여 모델링하는 **통합 도메인 적응 프레임워크**를 제안한다. 공유된 latent manifold를 통해 소스-접근 가능 및 소스-프리 환경 모두를 지원하며, 명시적인 정렬 기법 없이도 해부학적으로 타당하고 일관된 분할 성능을 달성하였다. 이 연구는 의료 영상 분석에서 단순한 데이터 정렬을 넘어, 구조적 사전 지식을 어떻게 모델에 내재화하고 활용할 수 있는지에 대한 원칙적인 방법론을 제시했다는 점에서 향후 연구에 중요한 기여를 할 것으로 보인다.