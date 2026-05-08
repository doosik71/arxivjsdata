# A Survey on Generative Modeling with Limited Data, Few Shots, and Zero Shot

Milad Abdollahzadeh, Guimeng Liu, Touba Malekzadeh, Christopher T.H Teo, Keshigeyan Chandrasegaran, Ngai-Man Cheung (2023/2025)

## 🧩 Problem to Solve

본 논문은 데이터 제약 상황에서의 생성 모델링(Generative Modeling under Data Constraint, 이하 GM-DC)이라는 연구 분야를 체계적으로 분석하고 정리하는 것을 목표로 한다.

전통적인 생성 모델(GANs, Diffusion Models 등)은 대규모의 다양하고 방대한 데이터셋을 전제로 설계되었다. 그러나 의료 영상, 위성 이미지, 예술적 도메인과 같은 실제 환경에서는 데이터 수집 비용이 매우 높거나 엄격한 개인정보 보호 제약으로 인해 대량의 데이터를 확보하기 어려운 경우가 많다. 데이터가 부족한 환경에서 모델을 학습시킬 경우 다음과 같은 치명적인 문제들이 발생한다.

1. **Overfitting (과적합):** 모델의 용량(Capacity)에 비해 데이터가 너무 적어, 데이터의 일반적인 통계적 분포를 학습하는 대신 개별 샘플을 그대로 암기해 버리는 현상이 발생한다.
2. **Mode Collapse (모드 붕괴):** 생성된 샘플의 다양성이 급격히 떨어지며, 특정 몇 가지 패턴만을 반복해서 생성하는 문제가 발생한다.
3. **Frequency Bias (주파수 편향):** 고주파 성분(세부 디테일)을 무시하고 저주파 성분(대략적인 형태)만을 학습하려는 경향이 있으며, 데이터가 적을수록 이 현상이 심화되어 생성 이미지의 품질이 저하된다.

따라서 본 논문은 Limited-data(50~5,000개), Few-shot(1~50개), Zero-shot(0개) 설정에서의 생성 모델링을 포괄적으로 조사하고, 이를 해결하기 위한 기술적 방법론을 체계화하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 GM-DC 분야의 방대한 문헌(230편 이상의 논문)을 분석하여 연구자들이 참고할 수 있는 실질적인 로드맵을 제시한 점에 있다. 구체적인 기여 사항은 다음과 같다.

1. **새로운 과제 분류 체계(Task Taxonomy) 제안:** 생성 모델링의 목적과 제약 조건에 따라 uGM-1~3, cGM-1~3, IGM, SGM 등 8가지의 세부 과제로 정의하여 연구 범위를 명확히 하였다.
2. **방법론적 분류 체계(Approach Taxonomy) 제안:** Transfer Learning, Data Augmentation, Network Architectures, Multi-Task Objectives, Frequency-aware modeling, Meta-Learning, Internal Patch Distribution Modeling의 7가지 핵심 접근 방식으로 방법론을 체계화하였다.
3. **연구 지형 시각화:** Sankey Diagram을 통해 과제(Task)-접근 방식(Approach)-개별 방법론(Method) 간의 상호작용을 시각적으로 제시하였다.
4. **심층 분석 및 실험적 비교:** 각 접근 방식의 설계 원리와 한계를 비판적으로 분석하고, 대표적인 방법론들 간의 정량적 성능(FID, LPIPS 등)을 비교 분석하였다.

## 📎 Related Works

본 논문은 GM-DC를 세 가지 관점에서 기존 연구와 차별화한다.

1. **판별 모델의 Few-shot Learning과의 차이:** 기존의 Few-shot Learning 연구들은 대부분 분류(Classification)나 회귀(Regression)와 같은 판별적 태스크(Discriminative tasks)에 집중해 왔다. 반면, GM-DC는 데이터의 분포 $P_{data}$를 학습하여 새로운 샘플을 생성하는 생성적 태스크를 다루므로 문제 정의부터 근본적으로 다르다.
2. **일반 생성 모델 서베이와의 차이:** 기존의 생성 모델 서베이들은 모델의 구조적 발전이나 고해상도 이미지 생성, 제어 가능성(Controllability) 등에 집중하며, 대량의 데이터가 있다는 가정하에 논의가 진행되었다. 데이터 제약 상황이라는 특수한 환경에서의 문제 해결책은 충분히 다루지 않았다.
3. **기존 GM-DC 서베이(Li et al., 2022d)와의 차이:** 이전의 서베이는 주로 GAN 모델에만 국한되었고 다루는 논문의 수도 매우 적었다(본 논문 대비 약 13% 수준). 본 논문은 GAN뿐만 아니라 VAE, Diffusion Models를 모두 포함하며, 최신 연구 흐름을 반영하여 훨씬 더 넓은 범위와 깊은 분석을 제공한다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 새로운 알고리즘을 제안하기보다, 기존 생성 모델의 기본 원리와 GM-DC를 해결하기 위한 방법론적 구조를 분석하는 데 집중한다.

### 1. 생성 모델의 기본 원리 및 방정식

논문에서는 GM-DC의 기반이 되는 세 가지 주요 모델을 다음과 같이 설명한다.

- **VAE (Variational Auto-Encoder):** 입력 $x$를 잠재 공간 $z$로 매핑하는 인코더 $E$와 이를 다시 복원하는 디코더 $D$로 구성된다. 손실 함수는 복원 오차와 잠재 공간의 정규화를 위한 KL-divergence의 합으로 정의된다.
  $$\mathcal{L} = ||x - D(z)||^2 + \text{KL}(N(\mu, \sigma^2), N(0, I))$$
- **GAN (Generative Adversarial Networks):** 생성자 $G$와 판별자 $D$가 서로 경쟁하며 학습하는 구조이다.
  $$V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
- **Diffusion Models (DM):** 데이터에 점진적으로 노이즈를 추가하는 forward 과정과, 이를 다시 제거하는 reverse 과정을 학습한다. 핵심은 노이즈 예측 모델 $\epsilon_\theta$를 학습시키는 것이며, 손실 함수는 다음과 같다.
  $$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2]$$

### 2. GM-DC 과제 분류 (Task Taxonomy)

본 논문은 다음과 같이 GM-DC 태스크를 정의한다.

- **uGM (Unconditional GM):** 조건 없이 샘플을 생성.
  - **uGM-1:** $K$개의 샘플만으로 해당 도메인 학습.
  - **uGM-2:** 사전 학습된 모델 $\text{G}_s$와 $K$개의 타겟 샘플을 이용해 타겟 도메인으로 적응(Adaptation).
  - **uGM-3:** 사전 학습된 모델 $\text{G}_s$와 텍스트 프롬프트만을 이용해 제로샷 적응.
- **cGM (Conditional GM):** 클래스 레이블 등 조건 $c$를 입력으로 받아 생성.
  - **cGM-1:** $K$개의 레이블된 샘플로 학습.
  - **cGM-2:** 학습된 클래스(Seen) 외의 새로운 클래스(Unseen)에 대해 $K$개의 샘플로 학습.
  - **cGM-3:** 소스 도메인 모델을 타겟 도메인의 조건부 생성 모델로 적응.
- **IGM (Internal Patch Distribution GM):** 단일 이미지 내의 패치 통계 특성을 학습하여 유사한 스타일의 새로운 이미지 생성 (예: SinGAN).
- **SGM (Subject-driven GM):** 특정 대상(Subject)의 소수 이미지와 텍스트를 이용해 해당 대상을 다양한 맥락에서 생성 (예: DreamBooth).

### 3. 방법론적 접근 방식 (Approach Taxonomy)

데이터 제약을 극복하기 위한 7가지 전략을 제시한다.

1. **Transfer Learning:** 사전 학습된 모델의 지식을 활용. (Regularizer 기반 미세 조정, 잠재 공간 조작, Modulation, 언어 가이드, 적응 인식, Prompt Tuning 등)
2. **Data Augmentation:** 이미지 수준, 특징 수준, 변환 기반 설계 등을 통해 데이터 분포를 인위적으로 확장.
3. **Network Architectures:** 특성 강화 모듈 추가, 사전 학습된 비전 모델 앙상블, 동적 네트워크 구조 설계.
4. **Multi-Task Objectives:** Contrastive Learning, Masking, Knowledge Distillation, Prototype Learning 등 보조 태스크를 추가하여 과적합 방지.
5. **Exploiting Frequency Components:** Haar Wavelet 등을 이용하여 고주파 성분을 명시적으로 학습함으로써 Frequency Bias 해결.
6. **Meta-Learning:** 여러 태스크로부터 공통 지식을 학습하여 새로운 클래스에 빠르게 적응 (Optimization, Transformation, Fusion 기반).
7. **Modeling Internal Patch Distribution:** 단일 이미지의 내부 통계적 특성을 다단계(Progressive) 또는 단일 단계로 학습.

## 📊 Results

본 논문은 각 태스크별 대표 방법론들의 성능을 정량적으로 비교하였다. 주요 지표로는 **FID (Fréchet Inception Distance)** (낮을수록 좋음)와 **LPIPS / Intra-LPIPS** (다양성 측정, 높을수록 좋음)를 사용하였다.

### 1. 주요 태스크별 성능 분석

- **uGM-1 (제한적 데이터 생성):** **DANI**가 가장 낮은 FID를 기록하며 최신 방법론의 우수성을 보였다. 초기 방법론인 ADA나 LeCam보다 정교한 정규화 및 적응 전략이 효과적임을 입증하였다.
- **uGM-2 (교차 도메인 적응):** **RICK**이 FID와 Intra-LPIPS 모두에서 가장 우수한 성능을 보였다. 이는 단순히 지식을 보존하는 것을 넘어, 타겟 도메인에 불필요하거나 상충되는 지식을 제거(Pruning)하는 '적응 인식(Adaptation-aware)' 전략의 중요성을 보여준다.
- **uGM-3 (텍스트 가이드 제로샷 적응):** **AIR**가 CLIP 공간에서의 오프셋 불일치(Offset misalignment) 문제를 반복적 앵커 샘플링으로 해결하여 가장 일관된 결과를 냈다.
- **cGM-1 (조건부 제한 데이터 생성):** **CbC**가 클래스 기반의 대조 전략을 통해 가장 낮은 FID를 달성하며, 단순 데이터 증강보다 클래스 간 분별력을 높이는 전략이 효과적임을 보였다.
- **SGM (대상 중심 생성):** **DreamBooth**가 가장 높은 충실도(Fidelity)를 보였으나 계산 비용이 높다. 최근의 **MoMA**와 같은 Tuning-free 방법론들이 효율성과 품질 사이의 균형을 잘 맞추고 있음을 확인하였다.

### 2. 실험적 통찰

- **도메인 거리의 영향:** 소스 도메인과 타겟 도메인의 의미적 거리가 멀수록(예: 사람 얼굴 $\rightarrow$ 꽃) 기존의 Transfer Learning 방법론들의 성능이 급격히 저하되며, 이는 '부적절한 지식 전이(Incompatible Knowledge Transfer)' 문제로 이어진다.
- **샘플 선택의 중요성:** 동일한 10-shot 설정이라도 어떤 샘플을 선택하느냐에 따라 FID 결과가 크게 달라짐을 확인하였으며, 이는 GM-DC에서 데이터 중심(Data-centric) 접근 방식이 필수적임을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 핵심 통찰

- **지식 보존 vs. 적응의 트레이드-오프:** Transfer Learning에서 가장 핵심적인 설계 원리는 소스 모델의 일반적 지식을 얼마나 보존할 것인가와 타겟 도메인의 특성에 얼마나 유연하게 적응할 것인가 사이의 균형을 맞추는 것이다.
- **주파수 편향의 재발견:** 생성 모델이 저주파 성분에 치우치는 경향이 있으며, 데이터가 적을수록 이 문제가 심화된다는 점을 명시하고 이를 해결하기 위한 Frequency-aware 설계의 필요성을 역설하였다.

### 2. 한계 및 미해결 과제

- **원거리 도메인 전이:** 현재의 GM-DC 연구들은 대부분 의미적으로 유사한 도메인(예: 성인 얼굴 $\rightarrow$ 아기 얼굴)에 집중되어 있다. 매우 이질적인 도메인 간의 전이는 여전히 매우 어려운 문제로 남아 있다.
- **평가 지표의 한계:** FID나 KID 같은 통계적 지표는 샘플 수가 극도로 적은(Few-shot) 상황에서는 통계적 유의성이 떨어진다. 따라서 GM-DC 전용의 새로운 평가 프레임워크가 필요하다.
- **데이터 중심 전략의 부재:** 대부분의 연구가 알고리즘(모델 구조, 손실 함수) 개선에 집중하고 있으나, 실제로는 어떤 데이터를 선택하고 정제하느냐가 성능에 더 큰 영향을 미친다.

### 3. 비판적 해석

본 논문은 매우 방대한 양의 연구를 체계적으로 정리하였으나, 개별 방법론들의 상세한 하이퍼파라미터 설정이나 구현상의 세부 사항보다는 분류와 결과 위주로 서술되어 있다. 하지만 이는 서베이 논문의 목적에 부합하며, 특히 Sankey Diagram과 Taxonomy를 통해 파편화되어 있던 GM-DC 연구들을 하나의 지도(Map)로 통합했다는 점에서 학술적 가치가 매우 높다.

## 📌 TL;DR

본 논문은 데이터가 극도로 부족한 상황에서의 생성 모델링(GM-DC)을 다루는 종합 서베이 보고서이다.

- **핵심 기여:** GM-DC를 위한 **8가지 태스크 분류 체계**와 **7가지 방법론적 접근 방식**을 제안하고, 230편 이상의 논문을 분석하여 연구 지형을 시각화하였다.
- **주요 결론:** Transfer Learning이 가장 지배적인 해결책이며, 최근에는 CLIP과 같은 거대 기초 모델(Foundation Models)을 활용한 제로샷/퓨샷 적응과 SGM(대상 중심 생성) 연구가 급증하고 있다.
- **향후 방향:** 원거리 도메인 간의 지식 전이 해결, 데이터 중심(Data-centric) 샘플 선택 전략 수립, 그리고 저데이터 환경에 최적화된 새로운 평가 지표 개발이 필수적이다.
- **의의:** 이 연구는 데이터 제약이 심한 의료, 위성 영상 등의 특수 분야에서 고품질 생성 모델을 구축하고자 하는 연구자들에게 명확한 기술적 방향성과 벤치마크를 제공한다.
