# Vision–Language Controlled Deep Unfolding for Joint Medical Image Restoration and Segmentation

Ping Chen, Zicheng Huang, Xiangming Wang, Yungeng Liu, Bingyu Liang, Haijin Zeng, Yongyong Chen (2026)

## 🧩 Problem to Solve

본 논문은 의료 영상 복원(Medical Image Restoration, MedIR)과 의료 영상 분할(Medical Image Segmentation, MedIS)을 하나의 통합된 프레임워크에서 동시에 수행하는 **All-in-One Medical Image Restoration and Segmentation (AiOMIRS)** 문제를 해결하고자 한다.

기존의 의료 영상 분석 파이프라인은 복원과 분할을 독립적인 단계로 처리하는 경향이 있다. 그러나 실제 임상 환경에서 입력 영상은 노이즈, 블러, 언더샘플링 등의 열화(Degradation)가 포함된 저화질(Low-Quality, LQ) 상태인 경우가 많으며, 이는 후속 단계인 분할 모델의 성능을 심각하게 저하시키는 도메인 시프트(Domain Shift) 문제를 야기한다.

따라서 본 연구의 목표는 저수준의 신호 복구(Low-level signal recovery)와 고수준의 의미론적 이해(High-level semantic understanding) 사이의 간극을 메워, 복원이 분할의 정확도를 높이고 분할의 의미론적 정보가 복원 과정을 규제하는 상호 시너지 효과를 갖는 통합 최적화 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 복원과 분할이 근본적으로 상호 보완적이라는 직관에서 출발하며, 이를 위해 다음과 같은 설계 요소를 도입한다.

1. **Vision-Language Dual Prior 메커니즘**: 의료 영상의 모달리티(Modality)와 열화 유형(Degradation)에 따른 분포 차이를 해결하기 위해, 의료 데이터셋으로 파인튜닝된 CLIP 모델을 사용하여 명시적인 모달리티 및 열화 사전 정보를 추출하고 이를 복원 과정에 주입한다.
2. **Frequency-Aware Mamba 기반 Deep Unfolding Network (DUN)**: 최적화 알고리즘을 신경망 구조로 풀어서 설계한 Deep Unfolding 방식을 채택하여 해석 가능성을 확보하였다. 특히, Mamba의 선형 복잡도로 글로벌 컨텍스트를 캡처하면서도, 고주파 텍스처가 손실되는 Spectral Bias 문제를 해결하기 위해 주파수 분리(Frequency-decoupling) 전략을 도입하였다.
3. **상호 협력적 공동 학습(Joint Collaborative Learning)**: 복원 헤드(Rec. Head)와 분할 헤드(Seg. Head)를 동시에 최적화함으로써, 분할 작업이 복원 과정에서 해(Solution)의 공간을 제한하는 의미론적 정규화(Semantic Regularization) 역할을 수행하게 한다.

## 📎 Related Works

### 1. All-in-One 의료 영상 작업

최근 다양한 모달리티와 열화 유형을 하나의 모델로 처리하려는 All-in-One MedIR 및 MedIS 연구가 진행되었다. 하지만 기존의 AiO 프레임워크들은 복원이나 분할 중 어느 한 가지 작업에만 집중하며, 두 작업을 통합하여 다루는 연구는 극히 드물며 대부분 단일 모달리티나 단일 열화 유형에 국한되어 있다.

### 2. 의료 영상 복원을 위한 VLM

CLIP과 같은 Vision-Language Model(VLM)을 사용하여 텍스트 프롬프트로 복원 네트워크를 가이드하려는 시도가 있었으나, 자연 영상 기반의 VLM은 의료 영상 특유의 도메인 갭과 미세한 열화 패턴을 인식하는 데 한계가 있다.

### 3. Deep Unfolding Networks (DUN)

DUN은 ADMM이나 Proximal Gradient Descent(PGD)와 같은 반복적 최적화 알고리즘을 네트워크 층으로 전개하여 모델 기반의 해석 가능성과 딥러닝의 학습 능력을 결합한다. 기존 DUN은 주로 CNN이나 Transformer를 사용하였으나, CNN은 글로벌 컨텍스트 캡처 능력이 부족하고 Transformer는 고해상도 영상 처리 시 계산 복잡도가 $O(N^2)$로 증가하는 문제가 있다.

## 🛠️ Methodology

### 1. 문제 정의

의료 영상의 열화 과정은 다음과 같은 선형 역문제(Linear Inverse Problem)로 모델링된다.
$$y = \Phi x + n, \quad n \sim \mathcal{N}(0, \Sigma)$$
여기서 $y$는 관측된 저화질(LQ) 영상, $\Phi$는 열화 행렬, $x$는 복원하고자 하는 고화질(HQ) 영상, $n$은 노이즈이다. AiOMIRS의 목표는 $y$로부터 $x$를 복원함과 동시에 의미론적 분할 마스크 $S$를 예측하는 것이다.

### 2. Vision-Language Prior Extraction

입력 영상 $y$에 대해 파인튜닝된 CLIP을 사용하여 두 가지 사전 정보를 추출한다.

* **Modality Prior**: CLIP의 Visual Encoder $E_v$와 선형 분류기를 통해 영상의 모달리티를 인식한다. 불균형한 데이터셋 문제를 해결하기 위해 Focal Loss를 사용하여 최적화한다.
* **Degradation Prior**: "Noisy medical image with severe grain"과 같은 고정된 텍스트 프롬프트를 사용하여 열화 유형을 인식한다. Image-Text Contrastive Loss를 통해 시각적 아티팩트와 텍스트 설명 간의 정렬을 수행한다.

### 3. Attention-Mamba based Deep Unfolding Network

본 모델은 Proximal Gradient Descent(PGD) 알고리즘을 $K$개의 스테이지로 전개한 구조를 가진다. 최소화하고자 하는 에너지 함수는 다음과 같다.
$$\hat{x} = \arg \min_x \frac{1}{2} \|y - \Phi x\|^2 + \lambda R(x)$$

#### (1) Attention-Mamba union Gradient Descent Module (AMGDM)

데이터 충실도(Data Fidelity) 항을 업데이트하는 모듈이다.

* **Cross-Attention**: 영상 특징을 Query(Q)로, CLIP에서 추출한 Dual Prior를 Key(K) 및 Value(V)로 사용하여 입력 영상의 특성에 맞는 열화 연산자 $\Phi$를 동적으로 추정한다.
* **Bi-Mamba**: 전역적인 잔차 오차 $\Phi x - y$를 매핑하여 효율적인 그래디언트 추정을 수행한다.

#### (2) Mamba-GDFN based Proximal Map Module (MDPMM)

추정된 결과를 깨끗한 영상 매니폴드로 투영하는 근접 연산자(Proximal Operator) 역할을 한다. Mamba의 Low-pass 필터링 특성(Spectral Bias)을 극복하기 위해 주파수를 분리하여 처리한다.
$$x_{low} = \text{AvgPool}(z^k), \quad x_{high} = z^k - x_{low}$$

* **High-Frequency Branch**: Bi-directional Mamba를 사용하여 고주파 성분 내의 전역적 의존성을 캡처함으로써, 단순 노이즈와 해부학적 텍스처를 구분하여 보존한다.
* **Low-Frequency Branch**: Gated-Dconv Feed-Forward Network (GDFN)를 사용하여 안정적인 해부학적 구조의 일관성을 유지한다.

최종적으로 복원 헤드(Rec. Head)와 분할 헤드(Seg. Head)로 분기되어 각각 고화질 영상과 분할 마스크를 출력한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: CLIP 사전 학습을 위해 8개의 의료 데이터셋(ACDC, BraTS2021, COVID19-CT 등)을 사용하였으며, 메인 작업 평가는 ACDC, BraTS2021, COVID19-CT, HCC-TACE-Seg 4개 데이터셋에서 수행되었다.
* **열화 시뮬레이션**: CT와 MRI 모두 주파수 도메인에서의 언더샘플링 과정으로 모델링하여 물리적 특성을 반영하였다.
* **비교 대상**: DenoiSeg, PromptIR, AMIR, VLU-Net, TAT, SAM, Med-SAM 등 최신 복원 및 분할 모델들과 비교하였다.

### 2. 정량적 결과

VL-DUN은 복원과 분할 모든 지표에서 SOTA(State-of-the-art) 성능을 달성하였다.

* **복원 성능**: 평균 PSNR이 $0.92\text{dB}$ 향상되었다.
* **분할 성능**: Dice 계수가 평균 $9.76\%$ 향상되었다.
* 특히, LQ 영상을 입력으로 받는 VL-DUN이 HQ 영상을 입력으로 받는 전용 분할 모델(SAM, Med-SAM)보다 더 높은 Dice 점수를 기록하였다.

### 3. 효율성 분석

파라미터 수와 FLOPs 측면에서 다른 All-in-One 모델(AMIR, VLU-Net 등)보다 낮은 수치를 기록하여 계산 효율성이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 1. 작업 간의 상호 시너지 (Task Synergy)

본 논문은 복원과 분할의 공동 최적화가 왜 효과적인지를 이론적으로 분석하였다.

* **복원 $\rightarrow$ 분할**: 복원 모듈이 저화질 영상의 불확실성을 제거하여 분할 헤드가 아티팩트가 아닌 실제 해부학적 경계선을 추적할 수 있게 한다.
* **분할 $\rightarrow$ 복원**: 분할 작업이 의미론적 제약 조건(Semantic Manifold Constraint)으로 작용하여, 복원 문제의 해 공간(Solution Space)을 획기적으로 줄인다. 이는 특히 해부학적 경계 부분에서 과도한 평활화(Over-smoothing)를 방지하고 날카로운 엣지를 유지하게 한다.

### 2. Mamba의 Spectral Bias 해결

일반적인 Mamba 구조는 저역 통과 필터(Low-pass filter)처럼 동작하여 고주파 세부 정보를 손실시키는 경향이 있다. 본 연구는 주파수 분리 전략을 통해 고주파 성분만을 Mamba로 처리하게 함으로써, 전역적 컨텍스트를 활용해 노이즈와 텍스처를 구분하는 능력을 확보하였다.

### 3. VLM(CLIP) 도입의 타당성

단순 CNN 분류기나 VAE 기반의 특징 추출보다 CLIP이 우월한 이유는 이미 방대한 양의 이미지-텍스트 쌍으로 사전 학습되어 모달리티와 열화 특징에 대한 풍부한 사전 지식을 가지고 있기 때문이다. 또한, 텍스트라는 공통의 참조 체계를 통해 서로 다른 모달리티 간의 정렬을 더 효율적으로 수행할 수 있다.

## 📌 TL;DR

본 논문은 의료 영상의 복원과 분할을 통합적으로 수행하는 **VL-DUN** 프레임워크를 제안한다. 파인튜닝된 CLIP을 통해 모달리티와 열화 정보를 추출하여 복원 과정을 가이드하며, 주파수 분리 전략을 적용한 Mamba 기반의 Deep Unfolding 구조를 통해 계산 효율성과 고해상도 텍스처 보존 능력을 동시에 확보하였다. 실험 결과, 복원과 분할 작업이 서로를 정규화하는 시너지 효과를 통해 기존 개별 처리 방식 및 최신 SOTA 모델들을 뛰어넘는 성능을 보였으며, 이는 향후 복잡한 임상 워크플로우를 위한 통합 의료 영상 분석 모델의 새로운 방향성을 제시한다.
