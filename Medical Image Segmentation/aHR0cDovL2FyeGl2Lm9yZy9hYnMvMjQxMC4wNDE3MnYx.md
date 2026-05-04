# DB-SAM: Delving into High Quality Universal Medical Image Segmentation

Chao Qin, Jiale Cao, Huazhu Fu, Fahad Shahbaz Khan, Rao Muhammad Anwer (2024)

## 🧩 Problem to Solve

본 논문은 자연 이미지(Natural Image)를 대상으로 학습된 Segment Anything Model (SAM)을 의료 영상 분할(Medical Image Segmentation) 분야에 적용할 때 발생하는 성능 저하 문제를 해결하고자 한다.

의료 영상 분할에서 범용적인 모델을 구축하는 것은 매우 중요하지만, SAM을 의료 영상에 직접 적용할 경우 자연 이미지와 의료 영상(2D 및 3D) 사이의 심각한 도메인 간극(Domain Gap)으로 인해 분할 품질이 크게 떨어진다. 이를 해결하기 위해 모델 전체를 의료 데이터로 재학습시키는 방법이 있으나, 이는 막대한 계산 자원과 대규모의 고품질 의료 데이터셋을 필요로 하므로 현실적으로 어렵다. 기존의 MedSAM과 같은 연구는 SAM의 Mask Decoder만을 미세 조정(Fine-tuning)하여 성능을 개선하려 했으나, 이는 의료 도메인 특유의 지식을 충분히 활용하지 못해 복잡한 외곽선을 가진 장기를 분할할 때 최적의 성능을 내지 못한다는 한계가 있다. 따라서 본 논문의 목표는 적은 파라미터 업데이트만으로도 의료 영상의 도메인 특성을 효과적으로 학습하여 고품질의 범용 의료 영상 분할을 수행하는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 ViT(Vision Transformer) 기반의 글로벌 특징 추출과 Convolution 기반의 로컬 특징 추출을 병렬로 수행하는 **Dual-branch adapted SAM** 구조를 설계하여 도메인 간극을 메우는 것이다.

가장 중점적인 설계는 SAM의 강력한 표현 능력을 유지하기 위해 기존 ViT 인코더는 동결(Frozen)시킨 채, 그 사이에 학습 가능한 **Channel Attention Block**을 삽입하여 도메인 특화된 로컬 특징을 포착하게 한 점이다. 동시에, ViT의 패치 임베딩 과정에서 발생하는 정보 손실을 보완하기 위해 경량 Convolution 브랜치를 추가하여 얕은 수준의 로컬 특징(Shallow features)을 직접 추출한다. 마지막으로, 두 브랜치에서 추출된 서로 다른 성격의 특징들을 **Bilateral Cross-Attention**과 **Automatic Selective Mechanism**을 통해 적응적으로 융합함으로써, 글로벌 문맥 정보와 정밀한 로컬 공간 정보를 모두 확보한 고품질의 마스크를 생성한다.

## 📎 Related Works

논문은 기본적으로 SAM(Segment Anything Model)과 그 의료 버전인 MedSAM을 주요 관련 연구로 다룬다. SAM은 대규모 데이터로 학습되어 뛰어난 범용성을 보이지만, 의료 영상과 같은 특수 도메인에서는 성능이 저하된다. MedSAM은 Mask Decoder만을 튜닝하여 효율성을 높였으나, 도메인 특화 지식(Domain-specific knowledge)을 완전히 활용하지 못하는 한계가 있다.

또한, NLP 분야에서 널리 사용되는 **Adapter** 개념을 도입하였다. Adapter는 사전 학습된 모델의 가중치는 고정한 채 소량의 파라미터만을 추가하여 학습시키는 방식으로, 계산 효율성이 높고 사전 학습된 모델의 일반화 능력을 유지하면서 새로운 도메인에 빠르게 적응시킬 수 있다는 장점이 있다. DB-SAM은 이러한 Adapter 개념을 컴퓨터 비전의 ViT와 Convolution 구조에 결합하여 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
DB-SAM은 입력 이미지를 서로 다른 해상도의 두 이미지($I_{vit} \in \mathbb{R}^{1024 \times 1024}$, $I_{conv} \in \mathbb{R}^{256 \times 256}$)로 리사이징하여 두 개의 병렬 브랜치에 입력한다. 전체 파이프라인은 다음과 같다:
- **ViT Branch**: 동결된 ViT 블록과 학습 가능한 Channel Attention Block의 결합.
- **Convolution Branch**: 경량 Convolution 블록을 통한 얕은 특징 추출.
- **Feature Fusion**: Bilateral Cross-Attention 및 ViT-Conv Fusion Block을 통한 특징 통합.
- **Mask Decoder**: 융합된 특징과 Prompt Embedding을 통해 최종 분할 맵 생성.

### 2. 주요 구성 요소 및 상세 설명

#### (1) ViT Branch 및 Channel Attention Block
ViT의 강력한 특징 표현력을 유지하기 위해 가중치는 고정하고, 각 Attention 블록 뒤에 **Channel Attention Block**을 추가하여 지역성 귀납적 편향(Locality inductive bias)을 부여한다. 이 블록의 연산 과정은 다음과 같다:

$$F_{out} = F_{vit} + \text{Conv}_{1\times 1}(\text{SE}(\text{DWConv}_{3\times 3}(\text{LN}(F_{vit}))))$$

여기서 $\text{LN}$은 Layer Normalization, $\text{DWConv}_{3\times 3}$은 Depth-wise Convolution, $\text{SE}$는 Squeeze-and-Excitation 블록을 의미한다. 이를 통해 ViT 인코더의 각 단계에서 도메인 특화된 고수준 특징을 추출한다.

#### (2) Convolution Branch 및 Bilateral Cross-Attention (BCA)
ViT의 패치 임베딩(16배 다운샘플링)으로 인한 로컬 디테일 손실을 막기 위해, $256 \times 256$ 해상도의 이미지로부터 직접 얕은 특징($F_s$)을 추출하는 경량 Convolution 브랜치를 운용한다. 이 브랜치는 2개의 $3\times 3$ Conv와 3개의 $1\times 1$ Conv 레이어로 구성된다.

두 브랜치의 특징을 융합하기 위해 **Bilateral Cross-Attention**을 사용한다. Deformable Attention을 기반으로 하며, 다음과 같은 상호 교차 쿼리 방식을 취한다:
- ViT 특징($F_d$)을 Query로, Conv 특징($F_s$)을 Key/Value로 사용하여 $F_{cd}$ 생성.
- Conv 특징($F_s$)을 Query로, ViT 특징($F_d$)을 Key/Value로 사용하여 $F_{cs}$ 생성.
이후 Layer Normalization과 Feed-forward 레이어를 거쳐 최종 업데이트된 특징 $F_1^d$와 $F_1^s$를 생성한다.

#### (3) ViT-Conv Fusion Block
최종적으로 두 브랜치의 출력 특징 $F_{od}$와 $F_{os}$를 자동 선택 메커니즘(Automatic Selective Mechanism)을 통해 융합한다. 각 특징은 먼저 Channel Attention 레이어를 통해 로짓($\Lambda_d, \Lambda_s$)으로 변환되며, 이를 합산하여 시그모이드 함수를 적용한 선택 마스크 $M$을 생성한다:

$$M = \text{Sigmoid}(\Lambda_d + \Lambda_s)$$

최종 융합 특징 $F_{output}$은 다음과 같이 계산된다:

$$F_{output} = F_{od} \otimes M + F_{os} \otimes (1 - M)$$

여기서 $\otimes$는 요소별 곱셈(element-wise multiplication)을 의미하며, 이를 통해 각 토큰이 글로벌 문맥 정보와 로컬 공간 정보를 적응적으로 선택하여 수용하게 한다.

### 3. 학습 절차
- **동결 및 학습**: ViT 인코더와 Prompt 인코더는 고정하며, 추가된 Adapter 모듈들과 Mask Decoder만을 학습시킨다.
- **프롬프트 생성**: 실제 임상 환경을 모사하기 위해 Ground-truth 마스크에 0~20 픽셀의 무작위 섭동(perturbation)을 준 Bounding box 프롬프트를 사용한다.
- **손실 함수**: Cross-Entropy Loss와 Dice Loss의 합을 사용하여 학습을 감독한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: MedSAM에서 수집한 30개의 공개 의료 데이터셋을 사용하였다. MR(T1, T2, ADC, FLAIR), CT, X-Ray, 내시경 이미지, 망막 이미지, 병리 이미지 등 매우 다양한 모달리티를 포함한다.
- **평가 지표**: Dice Similarity Coefficient (DSC)와 1mm 허용 오차 기반의 Normalized Surface Distance (NSD)를 사용하였다.

### 2. 정량적 결과
DB-SAM은 2D 및 3D의 모든 의료 영상 분할 작업에서 SAM 및 MedSAM을 유의미하게 앞질렀다.

- **3D 의료 영상 분할 (21개 작업)**:
    - **평균 DSC**: SAM(58.52%) $\rightarrow$ MedSAM(81.02%) $\rightarrow$ **DB-SAM(87.05%)**
    - **평균 NSD**: SAM(37.49%) $\rightarrow$ MedSAM(76.54%) $\rightarrow$ **DB-SAM(85.31%)**
    - 특히 복부 종양(Abdomen Tumor) 분할에서 MedSAM(65.54%) 대비 12.77%p 높은 78.31%의 DSC를 기록하였다.

- **2D 의료 영상 분할 (9개 작업)**:
    - **평균 DSC**: SAM(59.62%) $\rightarrow$ MedSAM(77.22%) $\rightarrow$ **DB-SAM(82.00%)**
    - **평균 NSD**: SAM(64.27%) $\rightarrow$ MedSAM(83.17%) $\rightarrow$ **DB-SAM(91.81%)**

### 3. 정성적 결과 및 Ablation Study
시각화 결과, DB-SAM은 특히 크기가 작거나 모양이 복잡한 장기(Small organs with complicated shapes)를 분할할 때 SAM과 MedSAM보다 훨씬 정밀한 마스크를 생성하는 것으로 나타났다.

Ablation Study 결과에 따르면, Channel Attention만 추가했을 때보다 Convolution 브랜치를 추가하고, 최종 융합 모듈(Final Fusion)까지 적용했을 때 단계적으로 성능이 향상됨이 확인되었다. (3D DSC 기준: Baseline 81.02% $\rightarrow$ Ch-Attn 85.25% $\rightarrow$ Conv-Branch 86.60% $\rightarrow$ Final Fusion 87.05%).

## 🧠 Insights & Discussion

본 논문의 강점은 SAM이라는 거대 모델의 일반화 성능을 유지하면서도, 의료 영상 특유의 **로컬 디테일(Local Detail)**을 효과적으로 포착할 수 있는 경량 구조를 설계했다는 점이다. 특히 ViT의 전역적 시야와 Convolution의 지역적 시야를 단순히 합치는 것이 아니라, Bilateral Cross-Attention과 선택적 융합 메커니즘을 통해 적응적으로 결합한 점이 성능 향상의 주요 원인으로 분석된다.

하지만 몇 가지 고려할 점이 있다. 입력 이미지를 두 가지 해상도로 리사이징하여 서로 다른 브랜치에 넣는 과정이 추가적인 연산 오버헤드를 발생시킬 수 있다. 또한, 3D 데이터를 2D 슬라이스로 나누어 처리하는 방식은 슬라이스 간의 연속성(Inter-slice continuity) 정보를 충분히 활용하지 못했을 가능성이 있다. 그럼에도 불구하고, 방대한 의료 데이터셋에 대해 범용적으로 적용 가능함을 입증했다는 점에서 실용적 가치가 매우 높다.

## 📌 TL;DR

- **주요 기여**: SAM의 도메인 간극을 해결하기 위해 ViT 브랜치(글로벌/도메인 특화)와 Convolution 브랜치(로컬/얕은 특징)를 병렬로 구성한 **DB-SAM** 프레임워크를 제안하였다.
- **핵심 기술**: Channel Attention Block, Bilateral Cross-Attention, 그리고 자동 선택 융합 메커니즘을 통해 고품질의 의료 영상 특징을 추출한다.
- **결과**: 30개의 다양한 의료 영상 태스크에서 MedSAM 대비 3D DSC 평균 6.03%, NSD 평균 8.77% 향상을 달성하였다.
- **의의**: 본 연구는 범용 기초 모델(Foundation Model)을 특정 전문 도메인에 효율적으로 적응시키는 효과적인 Adapter 설계 방식을 제시하며, 향후 다양한 의료 영상 분석 도구의 성능 향상에 기여할 가능성이 크다.