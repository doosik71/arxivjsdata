# Decoupling Continual Semantic Segmentation

Yifu Guo, Yuquan Lu, Wentao Zhang, Zishan Xu, Dexia Chen, Siyu Zhang, Yizhe Zhang, Ruixuan Wang (2026)

## 🧩 Problem to Solve

본 논문은 **Continual Semantic Segmentation (CSS)**에서 발생하는 **Catastrophic Forgetting (치명적 망각)** 문제를 해결하고자 한다. CSS는 새로운 클래스를 순차적으로 학습하면서 이전에 학습한 지식을 유지해야 하는 과제이다.

기존의 CSS 방법론들은 주로 단일 단계(Single-stage)의 Encoder-Decoder 아키텍처를 사용한다. 이러한 구조에서는 세그멘테이션 마스크(Intra-class consistency)와 클래스 레이블(Inter-class discrimination)이 모델 파라미터 내에서 강하게 결합(Tightly coupled)되어 있다. 이로 인해 새로운 클래스를 학습할 때 기존 클래스와의 구분 능력이 상실되거나, 이전 지식이 덮어씌워지는 간섭 현상이 발생하며, 결과적으로 지식 유지(Retention)와 새로운 학습 능력(Plasticity) 사이의 균형을 맞추는 데 어려움이 있다.

따라서 본 논문의 목표는 클래스 인식 단계와 실제 세그멘테이션 단계를 분리하여, 새로운 클래스 학습이 기존 지식에 미치는 영향을 최소화하면서도 높은 세그멘테이션 성능을 유지하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **DecoupleCSS**라는 새로운 2단계 프레임워크를 제안한다. 핵심 아이디어는 **Class-aware Detection (클래스 인식 탐지)**과 **Class-Agnostic Segmentation (CAS, 클래스 불가지론적 세그멘테이션)**을 완전히 분리(Decoupling)하는 것이다.

1. **프레임워크 수준의 기여**: 클래스 인식 탐지(Stage-I)와 영역 분할(Stage-II)을 분리하여, 지속적 학습(Continual Learning)의 대상은 탐지 단계로 한정하고 세그멘테이션 지식은 모든 클래스가 공유하도록 설계하였다. 특히 SAM(Segment Anything Model)과 같은 Foundation Model을 CSS의 기반으로 활용하는 관점을 제시하였다.
2. **방법론 수준의 기여**:
    - 언어 모델(LLM)을 활용하여 클래스별 특성을 반영한 **Language-driven Task-aware Class Detection (LTCD)** 모듈을 제안하였다.
    - 탐지된 결과를 바탕으로 SAM을 정밀하게 제어하기 위한 **Segmentation Prompt Generation (SPG)** 전략을 통해 클래스별 맞춤형 프롬프트를 생성한다.
3. **성능적 기여**: PASCAL VOC2012 및 ADE20K 데이터셋의 다양한 설정에서 기존 SOTA(State-of-the-art) 방법론들을 크게 상회하는 성능을 입증하였으며, 특히 매우 도전적인 설정(극소수 클래스 순차 학습)에서 압도적인 우위를 보였다.

## 📎 Related Works

기존의 CSS 연구들은 주로 다음과 같은 접근 방식을 취해왔다.

- **Data Replay**: 이전 데이터를 저장하여 함께 학습하는 방식이나, 메모리 증가와 개인정보 보호 문제가 있다.
- **Regularization**: 중요 파라미터의 변화를 억제하지만, 유지(Retention)와 가소성(Plasticity)의 균형을 잡기 어렵다.
- **Pseudo-labeling & Knowledge Distillation**: 이전 모델의 출력을 활용해 지식을 보존하려 하지만, 불확실한 픽셀이 배경으로 처리되면서 발생하는 Background Class Drift 문제가 존재한다.

본 논문은 이러한 기존 방식들이 대부분 단일 단계의 Encoder-Decoder 구조에 기반하여 마스크와 레이블이 결합되어 있다는 점을 지적한다. 반면, DecoupleCSS는 탐지와 분할을 분리함으로써 클래스 간 간섭을 원천적으로 차단하고, Foundation Model의 강력한 제로샷 분할 능력을 활용한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

DecoupleCSS는 크게 두 단계로 구성된다.

- **Stage-I (LTCD)**: 입력 이미지에서 어떤 클래스가 존재하는지 탐지하고, 해당 클래스의 위치 정보를 담은 시각적 임베딩을 추출한다.
- **Stage-II (CAS)**: Stage-I에서 생성된 클래스별 프롬프트를 입력받아 SAM이 정밀한 세그멘테이션 마스크를 생성하게 한다.

### 2. Language-driven Task-aware Class Detection (LTCD)

이 모듈은 사전 학습된 텍스트 및 이미지 인코더를 기반으로 하며, 각 태스크별로 최적화된 LoRA(Low-Rank Adaptation) 어댑터를 사용한다.

**텍스트 인코딩 및 적응적 가중치 합산**:
각 클래스에 대해 LLM을 통해 $M+1$개의 묘사 문구(Descriptive phrases)를 생성한다. 단순한 클래스 이름이 아닌 '형용사 + 클래스 이름' 형태의 간결한 구를 사용하여 시각적 변별력을 높인다. 입력 이미지 $x_i$에 대해 각 텍스트 임베딩 $g_{k,j}$의 가중치 $\alpha_{i,k,m}$는 다음과 같이 계산된다.

$$s_{i,k,j} = \cos(V_{cls}^i, g_{k,j})$$
$$\alpha_{i,k,m} = \frac{\exp(s_{i,k,m})}{\sum_{j=1}^{M+1} \exp(s_{i,k,j})}$$

최종 텍스트 임베딩 $e_{i,k}$는 이미지별로 다르게 생성되는 가중치 합으로 표현된다.
$$e_{i,k} = \sum_{m=1}^{M+1} \alpha_{i,k,m} \cdot g_{k,m}$$

**Task-Specific LoRAs**:
Swin Transformer 기반 이미지 인코더와 Grounding DINO의 Cross-attention 모듈에 태스크별 LoRA 어댑터를 추가한다. 이를 통해 각 태스크의 새로운 클래스에 특화된 특징을 추출하면서도, 이전 태스크의 어댑터를 보존함으로써 망각을 방지한다.

**Semantic Alignment**:
시각적 토큰 $V'_i$와 텍스트 임베딩 $E'_i$ 사이의 어피니티 행렬(Affinity Matrix) $S^i$를 생성한다.
$$S^i = \cos(V'_i, E'_i^T) \in \mathbb{R}^{N \times c_t}$$
임계값 $\tau$를 적용하여 강하게 정렬된 토큰만을 선택($V_{sel}^i$)하며, 각 토큰에 대해 가장 높은 유사도를 가진 클래스를 할당한다.

### 3. Class-Specific Prompt Generation (SPG)

선택된 시각적 임베딩 $V_{sel}^i$를 SAM의 입력으로 사용할 수 있는 위치 프롬프트로 변환하는 과정이다.

- **Channel Selection**: 각 클래스 $k$에 해당하는 토큰 집합 $T_k$를 추출한다.
- **pGen (Prompt Generator)**: $T_k$를 고정 길이로 변환하고 학습 가능한 클래스별 임베딩 $L_k$를 더한 뒤, 클래스 전용 MLP인 $\text{pGen}_k$를 통해 SAM용 포지셔널 프롬프트 $p^k$를 생성한다.
$$p^k = \text{pGen}_k(\hat{z}_k)$$
클래스별로 독립적인 MLP를 사용함으로써 클래스 간의 간섭을 방지하고 각 객체의 고유한 마스크 패턴을 학습한다.

### 4. Class-Agnostic Segmentation (CAS) 및 통합

- **SAM 활용**: 생성된 프롬프트 $p^k$를 입력받은 SAM은 클래스 정보와 무관하게 정밀한 마스크 $M^k$를 생성한다. CAS 모듈은 전체 학습 과정 동안 동결(Frozen) 상태로 유지된다.
- **최종 통합**: 각 클래스별 마스크를 통합하여 최종 세그멘테이션 맵을 생성하며, 픽셀 중첩이 발생할 경우 SAM 디코더의 신뢰도 점수(Confidence score)가 가장 높은 클래스에 할당한다.

### 5. 학습 목표 및 손실 함수

- **마스크 손실**: SAM의 학습 방식과 동일하게 Dice Loss와 Cross-Entropy(CE) Loss의 조합을 사용하여 $\text{pGen}$과 LoRA 어댑터를 학습시킨다.
$$\mathcal{L} = \mathcal{L}_{Dice} + \mathcal{L}_{CE}$$
- **탐지 손실**: 클래스 불균형 문제를 해결하기 위해 Asymmetric Loss (ASL)를 보조 손실 함수로 사용하여 클래스 탐지 성능을 최적화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: PASCAL VOC2012, ADE20K.
- **설정**: Overlapped protocol (이미지에 이전/미래 클래스가 섞여 있어 가장 어려운 설정).
- **평가 지표**: mIoU (mean Intersection over Union).

### 2. 정량적 결과

- **PASCAL VOC2012**: 10-1 설정(11개 태스크)에서 **83.12% mIoU**를 달성하여, 데이터 리플레이 기반의 SOTA인 IPSeg보다 4.52% 높은 성능을 보였다.
- **ADE20K**: 100-5 설정에서 기존 SOTA 대비 **17.99% mIoU**라는 압도적인 성능 향상을 기록했다.
- **극한 설정(Challenging Settings)**: 2-2, 4-2, 4-4와 같이 초기 학습 클래스 수가 매우 적은 설정에서 기존 방법론들은 성능이 급락하는 반면, DecoupleCSS는 각각 83.20%, 83.50%, 82.81%의 높은 성능을 유지하며 기존 대비 최대 74.94%의 격차를 벌렸다.

### 3. 분석 결과

- **Ablation Study**: LoRA, class-specific pGen, semantic aggregation 세 가지 요소가 모두 제거되었을 때 성능이 크게 하락함을 확인하였으며, 특히 pGen을 클래스 공유(shared) 방식으로 바꾸었을 때 성능이 급격히 떨어졌다.
- **SAM의 역할**: 단순히 SAM을 후처리 모듈로 사용하는 것은 효과가 미미했으나(6% 미만 향상), 본 논문의 프롬프트 생성 모듈을 통한 가이드는 매우 효과적임을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

본 연구의 가장 큰 성과는 **'탐지(Detection)'와 '분할(Segmentation)'의 분리**를 통해 CSS의 고질적인 문제인 Stability-Plasticity Dilemma를 효과적으로 해결했다는 점이다. 클래스별 독립적인 $\text{pGen}$과 태스크별 LoRA를 사용함으로써 새로운 지식을 흡수하는 능력(Plasticity)을 극대화하는 동시에, 기존 지식을 덮어쓰지 않는 안정성(Stability)을 확보하였다.

### 2. 한계 및 비판적 해석

- **추론 시간(Inference Time)**: 태스크별로 LoRA 어댑터를 순차적으로 교체하며 추론해야 하므로, 단일 모델 기반의 기존 CSS 방법론(약 0.5초/이미지)보다 느린 추론 시간(6개 태스크 기준 약 1.9초/이미지)이 발생한다. 이는 실시간 응용 분야에 적용하는 데 제약이 될 수 있다.
- **파라미터 증가**: 클래스별로 $\text{pGen}$ MLP가 추가되므로 클래스 수가 늘어날수록 저장 공간이 선형적으로 증가한다. 비록 개별 크기는 작으나(클래스당 8~10MB), 수천 개의 클래스를 학습해야 하는 환경에서는 문제가 될 수 있다.

### 3. 결론적 논의

그럼에도 불구하고, Foundation Model(SAM, VLM)의 능력을 CSS 프레임워크 내에서 적절히 디커플링하여 활용한 설계는 매우 영리한 접근이다. 특히 데이터 리플레이 없이도 데이터 리플레이 기반 모델보다 높은 성능을 낸다는 점은 실용적 가치가 매우 높다.

## 📌 TL;DR

DecoupleCSS는 **클래스 인식 탐지(Stage-I)**와 **클래스 불가지론적 세그멘테이션(Stage-II)**을 분리하여 치명적 망각을 방지하는 CSS 프레임워크이다. VLM 기반의 텍스트-이미지 정렬과 태스크별 LoRA를 통해 클래스를 탐지하고, 이를 기반으로 SAM에 최적화된 프롬프트를 생성하여 정밀한 마스크를 얻는다. PASCAL VOC와 ADE20K 데이터셋에서 SOTA를 달성했으며, 특히 클래스 수가 적은 극한의 순차 학습 설정에서 압도적인 성능을 보여주어 향후 자율주행 및 의료 영상 분석과 같은 실제 환경의 지속적 학습 연구에 중요한 기여를 할 것으로 기대된다.
