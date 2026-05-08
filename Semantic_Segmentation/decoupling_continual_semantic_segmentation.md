# Decoupling Continual Semantic Segmentation

Yifu Guo, Yuquan Lu, Wentao Zhang, Zishan Xu, Dexia Chen, Siyu Zhang, Yizhe Zhang, Ruixuan Wang (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 **Continual Semantic Segmentation (CSS)** 환경에서 발생하는 **Catastrophic Forgetting (치명적 망각)** 현상이다. CSS는 새로운 클래스를 지속적으로 학습하면서도 이전에 학습한 클래스에 대한 지식을 유지해야 하는 과제이다.

기존의 CSS 방법론들은 대부분 단일 단계의 Encoder-Decoder 아키텍처를 사용한다. 이러한 구조에서는 세그멘테이션 마스크(Intra-class consistency)와 클래스 레이블(Inter-class discrimination)이 모델 파라미터 내에서 긴밀하게 결합(Tightly coupled)되어 있다. 이로 인해 새로운 클래스를 학습할 때 기존 클래스의 경계와 특성을 구분하는 능력이 간섭을 받아 성능이 저하되며, 결과적으로 지식 유지(Retention)와 새로운 학습 능력(Plasticity) 사이의 균형을 맞추는 것이 매우 어렵다는 문제가 있다. 따라서 본 논문의 목표는 이러한 결합 구조를 해제(Decoupling)하여 망각을 최소화하고 학습 효율을 높이는 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **클래스 인식 탐지(Class-aware Detection)**와 **클래스 불가지론적 세그멘테이션(Class-agnostic Segmentation)**을 완전히 분리하는 두 단계 프레임워크인 **DecoupleCSS**를 제안하는 것이다.

중심적인 직관은 세그멘테이션 과정을 "이미지 내에 어떤 클래스가 존재하는지 찾는 것"과 "찾은 클래스의 영역을 정밀하게 획정하는 것"으로 나누는 것이다. 전자는 지속적 학습(Continual Learning)의 대상이 되어 새로운 클래스를 계속 추가 학습하고, 후자는 클래스 종류에 상관없이 영역을 따내는 공통 지식이므로 모든 태스크가 공유하도록 설계함으로써 상호 간섭을 제거한다. 이를 위해 사전 학습된 Vision-Language Model (VLM)과 Segment Anything Model (SAM)을 결합하여 활용한다.

## 📎 Related Works

기존의 CSS 접근 방식은 크게 세 가지로 분류된다:

1. **Data Replay**: 이전 데이터를 저장하여 함께 학습하지만, 메모리 증가와 개인정보 보호 문제가 있다.
2. **Regularization**: 중요 파라미터의 변화를 억제하지만, 유지와 가소성의 균형을 잡기 어렵다.
3. **Pseudo-labeling 및 Knowledge Distillation**: 이전 모델의 출력을 이용해 지식을 보존하려 하지만, 불확실한 픽셀이 배경(Background)으로 잘못 레이블링되는 배경 드리프트(Background drift) 문제가 발생한다.

본 논문은 최근 Mask2Former 기반의 CoMasTRe나 CoMFormer와 같이 디커플링 전략을 탐색한 연구들이 있으나, DecoupleCSS는 "탐지 후 세그멘테이션(Detection-then-Segmentation)"이라는 더 명확한 패러다임을 따르며, 특히 VLM과 SAM이라는 파운데이션 모델을 CSS 영역에 밀도 있게 결합했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

DecoupleCSS는 크게 두 단계(Stage)로 구성된다.

### 1. Stage-I: Language-driven Task-aware Class Detection (LTCD)

이 단계의 목적은 입력 이미지에서 어떤 클래스가 존재하는지 탐지하고, 이를 위한 위치 기반 프롬프트를 생성하는 것이다.

- **Text Encoding**: LLM을 사용하여 각 클래스에 대해 "형용사 + 클래스 이름" 형태의 간결한 묘사 문구 $M+1$개를 생성한다. 이를 텍스트 인코더로 임베딩하여 $\{g_{k,1}, \dots, g_{k,M+1}\}$을 얻는다.
- **Adaptive Re-weighting**: 입력 이미지의 시각적 특성에 맞게 텍스트 임베딩의 가중치를 조정한다. 이미지의 $\text{cls}$ 토큰 $V_{cls}^i$와 텍스트 임베딩 $g_{k,j}$ 사이의 코사인 유사도 $s_{i,k,j}$를 이용해 가중치 $\alpha_{i,k,m}$을 계산한다.
$$\alpha_{i,k,m} = \frac{\exp(s_{i,k,m})}{\sum_{j=1}^{M+1} \exp(s_{i,k,j})}$$
최종 텍스트 임베딩 $e_{i,k}$는 다음과 같이 가중 합으로 계산된다:
$$e_{i,k} = \sum_{m=1}^{M+1} \alpha_{i,k,m} \cdot g_{k,m}$$
- **Task-Specific LoRAs**: Swin Transformer 기반 이미지 인코더와 Cross-attention 모듈에 LoRA(Low-Rank Adaptation) 어댑터를 추가한다. 각 태스크마다 독립적인 어댑터를 사용하여 이전 태스크의 지식을 보존하고 간섭을 방지한다.
- **Semantic Alignment**: 시각적 토큰 $V'_i$와 텍스트 임베딩 $E'_i$ 간의 어피니티 행렬(Affinity Matrix) $S^i$를 계산한다.
$$S^i = \cos(V'_i, E'^iT)$$
임계값 $\tau$를 적용하여 강한 연관성을 가진 토큰들만 선택하여 $V_{sel}^i$를 구성하며, 각 토큰에 대해 가장 유사도가 높은 클래스를 할당한다.

### 2. Stage-II: Class-Specific Prompt Generation (SPG) & Class-Agnostic Segmentation (CAS)

탐지된 클래스 정보를 바탕으로 정밀한 마스크를 생성하는 단계이다.

- **Prompt Generation (pGen)**: 각 클래스별로 전용 MLP인 $\text{pGen}_k$를 둔다. 선택된 시각적 토큰 $T^k$와 학습 가능한 클래스 임베딩 $L^k$를 결합하여 SAM의 입력으로 사용할 위치 프롬프트 $p^k$를 생성한다.
$$p^k = \text{pGen}_k(\hat{z}_k)$$
여기서 클래스별 독립적인 $\text{pGen}$을 사용하는 것이 클래스 간 간섭을 막는 핵심이다.
- **Class-Agnostic Segmentation (CAS)**: 고정(Frozen)된 SAM을 사용하여 생성된 프롬프트 $p^k$를 기반으로 마스크 $M^k$를 생성한다. 최종 세그멘테이션 맵은 각 클래스 마스크를 신뢰도(Confidence score) 기반으로 병합하여 완성한다.

### 3. 학습 절차 및 손실 함수

모델은 다음과 같은 손실 함수의 조합으로 학습된다:

1. **Segmentation Loss**: SAM의 학습 방식과 동일한 Dice Loss와 Cross-Entropy (CE) Loss의 선형 결합을 사용하여 $\text{pGen}$과 LoRA 어댑터를 최적화한다.
$$\mathcal{L} = \mathcal{L}_{Dice} + \mathcal{L}_{CE}$$
2. **Asymmetric Loss (ASL)**: 클래스 불균형 문제를 해결하기 위해 이미지 카테고리 분류 결과에 대해 ASL을 보조 손실 함수로 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC2012 (20개 클래스), ADE20K (150개 클래스).
- **설정**: 가장 도전적인 Overlapped 설정(이전/미래 클래스가 배경에 섞여 있는 환경)을 사용하였다.
- **지표**: mIoU (mean Intersection over Union).

### 주요 결과

- **PASCAL VOC2012**: 10-1 설정(11개 태스크)에서 **83.12% mIoU**를 달성하여, 데이터 리플레이 방식의 SOTA인 IPSeg 대비 4.52%p 높은 성능을 보였다.
- **ADE20K**: 100-5 설정에서 SOTA 대비 **17.99%p라는 압도적인 mIoU 향상**을 기록하였다.
- **극한 설정 테스트**: 2-2, 4-2, 4-4와 같이 초기 학습 클래스 수가 매우 적은 극한의 설정에서도 기존 방법론들을 수십 %p 차이로 압도하며 매우 높은 안정성과 가소성을 입증하였다.

### 분석 및 어블레이션

- **구성 요소의 중요성**: LoRA, $\text{pGen}$, 시맨틱 어그리게이션 중 하나라도 제거했을 때 성능이 크게 하락하였다.
- **Class-specific vs Shared pGen**: 모든 클래스가 공유하는 $\text{pGen}$을 사용했을 때, 새로운 클래스 학습 시 기존 지식이 파괴되는 현상이 관찰되어 클래스별 전용 $\text{pGen}$의 필요성이 증명되었다.

## 🧠 Insights & Discussion

### 강점

본 연구의 가장 큰 성과는 세그멘테이션의 '무엇(What)'과 '어디(Where)'를 분리하여, 지속적 학습의 부담을 '탐지' 단계로 한정시킨 것이다. 특히 SAM이라는 강력한 클래스 불가지론적 모델을 활용함으로써, 세그멘테이션 성능 자체는 유지하면서 클래스 확장성만 확보하는 전략이 유효했음을 보여주었다.

### 한계 및 논의사항

1. **추론 시간 (Inference Latency)**: 각 태스크별 LoRA 어댑터를 순차적으로 교체하며 실행해야 하므로, 태스크 수 $T$에 비례하여 추론 시간이 선형적으로 증가한다. (예: 6개 태스크 시 이미지당 약 1.9초 소요)
2. **메모리-시간 트레이드오프**: 파라미터 증가량은 매우 적지만(LoRA 및 $\text{pGen}$은 전체 모델의 1% 미만), 실시간 응용 분야에 적용하기 위해서는 파라미터 병합(Parameter merging) 등의 추가 기술이 필요할 것으로 보인다.
3. **SAM 의존성**: SAM의 성능이 매우 뛰어나지만, 본 논문은 단순히 SAM을 붙이는 것이 아니라 정교한 프롬프트 생성기($\text{pGen}$)가 필수적임을 실험적으로 증명하였다.

## 📌 TL;DR

DecoupleCSS는 지속적 세그멘테이션(CSS)의 고질적인 문제인 망각 현상을 해결하기 위해 **[클래스 탐지 $\rightarrow$ 영역 세그멘테이션]**으로 파이프라인을 분리한 2단계 프레임워크이다. VLM과 LoRA를 통해 클래스 탐지 능력을 확장하고, 고정된 SAM을 통해 정밀한 마스크를 생성함으로써 PASCAL VOC와 ADE20K 데이터셋에서 SOTA 성능을 달성하였다. 특히 태스크 간 간섭을 최소화하는 구조 덕분에 극한의 학습 설정에서도 매우 강력한 성능을 보이며, 향후 자율주행이나 의료 영상 분석과 같은 실용적 CSS 분야에 중요한 방향성을 제시한다.
