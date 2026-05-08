# OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding

Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, Shuicheng Yan (2024)

## 🧩 Problem to Solve

본 논문은 시각적 이해의 세 가지 계층인 이미지 수준(Image-level), 객체 수준(Object-level), 픽셀 수준(Pixel-level)의 추론 및 이해를 하나의 모델로 통합하고자 한다.

기존의 범용 세그멘테이션(Universal Segmentation) 방법론들은 픽셀 수준의 이해 능력은 뛰어나지만, 텍스트 지시어에 따른 제어나 고차원적인 추론 능력이 부족하다. 반면, LLaVA와 같은 대규모 시각-언어 모델(MLLM)은 강력한 대화 및 추론 능력을 갖추고 있으나, 정밀한 픽셀 수준의 이해가 어렵고 사용자의 유연한 상호작용을 위한 시각적 프롬프트(Visual Prompts) 수용 능력이 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 단 하나의 LLM, 하나의 시각 인코더(Visual Encoder), 그리고 하나의 시각 디코더(Visual Decoder)만을 사용하는 간결하고 우아한(elegant) 프레임워크를 통해 위 세 가지 수준의 이해와 추론을 모두 수행하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 범용 인식 모델인 OMG-Seg를 시각 인코더 및 디코더로 활용하여 MLLM과 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **통합 아키텍처 설계**: 이미지-객체-픽셀 수준의 작업을 단일 모델로 통합하여, 이미지 캡셔닝(Image-level), 영역 캡셔닝(Object-level), 그리고 추론 기반 세그멘테이션(Pixel-level) 등을 모두 수행할 수 있게 하였다.
2. **Perception Prior Embedding 제안**: 고정된(frozen) 인식 모듈과 LLM 사이의 간극을 메우기 위해, 세그멘테이션 사전 정보를 이미지 특징에 통합하는 임베딩 전략을 제안하였다.
3. **유연한 시각적 프롬프트 수용**: 점(Point), 박스(Box), 마스크(Mask) 형태의 다양한 시각적 프롬프트를 효율적으로 처리할 수 있는 메커니즘을 구현하였다.
4. **효율적인 학습 파이프라인**: 사전 학습(Pre-training)과 지시어 튜닝(Instruction Tuning)의 2단계 과정을 통해 모델의 성능을 최적화하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별점을 제시한다.

- **MLLMs (e.g., LLaVA)**: 이미지 수준의 분석에는 능숙하지만 정밀한 위치 정보나 픽셀 수준의 출력을 생성하지 못한다.
- **Grounding/Segmentation MLLMs (e.g., LISA, GLaMM)**: 픽셀 수준의 출력을 위해 SAM과 같은 거대 모델이나 복잡한 추가 모듈을 사용하며, 이 과정에서 LLaVA가 가졌던 일반적인 이미지 수준의 분석 능력을 상실하는 경우가 많다. 특히 GLaMM의 경우 시스템 복잡도가 매우 높고 중복성이 크다.
- **Unified Segmentation Models (e.g., OMG-Seg)**: 다양한 세그멘테이션 작업을 하나의 모델로 통합했으나, MLLM과 같은 대화형 텍스트 생성 및 추론 능력은 갖추고 있지 않다.

OMG-LLaVA는 이러한 기존 연구들과 달리, 최소한의 구성 요소(1 Encoder, 1 Decoder, 1 LLM)만을 사용하여 세 가지 수준의 이해를 모두 달성함으로써 시스템의 효율성과 기능성을 동시에 확보하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 토큰 정의

OMG-LLaVA는 모든 작업을 '토큰-대-토큰 생성' 문제로 모델링한다. 이를 위해 세 가지 유형의 토큰을 정의한다.

- **텍스트 토큰 ($T_t$)**: 일반적인 텍스트 정보 인코딩.
- **픽셀 중심 시각 토큰 (Pixel-centric visual tokens, $T_{pv}$)**: 이미지의 조밀한 특징(dense features)을 나타내어 LLM에 포괄적인 이미지 정보를 제공한다.
- **객체 중심 시각 토큰 (Object-centric visual tokens, $T_{ov}$)**: 특정 객체의 특징을 인코딩하며, 이후 세그멘테이션 마스크로 쉽게 디코딩될 수 있다.

전체 프로세스는 다음과 같은 수식으로 정의된다:
$$T_{out}^t, T_{out}^{ov} = \text{LLM}(T_{in}^{pv}, T_{in}^{ov}, T_{in}^t)$$

### 2. 주요 구성 요소 및 역할

- **Image Encoder**: ConvNeXt-L 기반의 CLIP 모델을 사용하며, $1024 \times 1024$ 고해상도 이미지를 입력받는다. 계산 비용을 줄이기 위해 픽셀 셔플(Pixel Shuffle) 연산 등을 통해 최종적으로 $16 \times 16 = 256$개의 시각 토큰을 생성한다.
- **OMG Decoder**: 학습 가능한 객체 쿼리(Object Queries)와 시각적 프롬프트 쿼리를 입력으로 받아 객체 중심 토큰을 생성한다. 특히 박스 및 마스크 프롬프트의 경우, 해당 영역 외의 픽셀 특징에 대해 Attention Mask를 적용하여 사용자의 의도를 정밀하게 인코딩한다.
- **Perception Prior Embedding**: 고정된 인식 모듈과 LLM을 효과적으로 연결하기 위한 핵심 모듈이다.
    1. 객체 쿼리 $Q$로부터 얻은 세그멘테이션 마스크 $M$과 신뢰도 점수 $S$를 사용하여 픽셀별 마스크 점수 $MS$를 계산한다.
        $$MS = \text{Softmax}(M \odot S, \text{dim}=-1)$$
    2. 이 $MS$를 가중치로 하여 객체 쿼리의 가중 평균을 구하고, 이를 이미지 특징 $F$에 더해 픽셀 중심 토큰 $T_{pv}$를 생성한다.
        $$T_{pv} = MS \cdot Q + F$$
    3. 전경(foreground) 객체 쿼리는 $T_{ov}$로 처리되어 $T_{pv}$와 함께 LLM의 입력으로 들어간다.

### 3. 학습 절차 및 손실 함수

학습은 두 단계로 진행되며, 인식 모듈(인코더 및 디코더)은 모든 단계에서 고정(frozen) 상태를 유지한다.

**1단계: 사전 학습 (Pre-training)**
시각적 토큰을 LLM의 텍스트 임베딩 공간으로 매핑하는 Projector를 학습시킨다. 텍스트 회귀 손실과 더불어, 객체 중심 정보의 손실을 막기 위한 정규화 패널티($L_{reg}$)를 적용한다.
$$L_{pretrain} = L_{text} + L_{reg}, \quad L_{reg} = (T_{ov} - P_t(P_v(T_{ov})))^2$$

**2단계: 지시어 튜닝 (Instruction Tuning)**
LoRA를 사용하여 LLM을 미세 조정하고 Projector를 함께 학습시킨다. 텍스트 손실 외에도, $[SEG]$ 토큰으로 생성된 마스크의 정확도를 높이기 위해 Cross-Entropy 손실과 Dice 손실을 사용한다.
$$L_{instruction} = L_{text} + L_{mask}, \quad L_{mask} = \alpha L_{CE} + \beta L_{DICE}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LLaVA(이미지 수준), Osprey/MDVP(객체 수준), refCOCO/+/g 및 ADE20k/COCO-stuff(픽셀 수준), GranDf(Grounded Conversation) 등을 사용하였다.
- **기준 모델**: InternLM2-7B (LLM), ConvNeXt-L CLIP (Encoder).
- **지표**: cIoU (Referring Segmentation), METEOR, CIDEr, AP50, mIoU (Grounded Conversation) 등을 사용하여 정량 평가를 수행하였다.

### 2. 주요 결과

- **범용 능력**: OMG-LLaVA는 단일 모델로 8가지 이상의 멀티모달 작업을 수행하며, 특히 referring segmentation과 grounded conversation generation에서 SOTA 수준의 성능을 보였다.
- **Referring Expression Segmentation (RES)**: refCOCO 셋에서 LISA 대비 1.5 cIoU 높은 성능을 보였으며, 디코더를 해제하고 미세 조정했을 때는 LISA보다 최대 5.0 cIoU 더 높은 성능을 기록하였다.
- **Grounded Conversation Generation (GCG)**: LISA보다 이미지 묘사 능력(METEOR, CIDEr)과 픽셀 이해 능력(AP50, mIoU) 모두에서 우수하였으며, 훨씬 더 많은 데이터를 사용한 GLaMM과 비교해서도 대등하거나 일부 능가하는 결과를 보였다.
- **이미지 수준 벤치마크**: MME, MMBench 등에서 LLaVA-1.5보다 높은 점수를 기록하여, 픽셀 수준 학습이 이미지 수준의 이해 능력을 저하시키지 않았음을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **아키텍처의 간결함**: 다수의 인코더나 디코더를 사용하는 복잡한 시스템 대신, 단일 구성 요소들의 조합만으로 최상위 성능을 낸 것은 매우 효율적인 설계이다.
- **Perception Prior의 중요성**: Ablation study 결과, 단순 결합(Baseline)보다 Perception Prior Embedding을 적용했을 때 RES 성능이 비약적으로 상승(최대 13.8 cIoU)하였다. 이는 LLM이 고정된 인식 모듈의 특성을 이해하도록 돕는 사전 정보 주입이 필수적임을 시사한다.
- **답변 형식의 통일**: "Sure, it is [SEG]"와 같은 고정된 답변보다 "<p> Expression </p> [SEG]"와 같이 유연한 형식을 사용할 때 LLM의 지시어 수행 능력이 더 잘 유지됨을 발견하였다.

### 2. 한계 및 향후 과제

- **능력 간의 충돌**: 픽셀 수준 데이터로 학습 시 이미지 수준 능력이 일부 감소하는 경향이 발견되었으며, 이를 해결하기 위한 데이터 구성 최적화가 필요하다.
- **세분화 능력 부족**: OMG-Seg의 한계로 인해 부분(part-level) 세그멘테이션은 불가능하며, 이를 위해 더 강력한 인식 모듈 도입이 필요하다.
- **시공간 추론**: 현재는 정지 영상 중심이며, 비디오 수준의 픽셀 추론을 위한 데이터셋 확보와 학습이 향후 과제로 남아 있다.

## 📌 TL;DR

OMG-LLaVA는 단 하나의 시각 인코더, 디코더, 그리고 LLM만을 사용하여 **이미지-객체-픽셀 수준의 이해와 추론을 통합한 효율적인 MLLM**이다. 특히 **Perception Prior Embedding**을 통해 인식 모듈의 사전 정보를 LLM에 효과적으로 전달함으로써, 복잡한 구조 없이도 정밀한 세그멘테이션과 고차원적 추론을 동시에 수행할 수 있음을 증명하였다. 이 연구는 향후 MLLM 설계 시 모델의 구성 요소를 최소화하면서 기능을 최대화하는 새로운 기준(baseline)을 제시한다.
