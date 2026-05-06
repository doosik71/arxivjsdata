# Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos

Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, Ming-Hsuan Yang (2025)

## 🧩 Problem to Solve

본 연구는 이미지와 비디오 모두에서 **Dense Grounded Understanding**(밀집된 접지 이해)을 달성하기 위한 통합 모델의 부재라는 문제를 해결하고자 한다. 기존의 멀티모달 거대 언어 모델(MLLMs)과 비디오 인식 모델들은 다음과 같은 상충되는 한계를 가지고 있다.

1. **비디오 인식 모델의 한계**: SAM-2와 같은 최신 기초 모델은 정교한 세그멘테이션과 트래킹 능력을 갖추고 있으나, 텍스트 기반의 열린 질문 답변(Video VQA)이나 대화와 같은 open-ended 능력이 부족하다.
2. **Video MLLM의 한계**: LLaVA와 같은 모델은 긴 비디오를 이해하고 고수준의 VQA를 수행할 수 있으나, 픽셀 수준의 정밀한 인지 작업(Perception tasks)이나 시각적 프롬프트를 이해하는 능력이 결여되어 있다.

결과적으로, 텍스트 묘사를 통해 비디오 내 특정 객체를 정밀하게 분할(Segmenting)하고 동시에 전체 장면을 이해하는 통합적인 능력을 갖춘 모델을 구축하는 것이 본 논문의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **SAM-2(기초 비디오 세그멘테이션 모델)**와 **LLaVA-like MLLM(고급 시각-언어 모델)**을 하나의 프레임워크로 통합하여, 텍스트, 이미지, 비디오를 공유된 LLM 토큰 공간으로 단일화하는 것이다.

- **Unified Framework**: LLM이 생성하는 특수 토큰인 `[SEG]` 토큰을 SAM-2 디코더의 spatial-temporal 프롬프트로 사용하여, 텍스트 지시사항을 정밀한 마스크(Mask) 생성으로 연결한다.
- **Decoupled Design**: SAM-2의 디코더와 메모리 모듈을 동결(Frozen) 상태로 유지함으로써 SAM-2가 가진 강력한 인지 및 트래킹 능력을 보존하고, MLLM의 업데이트에 따라 유연하게 교체 가능한 plug-and-play 구조를 설계하였다.
- **Ref-SAV Dataset**: 복잡한 비디오 장면에서 72k개 이상의 객체 표현을 포함하는 자동 레이블링 데이터셋인 Ref-SAV를 제안하여 모델의 성능을 고도화하고 벤치마크를 제공하였다.

## 📎 Related Works

- **Multi-modal Large Language Models (MLLMs)**: LLaVA와 같은 모델들은 시각적 특징을 토큰으로 처리하여 VQA 성능을 높였으며, 최근에는 LLaVA-OneVision처럼 이미지와 비디오를 통합하려는 시도가 있었다. 그러나 대부분은 픽셀 수준의 정밀한 grounding 능력이 부족하다.
- **Referring Segmentation**: LISA, GLaMM 등은 reasoning 기반의 세그멘테이션을 시도하였으나, 주로 이미지 도메인에 국한되었거나 비디오 도메인으로 확장 시 효율적인 통합이 이루어지지 않았다.
- **Video Segmentation and Grounding**: 기존의 비디오 세그멘테이션은 닫힌 집합(Closed set) 내에서 작동하거나, LLM의 광범위한 지식 공간을 활용하지 못하는 한계가 있었다. VISA와 같은 최신 연구가 등장했으나, 엔드-투-엔드 학습의 부재로 인해 확장성(Scalability)이 제한적이었다.

## 🛠️ Methodology

### 전체 시스템 구조

Sa2VA는 LLaVA-like MLLM과 SAM-2가 결합된 형태이다. 입력으로 텍스트, 이미지, 비디오, 그리고 시각적 프롬프트(박스나 포인트)가 들어오면, 이를 통합된 토큰 공간에서 처리하여 텍스트 응답과 세그멘테이션 마스크를 동시에 생성한다.

### 통합 태스크 표현 (Unified Task Representation)

모든 작업을 단일 지시 튜닝(One-shot instruction-tuning) 프로세스로 정식화하였다. 전체 프로세스는 다음과 같은 방정식으로 표현된다.

$$T_o, M_o = \text{LLM}(\{I_i, V_i, V P_i\}, T_i)$$

여기서 $T_i$는 입력 텍스트, $I_i$는 이미지, $V_i$는 비디오, $V P_i$는 시각적 프롬프트이며, $T_o$는 출력 텍스트, $M_o$는 생성된 마스크(또는 비디오의 경우 masklets)를 의미한다.

### 주요 구성 요소 및 작동 원리

1. **`[SEG]` 토큰**: LLM의 출력 중 `[SEG]` 토큰의 hidden state를 추출하여 SAM-2 디코더의 새로운 형태의 프롬프트로 입력한다. 이를 통해 LLM의 고수준 이해력이 SAM-2의 저수준 픽셀 제어 능력과 연결된다.
2. **마스크 트래킹 (Mask Tracking)**: 비디오의 경우, 모든 프레임을 처리하는 대신 주요 프레임(Key frames)에 대해 `[SEG]` 토큰으로 마스크를 먼저 생성한다. 이후 SAM-2의 메모리 인코더를 통해 나머지 프레임의 마스크를 전파하는 방식을 사용하여 효율성을 높였다.
3. **Ref-SAV 데이터 생성 파이프라인**:
    - **Object-level**: 객체 영역을 크롭하여 InternVL2-76B를 통해 상세 묘사 생성.
    - **Scene-level**: 객체와 주변 환경과의 관계를 묘사.
    - **Video-level**: 8개 프레임을 샘플링하여 객체의 움직임과 동작을 묘사.

### 학습 절차 및 손실 함수

모델은 이미지/비디오 QA 및 세그멘테이션 데이터셋을 사용하여 joint co-training을 수행한다. 손실 함수는 텍스트 생성 손실과 마스크 생성 손실의 합으로 정의된다.

$$L_{instruction} = L_{text} + L_{mask}$$
$$L_{mask} = L_{CE} + L_{DICE}$$

여기서 $L_{CE}$는 픽셀 단위의 Cross-Entropy 손실이며, $L_{DICE}$는 Dice loss를 사용하여 마스크의 정밀도를 높인다.

## 📊 Results

### 실험 설정

- **데이터셋**: RefCOCO, RefCOCO+, RefCOCOg (이미지), MeViS, Ref-DAVIS17, ReVOS (비디오), MME, MMBench (Chat) 등.
- **지표**: 이미지 세그멘테이션은 $\text{cIoU}$, 비디오 세그멘테이션은 $\text{J\&F}$ score를 사용하였다.
- **기준선**: GLaMM, VISA, OMG-LLaVA 등 최신 MLLM 및 전문가 모델과 비교하였다.

### 주요 결과

- **이미지/비디오 세그멘테이션**: Sa2VA-8B 모델은 RefCOCO+와 RefCOCOg에서 SOTA를 달성하였으며, 특히 비디오 작업인 MeViS, Ref-DAVIS17, ReVOS에서 이전 SOTA인 VISA-13B를 크게 상회하는 성능을 보였다.
- **대화 및 이해 능력**: 기존의 Grounding MLLM들이 세그멘테이션 성능을 높이면 대화 능력이 저하되는 경향이 있었으나, Sa2VA는 joint co-training을 통해 MME, MMBench 등에서 강력한 Chat 성능을 유지하였다.
- **모델 확장성 (Scalability)**: InternVL2-26B 기반의 Sa2VA-26B 모델은 이미지 및 비디오 세그멘테이션 모두에서 가장 높은 성능을 기록하여, 모델 크기가 증가함에 따라 성능이 향상됨을 입증하였다.
- **Ref-SAV 효과**: Ref-SAV 데이터셋으로 학습했을 때, 제로샷(Zero-shot) 대비 성능이 크게 향상되었으며, 이는 복잡한 묘사와 폐색(Occlusion) 상황에 대한 대응 능력이 강화되었음을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 SAM-2의 정밀한 인지 능력과 LLaVA의 범용적 이해 능력을 성공적으로 결합하였다. 특히, 복잡한 아키텍처 변경 없이 `[SEG]` 토큰이라는 단순한 연결 고리와 decoupled design을 통해 성능과 유연성을 동시에 잡았다는 점이 고무적이다. 또한, Ref-SAV라는 고난도 데이터셋을 통해 비디오 grounding 연구의 새로운 기준을 제시하였다.

### 한계 및 미해결 질문

- **긴 비디오 처리의 한계**: Sa2VA는 온라인 모드(Online mode)로 작동하여 비디오 전체 내용을 미리 알지 못한 채 처리한다. 이로 인해 매우 길고 복잡한 텍스트 묘사가 포함된 비디오에서는 정렬(Alignment) 문제가 발생할 수 있다.
- **Trade-off 문제**: ablation study 결과, VQA 데이터를 대량으로 추가할 경우 세그멘테이션 성능이 일부 하락하는 현상이 관찰되었다. 인지 능력과 이해 능력 사이의 최적의 균형을 찾는 방법은 여전히 해결해야 할 과제로 남아있다.

### 비판적 해석

본 연구는 SAM-2라는 강력한 기초 모델에 크게 의존하고 있다. SAM-2의 디코더를 동결한 설계는 구현의 단순함과 효율성을 제공하지만, LLM의 고수준 시맨틱 정보가 SAM-2의 디코더 내부로 더 깊게 통합되지 못하는 병목 현상을 야기할 가능성이 있다.

## 📌 TL;DR

Sa2VA는 SAM-2의 정밀한 픽셀 제어 능력과 LLaVA의 언어 이해 능력을 `[SEG]` 토큰을 통해 통합한 최초의 unified 모델이다. 이미지와 비디오 모두에서 Referring Segmentation과 Chat 작업을 동시에 수행할 수 있으며, 특히 새롭게 제안된 Ref-SAV 데이터셋을 통해 복잡한 비디오 환경에서의 grounding 성능을 획기적으로 높였다. 이 연구는 향후 인터랙티브한 비디오 분석 및 정밀한 시각적 제어가 필요한 로봇 내비게이션, 영상 편집 등의 분야에 중요한 기반이 될 것으로 기대된다.
