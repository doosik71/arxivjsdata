# Multimodal Referring Segmentation: A Survey

Henghui Ding, Song Tang, Shuting He, Chang Liu, Zuxuan Wu, Yu-Gang Jiang (2025)

## 🧩 Problem to Solve

본 논문은 이미지, 비디오, 그리고 3D 장면과 같은 시각적 환경에서 텍스트나 오디오 형태의 Referring Expression(참조 표현)을 기반으로 대상 객체를 정밀하게 분할하는 'Multimodal Referring Segmentation(다중 모달 참조 분할)' 분야의 전반적인 연구 동향을 분석한다.

기존의 Semantic Segmentation이나 Instance Segmentation은 미리 정의된 카테고리 내에서만 객체를 분할할 수 있다는 한계가 있다. 반면, Referring Segmentation은 사용자가 자유로운 형식으로 기술한 문장이나 음성 신호를 통해 특정 객체를 지정하므로 훨씬 유연하고 사용자 친화적인 객체 인식이 가능하다. 그러나 이 작업은 텍스트/오디오와 시각 정보 간의 세밀한 정렬(Alignment)이 필요하며, 특히 복잡한 공간 관계, 속성, 동작 및 추론 능력을 요구한다는 점에서 기술적 난도가 높다.

또한, 기존의 서베이 논문들은 주로 2D 이미지나 특정 모달리티에 국한되어 있어, 이미지, 비디오, 3D 장면 및 다양한 입력 모달리티를 통합적으로 다루는 포괄적인 체계가 부재했다. 따라서 본 논문은 이 분야의 문제 정의, 데이터셋, 메타 아키텍처, 그리고 최신 방법론을 체계적으로 정리하여 연구자들에게 통합된 시각을 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **통합된 문제 정식화(Unified Formulation):** 다양한 참조 분할 작업을 하나의 수학적 프레임워크로 통합하였다. 입력 공간 $X$를 시각적 입력 $V$(이미지, 비디오, 3D 데이터 등)와 참조 신호 $E$(텍스트, 오디오 등)의 곱집합으로 정의하고, 이를 마스크 $Y$로 매핑하는 함수 $f: X \rightarrow Y$로 정의함으로써 서로 다른 작업 간의 공통점을 명확히 하였다.
2. **범용 메타 아키텍처(Unified Meta Architecture) 제시:** 특징 추출(Feature Extraction), 다중 모달 상호작용(Multimodal Interaction), 시간적 정보 처리(Temporal Processing), 그리고 분할 헤드(Segmentation Head)로 구성되는 일반적인 시스템 구조를 제안하여, 다양한 모델들이 어떤 공통 구조를 가지는지 분석하였다.
3. **방대한 연구 범위의 체계적 분석:** 600편 이상의 논문을 검토하여 이미지(RES), 비디오(RVOS, AVS), 3D(3D-RES) 장면뿐만 아니라, 다중 타겟 및 제로 타겟을 다루는 Generalized Referring Expression(GREx) 및 추론 기반 분할(Reasoning Segmentation)까지 광범위하게 다루었다.
4. **최신 트렌드 반영:** 특히 Segment Anything Model(SAM)과 Multimodal Large Language Models(MLLMs)가 참조 분할 분야에 어떻게 적용되고 있으며, 성능을 어떻게 향상시키고 있는지에 대한 최신 분석을 제공한다.

## 📎 Related Works

본 논문에서는 참조 분할의 기반이 되는 전통적인 분할 작업과의 차별점을 강조한다.

- **전통적 분할(Classic Segmentation):** Semantic 및 Instance Segmentation은 사전에 정의된 클래스(예: '사람', '자동차')를 기준으로 작동한다. Open-vocabulary segmentation이 카테고리 범위를 확장했으나, 여전히 명시적인 클래스 이름에 의존한다.
- **참조 분할(Referring Segmentation):** 자유 형식의 자연어 표현을 사용한다. 단순히 클래스 이름을 말하는 것이 아니라 "소파 맞은편에 있는 파란색 침대"와 같이 위치, 속성, 관계를 통해 객체를 유일하게 식별한다.

기존 서베이들은 주로 2D 이미지 기반의 Referring Expression Segmentation(RES)에 치중되어 있었으나, 본 논문은 이를 비디오(RVOS)와 3D 환경으로 확장하고, 오디오 모달리티까지 포함하는 Omnimodal 관점에서 접근함으로써 기존 연구들의 범위 제한 문제를 해결하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 메타 아키텍처

논문은 참조 분할 모델의 구조를 다음과 같은 단계로 설명한다.

- **특징 추출(Feature Extraction):**
  - **Vision Encoder:** 이미지의 경우 ViT나 CNN을 사용하며, 비디오는 3D CNN이나 Video Swin Transformer를, 3D 포인트 클라우드는 Sparse 3D U-Net나 PointNet을 사용하여 기하학적 특징을 추출한다.
  - **Text/Audio Encoder:** 텍스트는 BERT, RoBERTa 혹은 CLIP의 텍스트 인코더를 사용하고, 오디오는 VGGish나 wav2vec 같은 사전 학습된 모델을 통해 스펙트로그램이나 임베딩을 추출한다.

- **다중 모달 상호작용(Multimodal Interaction):**
  - **Fusion:** 단순 결합(Concatenation) 방식과 Attention 기반 방식이 있다. 최근에는 Cross-attention을 통해 텍스트 토큰과 시각적 영역 간의 동적인 정렬을 수행하는 방식이 주를 이룬다.
  - **Alignment:** Contrastive Learning(예: CLIP)을 통해 서로 다른 모달리티의 특징을 공통된 의미 공간에 배치하거나, Masked Multimodal Modeling을 통한 자기지도학습으로 정렬을 최적화한다.

- **시간적 정보 처리(Temporal Processing):** 비디오 작업의 경우 3D CNN, Temporal Attention, Memory Networks를 통해 프레임 간 일관성을 유지하고 객체의 움직임을 추적한다.

- **분할 헤드(Segmentation Head):** 추출된 융합 특징을 최종 마스크로 변환한다. 최근에는 SAM(Segment Anything Model)과 같은 프롬프트 기반 헤드를 도입하여 제로샷 성능을 높이는 추세이다.

### 2. 학습 목표 및 손실 함수

모델 학습에는 주로 다음과 같은 손실 함수가 사용된다.

- **분할 손실(Segmentation Loss):** 픽셀 단위의 분류 오류를 측정하는 Binary Cross-Entropy(BCE) 손실과 예측 마스크와 정답 마스크의 겹침 정도를 직접 최적화하는 Dice Loss가 결합되어 사용된다.
    $$\text{Dice Loss} = 1 - \frac{2|M_p \cap M_{gt}|}{|M_p| + |M_{gt}|}$$
- **정렬 및 그라운딩 손실(Alignment & Grounding Loss):** 시각적 영역과 텍스트 간의 대응 관계를 강화하기 위해 L1, L2, IoU 손실 또는 Contrastive Loss를 사용한다.

### 3. 주요 작업 정의

- **RES (Referring Expression Segmentation):** 이미지 내 텍스트 설명에 맞는 객체 분할.
- **RVOS (Referring Video Object Segmentation):** 비디오 내에서 텍스트 설명에 맞는 객체를 추적하며 분할.
- **AVS (Audio-Visual Segmentation):** 소리를 내는 객체를 분할.
- **OmniAVS:** 텍스트, 음성, 소리, 이미지 등 다양한 모달리티가 섞인 프롬프트를 통해 비디오 객체를 분할.
- **3D-RES:** 3D 포인트 클라우드 장면에서 참조 표현에 맞는 객체 분할.

## 📊 Results

### 1. 데이터셋 및 벤치마크

논문은 각 작업별 주요 데이터셋을 분석한다.

- **이미지:** RefCOCO, RefCOCO+, RefCOCOg (가장 널리 쓰임), ReasonSeg (추론 기반).
- **비디오:** MeViS (동작 중심), Ref-YouTube-VOS, Ref-DAVIS.
- **3D:** ScanRefer, Nr3D, Sr3D.
- **오디오-비디오:** AVSBench, Ref-AVS, OmniAVS.

### 2. 정량적 결과 분석

- **RES 성능:** 초기 LSTM-CNN 모델(mIoU $\approx 34\%$)에 비해 최근의 OneRef-L 모델은 RefCOCOg 테스트셋에서 $76.82\%$ mIoU를 달성하는 등 비약적인 발전을 보였다.
- **MLLM의 영향:** LISA, CoReS와 같은 MLLM 기반 모델들이 복잡한 추론이 필요한 Reasoning Segmentation에서 압도적인 성능을 보였다. 전통적 방식이 $20\text{--}30\%$ mIoU에 머문 반면, CoReS는 $68.1\%$ gIoU를 기록하였다.
- **RVOS 성능:** MeViS 데이터셋(동작 이해 중심)에서 GLUS 모델이 $51.30\%$ J&F score를 기록하며 SOTA를 달성하였다.
- **3D-RES 성능:** IPDN 모델이 ScanRefer 데이터셋에서 $60.60\%$ Acc@0.25를 기록하며 가장 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 발전 방향

- **범용 모델로의 전환:** 단순한 텍스트-이미지 쌍의 학습에서 벗어나 SAM2나 LLaVA와 같은 거대 파운데이션 모델을 어댑터 형태로 연결하여 제로샷 및 퓨샷 능력을 극대화하는 방향으로 발전하고 있다.
- **추론 능력의 결합:** 단순한 외형 묘사가 아닌 "단백질 함량이 가장 높은 음식"과 같이 외부 지식이 필요한 Reasoning Segmentation의 등장은 MLLM의 강력한 상식 추론 능력을 시각적 분할과 결합시킨 성과이다.

### 2. 한계 및 미해결 과제

- **모달리티 간 불균형:** 텍스트 기반 연구에 비해 오디오-비디오(AVS)나 3D-RES 연구의 데이터셋 규모와 방법론의 다양성이 상대적으로 부족하다.
- **실시간성 문제:** Transformer 기반의 거대 모델들은 높은 정확도를 보이지만, 계산 비용이 매우 커 실시간 애플리케이션(로봇 제어, 자율 주행 등)에 적용하기에는 한계가 있다.
- **복잡한 시나리오의 취약성:** 객체가 완전히 가려지거나(Occlusion), 사라졌다가 다시 나타나는 경우의 일관성 유지 문제가 여전히 해결해야 할 과제로 남아 있다.

### 3. 비판적 해석

본 논문은 매우 방대한 양의 연구를 체계적으로 정리하였으나, 각 모델의 세부적인 하이퍼파라미터나 학습 전략보다는 아키텍처의 분류와 결과 수치 제시에 집중하고 있다. 향후 연구에서는 단순히 성능 수치를 나열하는 것을 넘어, 왜 특정 아키텍처가 특정 데이터셋(예: 동작 중심의 MeViS)에서 더 잘 작동하는지에 대한 심층적인 분석(Ablation Study)이 더 보강될 필요가 있다.

## 📌 TL;DR

본 논문은 이미지, 비디오, 3D 장면에서 텍스트와 오디오 프롬프트를 통해 객체를 분할하는 **Multimodal Referring Segmentation** 분야의 포괄적인 서베이 보고서이다. 연구자는 전 분야를 아우르는 **통합 메타 아키텍처**를 제시하고, 최근 **SAM 및 MLLM**의 도입으로 인해 단순한 정렬(Alignment)을 넘어 복잡한 **추론 기반 분할(Reasoning Segmentation)**로 패러다임이 전환되고 있음을 분석하였다. 이 연구는 향후 Embodied AI, 지능형 로봇 제어 및 정밀한 이미지/비디오 편집 시스템 구축을 위한 핵심적인 이론적 토대를 제공한다.
