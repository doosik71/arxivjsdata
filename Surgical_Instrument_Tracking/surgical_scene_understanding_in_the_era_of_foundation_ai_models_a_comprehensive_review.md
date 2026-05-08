# Surgical Scene Understanding in the Era of Foundation AI Models: A Comprehensive Review

Ufaq Khan, Umair Nawaz, Adnan Qayyum, Shazad Ashraf, Muhammad Bilal, Junaid Qadir (2025)

## 🧩 Problem to Solve

본 논문은 최소 침습 수술(Minimally Invasive Surgery, MIS) 환경에서 수술 장면 이해(Surgical Scene Understanding)를 위해 인공지능을 통합하는 과정에서 발생하는 기술적, 윤리적 도전 과제를 해결하고자 한다. 수술 장면의 시야(Surgical Field of View, SFOV)는 가변적인 조명 조건, 급격한 기구 움직임으로 인한 모션 블러(Motion Blur), 혈액, 수술 중 발생하는 연기(Smoke) 및 기타 체액으로 인한 시야 방해 등 매우 복잡하고 예측 불가능한 특성을 가진다.

전통적인 영상 처리 방법론은 이러한 변동성에 대응하는 능력이 부족하여 실시간 응용에 한계가 있다. 따라서 본 연구의 목표는 최신 머신러닝(ML) 및 딥러닝(DL) 기술, 특히 최근 등장한 파운데이션 모델(Foundation Models, FMs)을 수술 워크플로우에 통합하여 세그멘테이션(Segmentation) 정확도를 높이고, 기구 추적(Instrument Tracking) 및 수술 단계 인식(Phase Recognition) 능력을 향상시키는 방안을 종합적으로 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 전통적인 ML/DL 기법을 넘어, 최신 파운데이션 모델(FMs)이 수술 장면 이해에 미치는 변혁적인 역할을 체계적으로 분석했다는 점에 있다.

- **파운데이션 모델 중심의 현대적 분석**: Vision Transformers(ViTs), Large Vision-Language Models(LVLMs), Segment Anything Model(SAM)과 같은 최신 모델들이 어떻게 수술 장면의 가변성과 환자 간 이질성 문제를 해결하고 일반화 성능을 높이는지 분석하였다.
- **변혁적 유즈케이스 제시**: 내시경 비디오 분석을 중심으로 자동 주석 달기(Automated Annotation), 비디오 요약, 교차 모달 추론(Cross-modal Reasoning) 등 파운데이션 모델이 가져올 수 있는 실질적인 이점을 제시하였다.
- **데이터셋 및 벤치마크의 포괄적 평가**: 파운데이션 모델의 학습과 검증에 사용되는 대규모 멀티모달 데이터셋을 분석하고 최신 카탈로그를 제공하여, 데이터 가용성과 변동성 문제를 해결하기 위한 방향성을 제시하였다.

## 📎 Related Works

논문은 기존의 수술 AI 관련 리뷰 논문들과의 차별점을 명확히 제시한다. 기존 연구들(Rivas-Blanco et al., Rueckert et al. 등)은 주로 최소 침습 수술의 자동화나 특정 기구의 세그멘테이션과 같은 개별 작업에 집중하는 경향이 있었다. 반면, 본 논문은 다음과 같은 차별점을 가진다.

- **범위의 확장**: 단순한 세그멘테이션을 넘어 추적, 워크플로우 인식, 훈련 및 시뮬레이션까지 포괄하는 통합적 접근 방식을 취한다.
- **최신 아키텍처 반영**: U-Net 변형 모델 등에 집중했던 이전 연구들과 달리, ViTs, LVLMs, SAM과 같은 최신 파운데이션 모델의 적용 가능성을 집중적으로 다룬다.
- **실시간 의사결정 강조**: 정적인 분석보다는 실제 수술실의 동적이고 위험도가 높은 환경에서 실시간 의사결정을 지원하기 위한 기술적 요구사항을 강조한다.

## 🛠️ Methodology

본 논문은 리뷰 논문으로서 수술 장면 이해를 위한 AI 기술의 계층적 구조와 적용 방법론을 다음과 같이 분류하여 설명한다.

### 1. 수술 장면 이해의 주요 태스크

- **기구 및 객체 탐지 및 추적**: 수술 기구와 해부학적 구조를 식별하고 움직임을 추적하여 상황 인식 능력을 높인다.
- **워크플로우 인식**: 수술의 각 단계(Phase)를 인식하여 실시간 문서화, 기술 평가, 컨텍스트 기반 지원을 가능하게 한다.
- **훈련 및 시뮬레이션**: 외과의의 제스처와 행동을 이해하여 교육용 시뮬레이션을 생성하고 숙련도를 평가한다.

### 2. 적용된 주요 AI 모델 및 아키텍처

- **CNN 및 세그멘테이션 모델**: 기본적인 특징 추출을 위해 CNN을 사용하며, 특히 의료 영상에 특화된 $\text{U-Net}$(대칭적 인코더-디코더 구조)과 $\text{DeepLabv3}$(Atrous Convolution을 통해 수용 영역을 확장하여 다양한 스케일의 객체 분할) 등이 사용된다.
- **Vision Transformers (ViTs)**: 셀프 어텐션(Self-Attention) 메커니즘을 통해 이미지 내의 장거리 의존성을 학습함으로써, CNN보다 복잡한 공간적 관계를 더 잘 파악한다.
- **Foundation Models (FMs)**:
  - **SAM (Segment Anything Model)**: 제로샷(Zero-shot) 세그멘테이션이 가능하며, 본 논문에서는 이를 수술 환경에 맞게 최적화한 $\text{Surgical-DeSAM}$(모듈 분리), $\text{AdaptiveSAM}$(동적 파라미터 조정), $\text{CycleSAM}$(One-shot 학습), $\text{Surgical SAM 2}$(프레임 프루닝을 통한 실시간성 확보) 등을 분석한다.
  - **LLMs & VLMs**: $\text{GPT}$, $\text{BERT}$, $\text{LLaMA}$ 등을 통해 수술 보고서 자동 생성, 멀티모달 데이터 통합 분석, 시각적 질의응답(VQA) 등을 수행한다.
  - **DINO**: 자기지도학습(Self-supervised Learning)을 통해 라벨이 없는 데이터에서도 유의미한 특징을 추출하여 기구 세그멘테이션 및 깊이 추정에 활용한다.

### 3. 계산 복잡도 완화 전략

대규모 파운데이션 모델의 임상 적용을 위해 다음과 같은 효율화 기법을 설명한다.

- **Adapters & LoRA**: 모델 전체를 재학습시키지 않고 일부 파라미터만 조정하여 계산 비용을 낮춘다.
- **Prompt Tuning**: 입력 프롬프트를 최적화하여 모델의 기존 지식을 특정 작업에 유도한다.
- **Knowledge Distillation**: 거대 모델(Teacher)의 지식을 작은 모델(Student)로 전이시켜 추론 속도를 높인다.

## 📊 Results

본 논문은 정량적 실험 결과보다는 광범위한 문헌 조사와 데이터셋 분석 결과를 제시한다.

- **데이터셋 분석 (Table III)**: $\text{EndoVis}$, $\text{Cholec80}$, $\text{M2CAI16}$ 등 2015년부터 최근까지의 주요 수술 데이터셋을 정리하였다. 특히 데이터의 규모, 수술 종류, 라벨 유형(Bounding-box, Pixel-wise mask 등)을 상세히 분석하여 연구자들이 적절한 벤치마크를 선택할 수 있도록 돕는다.
- **모델 성능 비교 (Table IX)**: 다양한 수술 절차(담낭절제술, 대장항문 수술 등)에서 적용된 모델들의 성능을 $\text{mAP}$, $\text{Dice Coefficient}$, $\text{IoU}$ 등의 지표로 정리하였다. 예를 들어, $\text{SurgicalSAM}$은 최소한의 튜닝 파라미터로 SOTA 성능을 달성했음을 보여준다.
- **VQA 및 멀티모달 성능 (Table VIII)**: $\text{LV-GPT}$와 같은 모델들이 $\text{Surg-QA}$ 데이터셋 등에서 기존 프레임워크보다 우수한 성능을 보이며 수술 보조 시스템으로서의 가능성을 입증했음을 기술한다.

## 🧠 Insights & Discussion

### 강점 및 가능성

본 논문은 AI가 단순한 도구 탐지를 넘어 수술의 전 과정(Pre-operative $\rightarrow$ Intra-operative $\rightarrow$ Post-operative)을 아우르는 지능형 시스템으로 진화하고 있음을 보여준다. 특히 파운데이션 모델의 도입은 데이터 부족 문제(Data Scarcity)를 해결하고, 제로샷 능력을 통해 새로운 수술 환경에 빠르게 적응할 수 있는 가능성을 열어주었다.

### 한계 및 비판적 해석

- **기술적 장벽**: 혈액으로 인한 가려짐(Occlusion), specular reflection(경면 반사) 등 수술실 특유의 광학적 노이즈는 여전히 강력한 해결책이 필요하다.
- **인간 기술의 퇴화(Skill Atrophy)**: 저자는 AI 자동화가 심화될수록 수술 트레이니들이 단순한 '구경꾼'으로 전락하여, 응급 상황 대응 능력과 같은 핵심적인 운동 및 인지 능력을 상실할 수 있다는 점을 심각하게 경고한다.
- **블랙박스 문제**: 딥러닝 모델의 결정 과정이 불투명하여, 생명과 직결된 수술 환경에서 의료진이 AI의 제안을 전적으로 신뢰하기 어렵다는 설명 가능성(Explainability) 문제가 제기된다.

### 윤리적 및 규제적 논의

환자 프라이버시 보호(HIPAA, GDPR 준수)와 더불어, AI 모델이 편향된 데이터로 학습되었을 때 발생할 수 있는 의료 불평등 문제, 그리고 AI 보조 하에 발생한 의료 사고에 대한 법적 책임 소재(Accountability) 문제가 해결되어야 함을 강조한다.

## 📌 TL;DR

본 논문은 수술 장면 이해를 위해 CNN, ViTs를 넘어 **파운데이션 모델(SAM, GPT, DINO 등)을 통합하려는 최신 흐름을 분석한 종합 리뷰 보고서**이다. 특히 SAM의 다양한 변형 모델들이 수술 기구 세그멘테이션의 효율성을 어떻게 높였는지, 그리고 LMM/VLM이 수술 워크플로우 분석에 어떻게 기여하는지를 상세히 다룬다. 이 연구는 향후 AI가 외과의를 대체하는 것이 아니라, **인간의 숙련도를 보존하면서도 정밀도를 극대화하는 '인간 증강(Human Augmentation)' 도구**로 발전해야 함을 시사하며, 이는 실시간 수술 내비게이션 및 맞춤형 수술 교육 시스템 구축에 중요한 이정표가 될 것이다.
