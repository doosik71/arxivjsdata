# Biomedical Foundation Model: A Survey

Xiangrui Liu, Yuanyuan Zhang, Yingzhou Lu, et al. (2025)

## 🧩 Problem to Solve

본 논문은 생의학(Biomedical) 분야에서 기존의 작업 특정적(Task-specific) AI 모델들이 가진 한계를 극복하기 위해, 대규모 데이터로 사전 학습된 **Foundation Model(기반 모델)**의 적용 현황과 가능성을 분석하는 것을 목표로 한다.

전통적인 생의학 AI 모델들은 지도 학습(Supervised Learning)에 의존해 왔으나, 의료 데이터의 특성상 정교한 레이블링(Labeling) 비용이 매우 높고 데이터 수집에 제약이 많아 모델의 범용성과 확장성이 떨어진다는 문제가 있었다. 이에 본 논문은 방대한 양의 무라벨(Unlabeled) 데이터를 통해 일반적인 표현(Representation)을 학습하고, 이를 다양한 하위 작업(Downstream tasks)에 적응시키는 Foundation Model의 패러다임이 생의학 연구와 의료 실무에 어떻게 기여할 수 있는지를 체계적으로 정리하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 생의학 분야의 Foundation Model 연구를 다섯 가지 주요 도메인으로 분류하고, 각 도메인에서의 모델 설계 직관과 적용 사례를 종합적으로 분석한 점이다.

1. **Computational Biology**: DNA, RNA, 단백질 서열을 일종의 '자연어'로 간주하여 언어 모델(Language Model) 아키텍처를 적용하는 접근 방식을 제시한다.
2. **Drug Discovery and Development**: 분자 구조의 표현 학습(Molecular Representation Learning)을 통해 신약 후보 물질 발굴 및 약물 재창출(Drug Repurposing) 과정을 가속화하는 방법론을 분석한다.
3. **Clinical Informatics**: 전자의무기록(EHR)과 같은 정형/비정형 임상 데이터를 임베딩하여 환자 상태를 표현하고 치료 효과를 예측하는 모델들을 다룬다.
4. **Medical Imaging**: 병리학(Pathology), 방사선학(Radiology), 망막 이미지 등 다양한 모달리티에서 self-supervised learning을 통한 일반화된 시각 특징 추출 방법을 논의한다.
5. **Public Health**: 소셜 미디어 데이터 및 시계열 데이터를 활용한 질병 감시 및 유행 예측 모델의 가능성을 탐색한다.

## 📎 Related Works

기존의 생의학 AI 접근 방식은 특정 질병의 진단이나 특정 단백질 구조 예측과 같은 **좁은 범위의 작업(Narrow tasks)**에 최적화된 모델을 구축하는 것이었다. 이러한 방식은 다음과 같은 한계가 있다.

- **데이터 효율성 저하**: 각 작업마다 대량의 레이블링된 데이터가 필요하며, 데이터가 부족한 희귀 질환의 경우 모델 구축이 어렵다.
- **지식 전이의 부재**: 한 작업에서 학습한 지식을 다른 유사 작업으로 이전(Transfer)하는 능력이 부족하다.

반면, 본 논문에서 다루는 Foundation Model은 **Scaling Law(규모의 법칙)**에 근거하여 모델 크기, 데이터셋 크기, 계산 자원을 늘림으로써 일반화 성능을 비약적으로 향상시킨다. 특히 GPT, Claude와 같은 범용 LLM/VLM의 성공을 생의학 도메인으로 확장하여, 적은 양의 레이블 데이터만으로도 효율적인 미세 조정(Fine-tuning)을 통해 뛰어난 성능을 낼 수 있음을 강조하며 기존 방식과 차별화한다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 특정 단일 알고리즘을 제안하지 않으나, 생의학 Foundation Model의 전반적인 파이프라인과 도메인별 방법론을 다음과 같이 설명한다.

### 1. 전체 파이프라인

전체적인 시스템 구조는 **[대규모 무라벨 데이터 $\rightarrow$ 자기지도학습(Self-supervised Learning) 기반 사전 학습 $\rightarrow$ 하위 작업별 적응(Adaptation/Fine-tuning)]**의 단계로 구성된다.

### 2. 도메인별 핵심 방법론

#### A. Computational Biology (계산 생물학)

- **DNA/RNA 언어 모델**: 염기서열(Nucleotide sequence)을 토큰화하여 Transformer 기반의 모델(예: DNABERT, HyenaDNA)로 학습한다. 특히 $\text{O}(n^2)$의 복잡도를 가진 Attention을 $\text{O}(n \log n)$이나 선형 복잡도로 줄인 Hyena operator 등을 사용하여 매우 긴 서열을 처리한다.
- **단백질 구조 및 설계**: AlphaFold2와 같이 단백질 서열을 입력으로 하여 3D 좌표를 예측하거나, RFDiffusion과 같은 Diffusion Model을 사용하여 특정 기능을 가진 단백질 구조를 역으로 설계한다.

#### B. Drug Discovery (신약 개발)

- **분자 표현 학습**: 분자 구조를 SMILES 문자열이나 그래프(Graph) 형태로 입력받아 임베딩 공간에 매핑한다.
- **ADMET 예측**: 약물의 흡수(Absorption), 분포(Distribution), 대사(Metabolism), 배설(Excretion), 독성(Toxicity)을 예측하기 위해 ChemBERTa-2와 같은 모델이 사용되며, 대규모 SMILES 데이터셋으로 사전 학습 후 미세 조정된다.

#### C. Clinical Informatics (임상 정보학)

- **환자 표현(Patient Representation)**: 정형 EHR 데이터를 BERT 구조에 적용한 BEHRT 등이 사용된다. 환자의 진료 기록을 시퀀스로 보고, 마스킹된 토큰을 예측하는 방식으로 환자의 건강 상태를 수치화된 벡터로 변환한다.
- **치료 효과 추정(TEE)**: 관찰 데이터(Observational data)로부터 특정 치료의 효과를 추정하기 위해 Transformer 기반의 TransTEE 등이 활용된다.

#### D. Medical Imaging (의료 영상)

- **Vision-Language Alignment**: CLIP 아키텍처를 활용하여 의료 영상과 그에 대응하는 판독문(Text report)을 동일한 잠재 공간(Latent space)에 매핑하는 방식(예: MedCLIP, BiomedCLIP)이 핵심이다.
- **Segmentation**: Segment Anything Model(SAM)을 의료 영상에 맞게 어댑터(Adapter)를 추가하여 미세 조정하는 방식(예: MedSAM)이 사용된다.

## 📊 Results

본 논문은 개별 연구들의 결과를 종합하여 Foundation Model의 효용성을 입증한다.

- **정량적 성과**:
  - **Med-PaLM 2**는 미국 의사 면허 시험(USMLE)을 통과할 정도로 높은 수준의 의학적 질문 답변 능력을 보여주었다.
  - **AlphaFold2**는 단백질 구조 예측 분야에서 실험적 수준에 근접한 정확도를 달성하여 구조 생물학의 패러다임을 바꾸었다.
  - **RETFound**는 대규모 망막 이미지 데이터셋으로 사전 학습되어, 다양한 망막 질환 검출 작업에서 일반화된 성능을 입증하였다.

- **정성적 성과**:
  - 기존의 작업 특정적 모델보다 적은 양의 데이터로도 새로운 하위 작업에 빠르게 적응할 수 있는 **Few-shot learning** 능력이 확인되었다.
  - 다중 모달리티(Multi-modality) 모델(예: BiomedGPT)을 통해 영상-텍스트 간 상호 변환 및 복합 질의응답이 가능해졌다.

## 🧠 Insights & Discussion

### 강점 및 기회

본 논문은 생의학 데이터의 방대함과 복잡성을 Foundation Model의 확장성(Scalability)으로 해결할 수 있음을 보여준다. 특히 서로 다른 모달리티(유전체 $\rightarrow$ 단백질 $\rightarrow$ 영상 $\rightarrow$ 임상 기록)를 통합하는 Generalist AI로의 발전 가능성이 매우 크다.

### 한계 및 미해결 과제

- **데이터 프라이버시**: 의료 데이터의 민감성으로 인해 대규모 공개 데이터셋 구축이 어려우며, 이는 모델 학습의 병목 현상이 된다.
- **해석 가능성(Interpretability)**: 의료 분야에서는 결과의 근거가 중요하지만, 거대 모델의 내부 작동 원리를 이해하는 '블랙박스' 문제는 여전히 해결해야 할 과제이다.
- **데이터 불균형**: 희귀 질환 데이터의 부족으로 인해 모델이 다수 클래스에 편향될 가능성이 있다.

### 비판적 해석

논문은 광범위한 모델들을 나열하고 있으나, 각 모델 간의 직접적인 벤치마크 비교보다는 개별 연구의 성과를 요약하는 데 집중하고 있다. 향후 연구에서는 생의학 전반을 아우르는 통일된 평가 지표(Unified Benchmark)가 도입되어 모델 간의 성능을 객관적으로 비교할 필요가 있다.

## 📌 TL;DR

본 논문은 생의학 분야의 **Foundation Model**을 계산 생물학, 신약 개발, 임상 정보학, 의료 영상, 공중 보건의 다섯 가지 영역으로 나누어 분석한 종합 서베이 보고서이다. 핵심은 대규모 무라벨 데이터를 이용한 **사전 학습(Pre-training)**과 **범용 표현 학습(Representation Learning)**을 통해, 데이터 부족 문제를 해결하고 의료 AI의 일반화 성능을 극대화하는 것이다. 이 연구는 향후 정밀 의료(Precision Medicine) 및 범용 의료 AI 시스템 구축을 위한 중요한 이정표를 제시한다.
