# Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights

Yuan Zhong, Ruinan Jin, Xiaoxiao Li, and Qi Dou (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석을 위한 전용 Medical Vision-Language Models (VLMs)의 개발과 사용에 따르는 막대한 계산 자원 및 데이터 큐레이션 비용 문제를 다룬다. 최근 MedCLIP, LLaVA-Med와 같은 일반ist 의료 VLM들이 등장하며 다양한 의료 영상 작업에서 성능을 보이고 있으나, 이러한 대규모 사전 학습(pre-training)은 경제적, 기술적 진입 장벽이 매우 높다.

이에 본 연구는 "효율적으로 파인튜닝된 일반 목적의 Common VLM(예: CLIP, LLaVA)이 특정 의료 영상 작업에서 일반ist 의료 VLM과 대등하거나 더 나은 성능을 낼 수 있는가?"라는 핵심 질문을 던진다. 특히, 도메인 특화 사전 학습이 반드시 필수적인지에 대해 의문을 제기하며, 비용 대비 성능의 최적 지점을 찾는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 직관은 일반 목적의 VLM이 이미 방대한 데이터로부터 풍부한 시각적-언어적 표현(representation)을 학습했으므로, 이를 가벼운 파인튜닝(lightweight fine-tuning)을 통해 의료 도메인에 적응시키는 것이 대규모 의료 전용 모델을 처음부터 학습시키는 것보다 훨씬 효율적이고 실용적일 수 있다는 점이다.

주요 기여 사항은 다음과 같다.

- 6개의 진단(Diagnosis) 데이터셋과 4개의 VQA 데이터셋을 활용하여, 총 138가지 시나리오에 대해 Common VLM과 Medical VLM의 성능을 체계적으로 비교 분석하였다.
- In-Domain(ID) 설정과 Out-of-Domain(OOD) 설정 모두에서 모델의 성능 갭과 일반화 능력을 평가하여, 사전 학습된 의료 지식의 실제 효용성을 검증하였다.
- LoRA(Low-Rank Adaptation)와 같은 효율적인 파인튜닝 기법이 Common VLM의 의료 도메인 적응력을 극대화하여, 고비용의 의료 전용 모델을 대체할 수 있는 가능성을 제시하였다.

## 📎 Related Works

본 논문은 두 가지 주요 VLM 패밀리를 다룬다.

1. **CLIP Family**: 이미지-텍스트 쌍의 대조 학습(contrastive learning)을 통해 정렬된 모델들이다. 일반 모델인 CLIP 외에 의료 전용인 MedCLIP(방사선 이미지), PubMedCLIP, PLIP(병리 이미지), BioMedCLIP(다양한 모달리티) 등이 존재한다.
2. **LLaVA Family**: CLIP의 시각 인코더와 거대 언어 모델(LLM)을 결합한 구조이다. 일반 모델인 LLaVA와 의료 데이터로 추가 정렬 및 인스트럭션 튜닝을 거친 LLAva-Med가 대표적이다.

기존 연구들은 주로 의료 전용 사전 학습이 의료 작업의 성능을 높인다고 주장해 왔으나, 본 논문은 이러한 전용 모델들이 실제로 보지 못한 새로운 모달리티(OOD)에 대해 얼마나 일반화 능력을 갖추었는지에 대한 평가가 부족했다는 점을 지적하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 작업 정의

연구는 크게 두 가지 작업으로 나누어 진행된다.

- **Disease Diagnosis (진단)**: CLIP 스타일 모델을 사용하여 이미지와 질병 프롬프트 간의 유사도를 계산하여 진단한다.
- **Medical VQA (시각적 질의응답)**: LLaVA 스타일 모델을 사용하여 의료 이미지에 대한 질문에 텍스트 답변을 생성한다.

### 파인튜닝 전략 (Fine-tuning Strategies)

Common VLM을 의료 도메인에 적응시키기 위해 다음 두 가지 효율적인 기법을 적용한다.

1. **Sparse FT (Sparse Fine-Tuning)**:
   - 진단 작업의 경우, 이미지 인코더에 Linear Probe(LP)를 적용하여 학습한다.
   - VQA 작업의 경우, 다른 부분은 고정(freeze)하고 MLP 브릿지 계층만 튜닝한다.
2. **PEFT (Parameter-Efficient Fine-Tuning)**:
   - LoRA(Low-Rank Adaptation)를 적용한다. LoRA는 가중치 행렬 $W$를 두 개의 저차원 행렬의 곱으로 분해하여 학습 파라미터를 획기적으로 줄이는 방식이다.
   - 진단 작업에서는 이미지 인코더에 적용하며, VQA 작업에서는 이미지 인코더, 텍스트 디코더, 또는 MLP 계층의 조합에 적용한다.

### 학습 및 추론 절차

- **Off-the-shelf**: 추가 학습 없이 사전 학습된 가중치를 그대로 사용하여 Zero-shot 성능을 측정한다.
- **Fine-tuning**: 타겟 의료 데이터셋을 사용하여 위에서 정의한 Sparse FT 또는 LoRA를 적용한다.
- **OOD Evaluation**: 모델이 사전 학습 단계에서 한 번도 보지 못한 의료 모달리티(unseen modality) 데이터셋으로 학습 및 테스트를 진행하여 적응력을 측정한다.

### 평가 지표

- **Diagnosis**: 클래스 불균형 문제를 고려하여 AUC(Area Under the Curve)를 단일 지표로 사용한다.
- **VQA**:
  - Closed-ended (Yes/No): Accuracy를 측정한다.
  - Open-ended: 텍스트의 단순 일치도가 아닌 의미적 정확성을 평가하기 위해 GPT-4를 활용하여 1~100점 척도로 점수를 매긴다.

## 📊 Results

### RQ1: Off-the-shelf 성능 갭 (In-Domain)

- **결과**: ID 설정(사전 학습 시 본 모달리티)에서는 Medical VLM이 Common VLM보다 전반적으로 우수한 성능을 보였다. 특히 진단 작업의 Camelyon17 데이터셋에서는 Medical CLIP 계열이 일반 CLIP보다 20포인트 이상 높게 나타났다.
- **의미**: 도메인 특화 사전 학습이 초기 벤치마크 성능 향상에 분명한 이점이 있음을 확인하였다.

### RQ2: 효율적 파인튜닝의 효과

- **결과**: Common VLM에 가벼운 파인튜닝(LP, LoRA)을 적용한 결과, 모든 케이스에서 off-the-shelf Medical VLM의 성능을 상회하였다.
- **의미**: 일반 VLM이 이미 보유한 범용적 지식이 적은 양의 의료 데이터로도 충분히 특화될 수 있음을 시사한다.

### RQ3: OOD(Out-of-Domain) 일반화 능력

- **결과**:
  - 사전 학습만으로는 Medical VLM이 OOD 데이터에서 Common VLM보다 특별히 우월하지 않았으며, 오히려 일반 CLIP 모델이 일부 OOD 진단 작업에서 더 높은 성능을 보였다.
  - 파인튜닝을 적용했을 때, Common VLM은 새로운 모달리티에 매우 빠르게 적응하여 Medical VLM과 대등하거나 더 나은 성능을 기록하였다. 특히 VQA 작업에서 MLP 브릿지만 튜닝한 Common LLaVA가 매우 효율적인 OOD 적응력을 보였다.

## 🧠 Insights & Discussion

### 강점 및 시사점

본 연구는 의료 AI 개발의 패러다임을 '거대 전용 모델 구축'에서 '효율적인 범용 모델 적응'으로 전환할 수 있는 근거를 제시한다. 특히 LoRA와 같은 PEFT 기법을 사용하면 매우 적은 비용으로도 전용 모델 수준의 성능을 확보할 수 있다는 점이 입증되었다.

### 한계 및 비판적 해석

- **데이터 의존성**: Common VLM이 파인튜닝 후 Medical VLM을 이긴 것은, 결국 '타겟 데이터셋'을 사용하여 학습했기 때문이다. 이는 사전 학습의 힘이라기보다 파인튜닝의 효과일 수 있다.
- **비용 계산의 단순함**: 논문에서 제시한 비용 분석은 단순 파라미터 수와 샘플 수의 곱으로 계산되었으며, 실제 학습 시의 수렴 속도나 하이퍼파라미터 튜닝 비용은 완전히 반영되지 않았을 가능성이 있다.
- **미해결 질문**: 어느 정도의 데이터 양이 확보되었을 때 Common VLM의 효율성이 Medical VLM의 사전 학습 이점을 완전히 압도하는지에 대한 임계점(threshold) 분석이 추가로 필요하다.

## 📌 TL;DR

본 논문은 **"비싼 의료 전용 VLM을 만드는 대신, 일반 VLM을 가볍게 파인튜닝하는 것이 더 효율적일 수 있다"**는 것을 실험적으로 증명하였다. 특히 LoRA 기반의 적응 방식은 ID 및 OOD 작업 모두에서 매우 강력한 성능을 보였으며, 이는 향후 의료 AI 연구가 거대 모델의 사전 학습보다는 효율적인 도메인 적응(Domain Adaptation) 전략에 집중해야 함을 시사한다.
