# Distilling HuBERT with LSTMs via Decoupled Knowledge Distillation

Danilo de Oliveira, Timo Gerkmann (2023)

## 🧩 Problem to Solve

본 논문은 HuBERT와 같은 자기지도학습(Self-Supervised Learning, SSL) 모델이 매우 강력한 성능을 제공하지만, 모델의 크기가 너무 커서 메모리 소비가 심하고 실제 환경에 적용하기 어렵다는 문제를 해결하고자 한다. 

기존의 HuBERT 압축 연구들(예: DistilHuBERT, FitHuBERT)은 주로 교사 모델의 내부 중간 레이어 표현(Internal features/hints)을 학생 모델이 모방하도록 하는 feature-based distillation 방식을 사용했다. 그러나 이러한 방식은 학생 모델이 교사 모델과 유사한 아키텍처(주로 Transformer)를 가져야 한다는 제약이 있으며, 단순히 레이어를 줄이거나 차원을 축소하는 수준에 머문다는 한계가 있다. 따라서 본 연구의 목표는 아키텍처의 제약에서 벗어나 더 효율적인 구조를 탐색하고, 이를 통해 파라미터 수를 획기적으로 줄이면서도 성능을 유지하거나 향상시킨 압축 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 HuBERT의 사전 학습(Pre-training) 과정이 본질적으로 클러스터 예측(Cluster prediction)이라는 분류(Classification) 문제라는 점에 주목한 것이다. 

연구진은 내부 특징을 모방하는 대신, 모델의 최종 출력값인 로짓(Logits)을 기반으로 하는 Knowledge Distillation (KD) 및 Decoupled Knowledge Distillation (DKD)을 적용함으로써 학생 모델의 아키텍처를 자유롭게 선택할 수 있게 하였다. 이를 통해 Transformer 기반의 학생 모델뿐만 아니라, 시퀀스 모델링에 효율적인 LSTM(Long Short-Term Memory) 기반의 모델로 HuBERT의 지식을 전이시켰으며, 결과적으로 더 적은 파라미터로도 자동 음성 인식(ASR) 등 특정 작업에서 더 나은 성능을 달성하였다.

## 📎 Related Works

기존의 지식 증류(Knowledge Distillation) 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 Hinton 등이 제안한 KD로, 교사 모델이 생성한 소프트 타겟 확률 분포를 학생 모델이 학습하도록 하는 방식이다. 둘째는 최근 제안된 DKD(Decoupled Knowledge Distillation)로, KD 손실 함수를 타겟 클래스 지식(TCKD)과 비타겟 클래스 지식(NCKD)으로 분리하여 교사 모델의 확신도(Confidence)에 따른 성능 저하 문제를 해결한 방식이다.

음성 SSL 모델의 압축을 위한 기존 연구인 DistilHuBERT와 FitHuBERT는 앞서 언급한 feature-based distillation을 사용한다. 이들은 교사 모델의 특정 레이어 출력을 힌트로 사용하여 학생 모델을 학습시키므로, 학생 모델 역시 Transformer 인코더 구조를 유지해야만 한다. 본 논문은 이러한 특징 기반 방식과 달리 로짓 기반 방식을 채택함으로써 LSTM과 같은 전혀 다른 아키텍처를 적용할 수 있다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 지식 증류 방법론 (KD 및 DKD)

기본적인 KD는 교사 모델의 로짓 $l_c$를 온도 파라미터 $\tau$를 이용해 소프트맥스 확률 분포 $p_c$로 변환하여 사용한다.

$$p_c = \frac{\exp(l_c/\tau)}{\sum_{c'=1}^{C} \exp(l_{c'}/\tau)}$$

이때 KD 손실 함수는 다음과 같이 Kullback-Leibler(KL) 발산으로 정의된다.

$$L_{KD} = KL(p^T || p^S) \cdot \tau^2$$

여기서 $p^T$와 $p^S$는 각각 교사 모델과 학생 모델의 소프트맥스 확률 점수이다.

본 논문에서 적용한 DKD는 위 $L_{KD}$를 TCKD와 NCKD로 분리하여 각각에 하이퍼파라미터 $\alpha$와 $\beta$를 할당한다.

$$L_{DKD} = \alpha \cdot TCKD + \beta \cdot NCKD$$

TCKD는 타겟 클래스에 대한 지식을, NCKD는 타겟 클래스를 제외한 나머지 클래스들 간의 관계에 대한 지식을 전달한다. 특히 NCKD는 교사 모델의 확신도가 높을 때 억제되는 경향이 있는데, DKD는 이를 분리하여 가중치를 조절함으로써 더 효과적인 지식 전이를 가능하게 한다.

### 2. 제안하는 시스템 구조

- **교사 모델**: HuBERT-LARGE 모델을 사용하며, 500개의 클러스터에 대한 로짓을 생성한다.
- **학생 모델 (LSTM-based)**: 4개의 Bi-directional LSTM 레이어를 사용하며, 각 레이어의 hidden size는 384이다. 이는 Transformer 기반의 DistilHuBERT와 유사한 메모리 점유율을 유지하면서도 효율적인 연산을 가능하게 한다.
- **입력 및 출력**: 교사 모델과 동일한 Convolutional feature extractor를 사용한다. 학습 시에는 단순한 분류 프로젝션(Classification projection) 레이어를 추가하여 로짓을 생성하며, 다운스트림 태스크에 적용할 때는 이 프로젝션 레이어를 제거한다.

### 3. 학습 절차

- **데이터셋**: LibriSpeech 학습 데이터 960시간을 사용한다.
- **학습 설정**: 200k steps 동안 학습하며, 학습률은 $2 \cdot 10^{-4}$, 배치 사이즈는 32를 적용한다. 초기 14k update 동안 linear warm-up을 수행한 후 linear decay를 적용한다.
- **손실 함수**: 클래스 할당값(Cluster assignments)을 정답으로 하는 Cross-Entropy 손실과 $L_{KD}$ 또는 $L_{DKD}$의 선형 결합으로 학습한다. $\tau=1, \alpha=1$로 설정하고, DKD의 경우 $\beta \in \{1, 4\}$를 탐색한다.

## 📊 Results

### 1. 실험 설정
- **벤치마크**: SUPERB (10가지 태스크를 Content, Speaker, Semantics, Paralinguistics의 4개 카테고리로 분류)
- **비교 대상**: HuBERT-BASE/LARGE, LightHuBERT, DistilHuBERT, FitHuBERT 및 제안하는 KD/DKD-LSTM 모델.

### 2. 정량적 결과
- **음성 인식 성능**: LSTM 기반의 DKD 모델은 Phoneme Recognition(PR)과 Automatic Speech Recognition(ASR) 작업에서 DistilHuBERT나 FitHuBERT보다 유의미하게 높은 성능을 보였다. 특히 $\beta=4$일 때 PR, ASV, QbE 작업에서 성능 향상이 두드러졌다.
- **파라미터 효율성**: LSTM 기반 모델은 DistilHuBERT보다 더 적은 수의 파라미터를 사용하면서도 ASR 성능은 오히려 향상시켰다.
- **기타 작업**: Slot Filling(SF)과 Intent Classification(IC) 역시 LSTM 구조의 이점을 얻었으나, Query-by-Example(QbE)과 Automatic Speaker Verification(ASV)에서는 Transformer 기반 KD 모델 대비 약간의 성능 하락이 있었다.

### 3. 리소스 분석 (Model Profiling)
- **메모리 및 연산량**: LSTM 모델은 Transformer 기반 모델보다 연산량(GMACs)이 적으며, 특히 입력 시퀀스 길이에 따라 메모리 할당량이 선형적으로 증가한다(Transformer는 제곱으로 증가).
- **실행 시간**: 레이어 수가 적은 DistilHuBERT가 약간 더 빠르지만, 전반적으로 두 모델 모두 교사 모델 대비 획기적으로 빠른 추론 속도를 보여 실용적임을 입증하였다.

## 🧠 Insights & Discussion

본 연구 결과, 로짓 기반의 KD/DKD를 통해 LSTM 모델을 구축했을 때 ASR 및 PR 성능이 크게 향상된 점은 매우 중요한 통찰을 제공한다. 이는 HuBERT의 사전 학습 목적 자체가 마스크된 유닛을 예측하는 분류 작업이며, 이는 본질적으로 언어적 내용(Linguistic content)을 식별하는 것과 맞닿아 있기 때문이다. 여기에 LSTM의 순차적 처리(Sequential processing) 특성이 결합되어 음성 인식 작업에서 시너지를 낸 것으로 해석된다.

다만, Speaker Identification(SID)과 같은 작업에서는 성능이 저하되는 경향을 보였는데, 이는 로짓 기반 증류가 모델의 깊은 층에 있는 언어적 정보에 집중하는 반면, 화자 식별에 필요한 전역적 음향 특징(Global acoustic features)은 초기 레이어에 더 많이 분포하기 때문으로 추측된다. 

결론적으로, 본 논문은 지식 증류의 대상을 내부 특징에서 로짓으로 변경함으로써 학생 모델의 아키텍처 선택지를 넓혔으며, 이를 통해 특정 도메인(ASR)에서 더 효율적이고 강력한 경량 모델을 구축할 수 있음을 보여주었다.

## 📌 TL;DR

본 논문은 HuBERT 모델을 압축하기 위해 기존의 특징 기반 증류 대신 **로짓 기반의 Knowledge Distillation(KD) 및 Decoupled Knowledge Distillation(DKD)**를 제안하였다. 이를 통해 학생 모델로 **LSTM 아키텍처**를 채택할 수 있었으며, 결과적으로 **파라미터 수를 줄이면서도 자동 음성 인식(ASR) 및 음소 인식(PR) 성능을 기존의 Transformer 기반 경량 모델(DistilHuBERT 등)보다 향상**시켰다. 이 연구는 SSL 모델의 압축 시 아키텍처의 유연성을 확보하는 것이 성능 최적화에 중요한 역할을 할 수 있음을 시사한다.