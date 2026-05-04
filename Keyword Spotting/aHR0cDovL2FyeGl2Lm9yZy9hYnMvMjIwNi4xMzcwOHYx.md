# Personalized Keyword Spotting through Multi-task Learning

Seunghan Yang, Byeonggeun Kim, Inseop Chung, Simyung Chang (2022)

## 🧩 Problem to Solve

기존의 Keyword Spotting (KWS) 시스템인 Conventional KWS (C-KWS)는 정의된 키워드 자체를 검출하는 데 집중하며, 누가 그 단어를 말했는지에 대한 사용자 정보(User Information)를 고려하지 않는다. 이러한 사용자 불가지론적(User-agnostic) 특성으로 인해, 실제 환경에서는 다음과 같은 문제가 발생한다.

첫째, TV 방송, 온라인 회의, 또는 주변 사람들의 대화 속에서 타겟 키워드가 포함된 경우 이를 사용자의 호출로 오인하여 시스템이 불필요하게 활성화되는 False Alarm이 빈번하게 발생한다. 둘째, 이는 스마트 기기의 전력 소모를 불필요하게 증가시키는 원인이 된다.

본 논문의 목표는 사용자 정보를 활용하여 특정 사용자에게 최적화된 Personalized Keyword Spotting 시스템을 구축하는 것이며, 이를 위해 Target user Biased KWS (TB-KWS)와 Target user Only KWS (TO-KWS)라는 두 가지 새로운 개인화 태스크를 정의하고 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Keyword Spotting(KWS)과 Speaker Verification(SV)을 Multi-task Learning(MTL) 구조로 결합하여, 모델이 키워드의 특성과 화자의 특성을 동시에 학습하게 하는 것이다.

단순히 두 모델을 병렬로 사용하는 것이 아니라, 하위 계층의 Encoder를 공유하여 연산 효율성을 높이고, 상위 계층에서 각 태스크의 특성을 추출하는 Sub-network를 배치함으로써 화자 정보를 KWS 시스템에 효과적으로 주입한다. 또한, 학습된 표현(Representation)을 각 개인화 태스크(TB-KWS, TO-KWS)에 맞게 최적화하는 Task-specific Scoring Function인 Score Combination Module (SCM)과 Task Representation Module (TRM)을 제안하여 개인화 성능을 극대화하였다.

## 📎 Related Works

기존의 Query-by-example KWS 연구들은 사용자가 직접 키워드를 등록할 수 있게 하여 새로운 키워드에 적응하는 방식이었으나, 화자의 정체성(User Identity)을 명시적으로 고려하지는 않았다. 또한, KWS와 SV를 결합한 기존의 MTL 연구들이 존재하지만, 이들은 주로 각 개별 태스크의 성능을 높이는 데 집중했을 뿐, 화자 정보를 활용해 KWS의 개인화를 달성하려는 목적과는 차이가 있다.

본 연구는 화자 정보를 단순히 보조적인 수단으로 쓰는 것이 아니라, 타겟 사용자와 비타겟 사용자를 구분하는 기준점으로 삼아 False Alarm Rate를 획기적으로 낮추는 데 차별점이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
제안된 PK-MTL(Personalized Keyword Spotting through Multi-task Learning)은 크게 **Multi-task Learning** 단계와 **Task-adaptation** 단계의 두 가지 과정으로 구성된다.

### 2. Multi-task Learning (MTL)
- **Shared Encoder**: 입력 오디오 특징(Spectrogram 등)을 처리하는 하위 계층 $f_\phi(\cdot)$를 공유하여 메모리와 연산 비용을 절감한다.
- **Sub-networks**: 공유 엔코더 이후에 KWS를 위한 $f_k^\phi(\cdot)$와 SV를 위한 $f_s^\phi(\cdot)$를 각각 두어, 키워드 특성과 화자 특성을 분리하여 학습한다.
- **Cosine Classifier**: 각 특성 벡터 $z$와 학습 가능한 가중치 $W$ 사이의 코사인 유사도를 기반으로 분류를 수행한다.
  $$g(z) = \text{softmax}(w \cdot \text{sim}(z, W) + b)$$
- **Loss Function**: KWS 손실 $L_k$와 SV 손실 $L_s$를 결합한 전체 손실 함수를 최소화한다.
  $$L_{mtl} = L_k + \lambda L_s$$
  여기서 $\lambda$는 화자 정보의 중요도를 조절하는 하이퍼파라미터이다.

### 3. Task-adaptation
학습된 표현을 개인화 태스크에 적용하기 위해 두 가지 scoring 방식을 제안한다.

- **Score Combination Module (SCM)**: 
  KWS 점수 $\psi_k$와 SV 점수 $\psi_s$를 단순 선형 결합하여 최종 점수를 산출하는 방식이다.
  $$\text{Score} = \alpha \cdot \psi_k + (1-\alpha) \cdot \psi_s$$
  이때 $\alpha$는 검증 셋에서 Target FAR(False Alarm Rate)을 기준으로 최적화하여 결정한다.

- **Task Representation Module (TRM)**: 
  단순 결합의 한계를 극복하기 위해, KWS 임베딩 $z_k$와 SV 임베딩 $z_s$를 입력으로 받아 태스크 전용 임베딩을 생성하는 학습 가능한 신경망(Attention 기반)이다. TRM은 **Angular Prototypical Loss**를 사용하여 긍정 샘플은 가깝게, 부정 샘플은 멀게 배치하도록 학습된다.
  $$\psi_{tb} = \text{sim}(\text{TRM}_{tb}(z_{k,i}, z_{s,i}), \text{TRM}_{tb}(p_{k,j}, p_{s,j}))$$
  이를 통해 TB-KWS와 TO-KWS 각각의 목적에 맞는 정교한 판별 경계를 학습한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Commands v1 (기본 학습), WSJ-SI200 및 Librispeech (실제 환경의 부정 샘플로 활용).
- **백본 네트워크**: BC-ResNet, Res15, DS-ResNet.
- **평가 지표**: FAR, FRR, EER 및 Top-1 Accuracy.

### 2. 주요 결과
- **개인화 성능**: Table 1에 따르면, PK-MTL은 기존 Vanilla 모델 대비 TB-KWS와 TO-KWS에서 EER 및 FRR을 대폭 낮추었다. 특히 TRM을 적용했을 때 가장 우수한 성능을 보였다.
- **실제 시나리오(Realistic Scenario)**: TV 방송이나 일반 대화 데이터(WSJ, Librispeech)를 부정 샘플로 사용했을 때, Vanilla 모델은 타겟 키워드가 포함되어 있으면 화자와 상관없이 높은 FAR를 보였다. 반면 PK-MTL(특히 TO-KWS)은 화자 정보를 활용해 이를 효과적으로 거부함으로써 FAR를 극적으로 낮추었다(Table 2 참조).
- **효율성**: MTL 구조를 통해 파라미터 수와 연산량의 증가를 최소화하면서도 개인화 기능을 추가하는 데 성공하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 KWS와 SV라는 서로 다른 성격의 태스크를 MTL로 묶어, 화자 정보를 KWS 시스템의 필터로 활용했다는 점이다. 특히 TRM을 통해 단순히 점수를 합산하는 수준을 넘어, '누가 어떤 단어를 말했는가'에 대한 태스크 특화 표현을 학습하게 한 점이 성능 향상의 핵심이다.

분석 결과, 일반적인 KWS 시스템이 겪는 고질적인 문제인 '주변 소음 속 키워드 오검출' 문제를 화자 식별 능력을 통해 해결할 수 있음을 입증하였다. 다만, 본 논문에서는 타겟 사용자의 등록 음성(Enroll utterance)이 존재한다는 가정을 전제로 하고 있으며, 등록 데이터의 품질이나 양에 따른 성능 변화에 대한 논의는 명시적으로 다뤄지지 않았다.

## 📌 TL;DR

본 논문은 화자 정보가 배제된 기존 KWS의 높은 오검출률 문제를 해결하기 위해, KWS와 SV를 결합한 Multi-task Learning 기반의 개인화 프레임워크(PK-MTL)를 제안한다. 제안된 시스템은 타겟 사용자 중심의 TB-KWS와 타겟 사용자 전용인 TO-KWS 태스크를 효과적으로 수행하며, 특히 실제 환경의 배경 소음이나 타인 음성으로 인한 False Alarm을 획기적으로 줄일 수 있음을 증명하였다. 이는 향후 스마트 기기의 전력 효율 향상 및 사용자 경험 개선에 기여할 가능성이 높다.