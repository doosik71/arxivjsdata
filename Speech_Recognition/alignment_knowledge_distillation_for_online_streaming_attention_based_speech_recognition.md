# Alignment Knowledge Distillation for Online Streaming Attention-based Speech Recognition

Hirofumi Inaguma and Tatsuya Kawahara (2021)

## 🧩 Problem to Solve

본 논문은 온라인 스트리밍 환경에서 동작하는 Attention-based Encoder-Decoder (AED) 자동 음성 인식(ASR) 시스템의 효율적인 학습 방법을 다룬다. AED 모델은 오프라인 시나리오에서는 매우 뛰어난 성능을 보이지만, 추론 시 전체 입력 시퀀스가 필요하다는 특성 때문에 스트리밍 적용이 어렵다. 이를 해결하기 위해 Monotonic Chunkwise Attention (MoChA)와 같은 기법이 제안되었으나, 여전히 다음과 같은 두 가지 핵심적인 문제가 존재한다.

첫째, **정렬 확률의 소멸(Vanishing alignment probabilities)** 문제이다. MoChA의 정렬 계산 과정은 긴 음성 발화에 대해 강건하지 않으며, 학습 과정에서 정렬 확률이 빠르게 감쇠하여 조기 종료(premature endpointing)가 발생하고, 이는 결과적으로 삭제 오류(deletion errors)의 증가로 이어진다.

둘째, **토큰 생성 지연(Delayed token generation)** 문제이다. E2E 모델은 시퀀스 수준의 최적화 목표를 가지므로, 디코더가 가능한 한 많은 미래의 관측값을 사용하려는 경향이 있다. 이로 인해 실제 음성 경계보다 토큰 생성이 수 프레임 뒤처지게 되어, 사용자가 체감하는 지연 시간(perceived latency)이 증가하게 된다.

따라서 본 연구의 목표는 외부의 정렬 정보 없이 순수하게 end-to-end 방식으로 MoChA 모델의 정렬 학습을 강화하여, 인식 정확도를 높이고 토큰 생성 지연을 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **CTC Synchronous Training (CTC-ST)**이다. 이는 CTC(Connectionist Temporal Classification) 모델이 학습하는 정렬(alignment) 지식을 MoChA 모델로 전이하는 일종의 **자기 증류(self-distillation)** 방식이다.

핵심 직관은 CTC 모델이 프레임 단위의 조건부 독립성을 가정하고 Forward-Backward 알고리즘으로 최적화되기 때문에, MoChA보다 훨씬 더 정확하고 날카로운(peaky) 정렬 경계를 생성한다는 점이다. 따라서 CTC의 정렬 피크(peak) 위치를 참조점으로 삼아 MoChA의 토큰 경계 위치를 동기화함으로써, MoChA가 최적의 단조 정렬(monotonic alignment)을 학습하도록 유도한다.

## 📎 Related Works

### 1. 스트리밍 AED 모델

기존 스트리밍 AED 모델은 입력 분할 방식에 따라 두 그룹으로 나뉜다.

- **인코더 측 분할:** NT, ACS, CIF, SCAMA 등이 있으며, 고정 크기 블록이나 적응형 정지 메커니즘을 통해 토큰을 생성한다.
- **디코더 측 분할:** Local windowing, GMM attention, HMA, MoChA 등이 있으며, 디코더 상태를 쿼리로 사용하여 입력을 분할한다. 특히 MoChA는 추론 시 선형 시간 복잡도를 가지며 문맥 정보를 활용할 수 있어 효율적이다.

### 2. 생성 지연 감소 기법

E2E ASR의 지연 문제를 해결하기 위해 과거에는 하이브리드 ASR 시스템에서 추출한 프레임 수준의 정렬 정보를 사용하는 방식(예: DeCoT, MinLT)이 연구되었다. 그러나 이러한 방식은 외부 하이브리드 모델에 의존해야 한다는 한계가 있다.

### 3. 지식 증류(Knowledge Distillation)

기존의 지식 증류는 주로 교사 모델의 확률 분포를 학생 모델이 모방하게 하는 방식이었다. 반면, 본 논문은 확률 분포가 아닌 **토큰 경계의 위치(position)** 자체를 증류하는 것에 집중하며, 교사와 학생 모델이 인코더를 공유하며 동시에 학습되는 구조를 취한다는 점에서 차별점이 있다.

## 🛠️ Methodology

### 1. 시스템 구조

CTC-ST는 하나의 공유 인코더(Shared Encoder)와 두 개의 분기(Branch)인 CTC 디코더와 MoChA 디코더로 구성된다.

- **CTC 분기:** 인코더의 출력을 받아 CTC 손실 함수로 학습하며, 정렬 정보를 생성하는 교사 역할을 한다.
- **MoChA 분기:** CTC 분기에서 제공하는 정렬 경계를 가이드 삼아 토큰을 생성하는 학생 역할을 한다.

### 2. CTC 정렬 추출

학습 과정에서 Viterbi alignment를 통해 가장 확률이 높은 CTC 경로 $\bar{\pi}$를 추출한다. 여기서 blank 토큰이 아닌 실제 토큰이 나타나는 시점의 인덱스를 추출하여 참조 토큰 경계 $b_{ctc} = (b_{ctc}^1, \dots, b_{ctc}^U)$를 생성한다.

### 3. 학습 목표 및 손실 함수

본 모델은 MoChA의 예상 토큰 경계 위치 $b_{mocha}$가 CTC의 참조 경계 $b_{ctc}$와 일치하도록 하는 **동기화 손실(synchronization loss)** $L_{sync}$를 도입한다.

$$L_{sync} = \frac{1}{U} \sum_{i=1}^{U} |b_{ctc}^i - b_{mocha}^i|$$

여기서 $U$는 출력 시퀀스의 길이이며, $b_{mocha}$는 MoChA의 정렬 확률 $\alpha_{i,j}$를 기반으로 계산된 기대 위치이다. 전체 목적 함수 $L_{total}$은 다음과 같이 정의된다.

$$L_{total} = (1-\lambda_{ctc})L_{mocha} + \lambda_{ctc}L_{ctc} + \lambda_{sync}L_{sync}$$

### 4. 커리큘럼 학습 전략 (Curriculum Learning)

무작위 초기 상태에서 CTC-ST를 적용하면 학습이 불안정해지므로, 2단계 전략을 사용한다.

- **Stage 1:** 양방향 인코더(BLSTM)와 Quantity Regularization(QR)을 사용하여 오프라인 모드로 학습시킨다. 이 단계에서 모델은 $\alpha_{i,j}$의 적절한 스케일을 학습한다.
- **Stage 2:** Stage 1의 가중치를 초기값으로 하여, 지연 시간이 제어된 인코더(LC-BLSTM)와 함께 CTC-ST 손실을 적용해 학습시킨다. 이 단계에서 정확한 토큰 경계 위치를 학습한다.

### 5. SpecAugment와의 결합

SpecAugment는 입력 데이터에 마스크를 씌워 성능을 높이지만, MoChA에서는 정렬 확률 $\alpha_{i,j}$의 재귀적 계산을 무너뜨려 성능을 저하시키는 문제가 있다. CTC-ST는 프레임 단위 독립성을 가진 CTC 스파이크를 활용하므로, 마스킹된 영역에서도 MoChA가 정렬을 회복할 수 있도록 돕는다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** TEDLIUM2, Librispeech, CSJ, AMI (SDM) 등 4종의 벤치마크 데이터셋을 사용하였다.
- **비교 대상:** baseline MoChA (+QR), RNN-T, 그리고 하이브리드 시스템 기반의 DeCoT, MinLT.
- **평가 지표:** Word Error Rate (WER), Token Emission Latency (TEL), Word Emission Latency (WEL).

### 2. 주요 결과

- **인식 정확도 향상:** TEDLIUM2에서 CTC-ST 적용 시 baseline MoChA 대비 WER이 유의미하게 감소하였다. 특히 긴 발화(20초 이상)에서 성능 향상이 두드러졌으며, 이는 정렬 확률 소멸 문제를 효과적으로 해결했음을 보여준다.
- **지연 시간 감소:** TEL 측정 결과, CTC-ST는 토큰 생성 시점을 앞당겨 체감 지연 시간을 크게 줄였다. TEDLIUM2의 중앙값(PT@50) 기준, baseline 대비 약 240ms의 지연 시간이 감소하였다.
- **강건성 입증:** SpecAugment를 적용했을 때, 일반 MoChA는 성능이 하락했으나 CTC-ST를 적용한 모델은 오히려 성능이 향상되어 RNN-T 수준의 정확도에 도달하였다.
- **타 모델과의 비교:**
  - **vs Hybrid Distillation:** 외부 정렬 정보 없이도 DeCoT나 MinLT와 비슷하거나 더 나은 정확도-지연 시간 트레이드오프를 달성하였다.
  - **vs RNN-T:** 인식 정확도는 RNN-T와 대등한 수준까지 끌어올렸으며, 오히려 Word Emission Latency(WEL) 측면에서는 RNN-T보다 더 낮은 지연 시간을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 AED 모델의 고질적인 문제인 '긴 발화에서의 불안정성'과 '생성 지연'을 CTC의 정렬 지식을 이용한 자기 증류 방식으로 해결하였다.

**강점:**

- **Purely End-to-End:** 외부의 하이브리드 모델이나 수동 정렬 데이터 없이 학습이 가능하다는 점이 매우 실용적이다.
- **상호 보완적 구조:** CTC-ST가 MoChA뿐만 아니라 공유 인코더를 통해 CTC 분기 자체의 지연 시간도 함께 줄이는 상호작용 효과가 관찰되었다.
- **효율성:** 추론 시 RTF(Real-Time Factor) 측정 결과, MoChA가 RNN-T보다 빠른 추론 속도를 보였다.

**한계 및 논의:**

- 매우 긴 발화의 경우 여전히 RNN-T와의 성능 격차가 일부 존재하며, 이는 향후 해결해야 할 과제이다.
- MoChA 특유의 결과값이 계속 변하는 'flicker' 현상에 대한 언급이 있으며, 이를 해결하기 위해 안정적인 부분 가설(partial hypotheses)을 선택하는 기법이 필요함을 시사한다.

## 📌 TL;DR

본 연구는 스트리밍 AED 모델인 MoChA의 인식 오류와 토큰 생성 지연을 해결하기 위해, CTC 모델의 정렬 경계를 가이드로 사용하는 **CTC Synchronous Training (CTC-ST)** 기법을 제안하였다. 이를 통해 외부 데이터 없이 순수 E2E 학습만으로도 긴 발화와 노이즈에 강건한 성능을 확보하였으며, 인식 정확도는 RNN-T 수준으로 높이면서 지연 시간과 추론 속도는 오히려 더 개선하는 성과를 거두었다. 이 연구는 실시간 음성 인식 시스템에서 AED 모델의 실용성을 크게 높인 연구라고 평가할 수 있다.
