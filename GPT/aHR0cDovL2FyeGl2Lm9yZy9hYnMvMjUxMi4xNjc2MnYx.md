# NRGPT: An Energy-based Alternative for GPT

Nima Dehmamy, Benjamin Hoover, Bishwajit Saha, Leo Kozachkov, Jean-Jacques Slotine and Dmitry Krotov (2025)

## 🧩 Problem to Solve

본 논문은 현대 언어 모델링의 주류인 Generative Pre-trained Transformer(GPT) 아키텍처와 Energy-Based Modeling(EBM) 패러다임을 통합하고자 한다.

일반적인 GPT는 입력 시퀀스를 고정된 레이어들에 통과시켜 다음 토큰을 예측하는 순방향 연산(forward pass) 구조를 가진다. 반면, EBM은 추론 과정을 에너지 지형(energy landscape) 위에서의 동역학적 과정(dynamical process)으로 간주하며, 데이터와 유사한 샘플은 낮은 에너지 상태에, 그렇지 않은 샘플은 높은 에너지 상태에 배치한다.

EBM 프레임워크를 LLM에 도입하는 것은 매우 중요하다. 왜냐하면 EBM은 추론 과정을 명시적인 최적화 문제로 변환함으로써, 솔루션 공간의 체계적인 탐색을 가능하게 하고, 문제의 난이도에 따른 가변 연산량(variable computation) 설정 및 정규화(regularizers)를 통한 모델 정렬(model alignment)에 자연스러운 해법을 제공하기 때문이다. 그러나 기존의 에너지 기반 트랜스포머들은 주로 마스크드 토큰 예측(masked token prediction)에 치중되어 있어, GPT와 같은 인과적(causal) 언어 모델링 설정에 적용하는 데 한계가 있었다. 따라서 본 연구의 목표는 GPT 설정을 에너지 기반 프레임워크로 변환한 **NRGPT(eNeRgy-GPT)**를 제안하고 그 성능과 이론적 성질을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **GPT 블록의 연산을 에너지 함수의 기울기(gradient)로 재정의**하여, 모델의 추론 과정을 에너지 지형에서의 탐색 과정으로 만드는 것이다.

1. **에너지 기반 GPT 구조 설계**: GPT의 Attention 및 Feed-Forward 네트워크를 각각 에너지 함수 $E_{AT}$와 $E_{FF}$로 설계하고, 이들의 합인 전체 에너지 $E$의 기울기를 통해 토큰 상태를 업데이트하는 규칙을 제안하였다.
2. **이론적 수렴성 증명**: 특정 조건(inference rate $\eta$의 설정) 하에서 NRGPT의 업데이트 규칙이 에너지를 점진적으로 감소시킨다는 점을 증명하였으며, 인과적 마스크(causal mask) 덕분에 토큰들이 순차적으로 고정되는 **점근적 안정성(Asymptotic Stability)** 현상을 규명하였다.
3. **성능 검증 및 효율성 확인**: ListOps(수학 연산), Shakespeare, OpenWebText(OWT) 데이터셋에서 실험을 진행하여, recurrent GPT 및 표준 GPT와 대등한 성능을 보이면서도 파라미터 수 측면에서 더 효율적임을 입증하였다.

## 📎 Related Works

논문에서는 기존의 트랜스포머와 EBM의 관계를 다룬 연구들을 다음과 같이 분석한다.

- **In-context Learning(ICL) 연구**: 일부 연구는 트랜스포머의 ICL 과정이 경사하강법(GD)과 유사하다고 주장하지만, 이는 주로 선형 트랜스포머나 softmax를 생략한 단순화된 설정에서만 논의되었다.
- **Energy Transformer (ET)**: 마스크된 토큰을 복원하는 구조로, 추론 시 에너지 기울기를 따라 최적화를 수행한다. 그러나 이는 양방향(bidirectional) 어텐션을 사용하므로, 미래 토큰을 참조하지 않아야 하는 GPT의 인과적 생성 방식에는 부적합하다.
- **Energy-Based Transformers (EBT)**: 표준 트랜스포머의 출력을 이용해 에너지 값을 계산한다. 즉, 트랜스포머가 에너지 함수를 '계산'하는 도구로 쓰이는 반면, NRGPT는 트랜스포머 블록 '자체'가 에너지 함수의 기울기로 정의된다는 점에서 근본적으로 다르다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

NRGPT는 입력 토큰 시퀀스를 다음 토큰 시퀀스로 매핑하며, 이 과정에서 단일 NRGPT 블록을 반복적으로 적용하는 recurrent 구조를 가진다. 각 반복 단계는 에너지 지형에서의 한 단계의 경사하강법(GD)으로 해석된다.

### 2. 에너지 함수 설계

전체 에너지는 어텐션 에너지와 피드포워드 에너지의 합으로 정의된다:
$$E = E_{AT} + E_{FF}$$

- **Attention Energy ($E_{AT}$)**:
  표준 Multi-Head Attention(MHA)의 동작을 모사하기 위해 다음과 같이 정의한다.
  $$E_{AT}^A(g) = -\frac{1}{\beta} \sum_{h} \alpha_h \log \left( \sum_{B<A} \exp(\beta g_B^T J_h g_A) \right)$$
  여기서 $g$는 정규화된 토큰 상태이며, $J_h$는 학습 가능한 가중치이다. 이 에너지의 기울기를 취하면 표준 어텐션의 출력과 매우 유사한 형태가 된다.

- **Feed-Forward Energy ($E_{FF}$)**:
  Dense Associative Memory 개념을 도입하여 다음과 같은 형태를 제안한다.
  $$E_{FF} = -\sum_{A=1}^N \frac{1}{T} F(W_1 g_A)$$
  여기서 $F$는 활성화 함수 $\sigma$와 관련된 스칼라 함수이며, 이를 통해 표준 MLP의 연산을 에너지 기반으로 구현한다.

### 3. 업데이트 규칙 및 추론 절차

토큰 상태 $x$의 업데이트는 다음과 같은 규칙을 따른다:
$$\dot{x} = x^{(t+1)} - x^{(t)} = -\eta^{(t)} \frac{\partial E}{\partial g^{(t)}}$$
여기서 $\eta^{(t)}$는 **Inference Rate** 행렬로, 추론 시 에너지 지형에서 이동하는 보폭을 결정한다.

### 4. 정규화 및 수렴 조건

LayerNorm 또는 RMSNorm을 사용하여 $g$를 정의하며, $\eta = c \text{diag}(\gamma)$ (여기서 $\gamma$는 정규화 가중치)일 때 에너지가 점진적으로 감소함($\dot{E} < 0$)이 증명되었다.

### 5. 점근적 안정성 (Asymptotic Stability)

인과적 마스크(Causal Mask)로 인해 토큰 $A$의 에너지는 $B \le A$인 토큰들의 상태에만 의존한다. 따라서:

1. 첫 번째 토큰 $x_1$은 외부 영향 없이 자신의 에너지를 최소화하며 수렴한다.
2. $x_1$이 수렴하여 고정되면, 두 번째 토큰 $x_2$는 고정된 $x_1$을 배경으로 자신의 에너지를 최소화하며 수렴한다.
3. 이 과정이 재귀적으로 반복되어 모든 토큰이 결국 안정된 상태(fixed point)에 도달하게 된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: ListOps (중첩 산술 연산), Shakespeare (문자 단위 생성), OpenWebText (OWT, 자연어 생성).
- **비교 대상**: 표준 GPT, Recurrent GPT (GPT_Rec_parallel).
- **지표**: Accuracy (ListOps), Perplexity, GQS (Grammar Quality Score), Distinct-1/2 (다양성).

### 2. 주요 결과

- **ListOps**: NRGPT 변형 모델들이 Recurrent GPT와 유사한 정확도를 보였으며, 일부 설정에서는 더 적은 파라미터로도 학습 가능함을 확인하였다.
- **Shakespeare**: 파라미터 규모에 따른 검증 손실(Validation Loss)이 Recurrent GPT와 대등하였으며, 특히 모델 규모가 커질 때 표준 GPT보다 **과적합(overfitting)에 더 강한 모습**을 보였다.
- **OpenWebText**:
  - 파라미터 수가 표준 GPT보다 약 3,400만 개 적음에도 불구하고, Perplexity 및 생성 품질 지표(GQS, Diversity)에서 매우 경쟁력 있는 결과를 냈다.
  - **MMLU 벤치마크**: 128M 규모의 모델에서 29.3%의 정확도를 기록하며, 동일 파라미터 규모의 GPT 및 Recurrent GPT보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 가치

NRGPT는 GPT의 순방향 연산을 명시적인 에너지 최소화 과정으로 해석함으로써, LLM의 추론 과정을 최적화 관점에서 분석할 수 있는 기반을 마련하였다. 특히 파라미터 효율성이 높으며, 과적합 억제 능력이 뛰어난 점이 고무적이다.

### 2. 한계 및 비판적 해석

- **연산 복잡도(FLOPs)**: 파라미터 수는 적지만, 에너지 기울기를 계산하는 과정에서 발생하는 연산량(FLOPs)은 표준 트랜스포머보다 약 1~2배 높다. 즉, 메모리 효율성을 얻는 대신 계산 시간을 지불하는 트레이드오프가 존재한다.
- **하이퍼파라미터 민감도**: 에너지 기반의 제약 조건으로 인해 표준 GPT보다 하이퍼파라미터 설정에 더 민감하게 반응하는 경향이 있다.
- **학습 속도**: 에너지 지형을 통한 수렴 과정이 필수적이므로, 단순 피드포워드 방식보다 추론 시의 벽시계 시간(wall-clock time)이 느릴 수 있다.

### 3. 논의 사항

저자들은 $\eta$에 엄격한 제약을 두지 않더라도 경험적으로는 수렴이 일어남을 관찰하였다. 이는 모델의 목적 함수 자체가 에너지 최소화를 유도하도록 학습되기 때문일 가능성이 크며, 향후 $\eta$를 더 유연하게 설계할 수 있는 가능성을 시사한다.

## 📌 TL;DR

본 논문은 GPT의 아키텍처를 에너지 기반 모델(EBM)로 재구성한 **NRGPT**를 제안한다. NRGPT는 추론 과정을 에너지 함수 $E = E_{AT} + E_{FF}$의 기울기를 따라가는 최적화 과정으로 정의하며, 인과적 마스크를 통해 모든 토큰이 순차적으로 안정화되는 '점근적 안정성'을 이론적으로 증명하였다. 실험 결과, NRGPT는 더 적은 파라미터로 표준 GPT와 대등하거나 더 우수한 성능(특히 MMLU 및 과적합 억제 측면)을 보였으며, 이는 향후 LLM의 가변 연산 및 모델 정렬 연구에 새로운 방향성을 제시한다.
