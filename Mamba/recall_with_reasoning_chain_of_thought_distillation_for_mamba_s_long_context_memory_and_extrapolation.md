# Recall with Reasoning: Chain-of-Thought Distillation for Mamba’s Long-Context Memory and Extrapolation

Jun-Yu Ma, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu (2025)

## 🧩 Problem to Solve

Mamba와 같은 State Space Model(SSM)은 이론적으로 무한한 컨텍스트 처리 능력을 갖추고 있으며, 추론 시 선형 복잡도(linear complexity)를 보장한다. 하지만 실제 환경에서는 입력 시퀀스의 길이가 모델의 학습 길이(training length)를 크게 초과할 경우, 장기 기억(long-context memory) 능력이 급격히 저하되는 문제와 길이 외삽(length extrapolation) 능력이 부족한 문제가 발생한다.

기존의 접근 방식인 DeciMamba나 ReMamba 등은 중요하지 않은 토큰을 압축하거나 필터링하여 입력 길이를 줄임으로써 이 문제를 해결하려 했다. 그러나 이러한 방식은 원래의 입력 문장이 훼손되어 언어 모델링(language modeling) 전반의 성능을 저하시킬 수 있으며, 필터링 후에도 여전히 매우 긴 컨텍스트에서는 한계를 보인다. 본 논문의 목표는 모델의 아키텍처 수정 없이, 데이터 기반의 Chain-of-Thought (CoT) 증류(distillation) 기법을 통해 Mamba가 긴 컨텍스트에서 핵심 정보를 능동적으로 회상(recall)하고 추론할 수 있도록 만드는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Recall with Reasoning (RwR)**이다. 이는 Mamba의 두 가지 특성, 즉 (1) 디코딩 시 최근 토큰에 우선순위를 두는 경향과 (2) 마지막 토큰의 상태 표현(state representation)이 이론적으로 전체 이력을 인코딩한다는 점에 기반한다.

RwR의 중심 설계는 Mamba가 정답을 바로 내놓는 대신, 먼저 고정된 크기의 상태 메모리에서 질문과 관련된 핵심 내용을 요약하여 추출(recall)하고, 그 요약된 내용을 바탕으로 최종 답변을 생성(reasoning)하도록 학습시키는 것이다. 이를 위해 더 강력한 교사 모델(Transformer 기반 LLM)로부터 생성된 쿼리 맞춤형 요약본을 CoT 프롬프트로 사용하여 Mamba를 미세 조정(Fine-tuning)한다.

## 📎 Related Works

본 연구는 크게 세 가지 관련 연구 분야를 다룬다.

1. **State Space Models (SSMs):** S4를 시작으로 데이터 의존적 SSM 레이어인 S6를 도입한 Mamba가 대표적이다. Mamba는 선형 복잡도를 가지며 트랜스포머보다 효율적이지만, 앞서 언급한 길이 외삽 문제가 존재한다.
2. **Long-context Memory:** 고정 길이로 학습된 모델이 긴 컨텍스트에서 성능이 저하되는 문제를 해결하기 위해, DeciMamba나 ReMamba와 같이 중요하지 않은 토큰을 제거하는 압축 방식이 제안되었다. 하지만 이러한 방식은 입력 토큰의 손실을 초래하여 기본 언어 모델링 성능을 떨어뜨리는 한계가 있다.
3. **CoT Distillation:** 거대 모델(예: GPT-4)의 추론 과정(rationales)을 추출하여 작은 모델(SLM)에게 전수하는 기법이다. 본 논문은 이를 응용하여, 단순한 추론 단계의 전수가 아닌 '장기 기억의 회상'을 위한 요약 능력을 증류하는 데 초점을 맞추었다.

## 🛠️ Methodology

RwR의 전체 과정은 데이터 생성 단계와 추론 전략 단계로 나뉜다.

### 1. Summary-based CoT Construction

Mamba에게 "관련 정보 요약 $\rightarrow$ 정답 생성"의 사고 흐름을 학습시키기 위해 다음과 같은 데이터셋 $D_s$를 구축한다.

- **Valid Summary Extraction ($D_f$):**
    OpenOrca 데이터셋에서 배경 컨텍스트($c$), 쿼리($q$), 정답($a$)으로 구성된 쌍을 선택한다. Llama-3.1-8B-Instruct 모델을 사용하여 쿼리에 답변하는 데 필요한 핵심 요약본 $s$를 추출한다. 추출된 요약본 $s$가 실제 정답 $a$와 일치하는지 GPT-4o를 통해 검증하며, 일치하는 샘플만 $D_f = \{(c, q, s)\}$에 저장한다.
- **Empty Summary Construction ($D_e$):**
    모든 쿼리가 컨텍스트 내에서 답을 찾을 수 있는 것은 아니다. 모델의 과잉 확신(overconfidence)을 방지하고 무관한 문단을 구별하는 능력을 키우기 위해, 정답이 포함된 부분을 의도적으로 제거한 수정된 컨텍스트 $c_r$을 생성한다. 이에 대해 요약본이 "정보 부족(empty)"임을 학습하도록 $D_e = \{(c_r, q, \text{empty})\}$를 구축한다.
- **Training:**
    최종 학습 데이터셋은 $D_s = [D_f, D_e]$이며, 이를 OpenOrca 데이터셋과 함께 사용하여 Mamba를 SFT(Supervised Fine-tuning)한다.

### 2. Segmented Summarization for Answering (SSA)

매우 긴 입력 시퀀스를 처리할 때, 모델의 메모리 부담을 줄이기 위해 적용하는 추론 전략이다.

1. 전체 긴 컨텍스트를 여러 개의 작은 조각으로 나눈다.
2. 학습된 RwR Mamba를 이용해 각 조각별로 요약본을 생성한다.
3. 생성된 모든 요약본을 한데 모아 최종 쿼리에 대한 답변을 생성한다.

이 방식은 Mamba의 선형 복잡도 덕분에 계산 자원 수요를 크게 늘리지 않으면서도, 각 처리 단계의 길이를 관리 가능한 수준으로 유지하여 기억 능력을 극대화한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 지표:** LONGMEMEVAL(6가지 기억 작업), HELMET(7가지 범주 작업)을 사용하였다.
- **비교 대상 (Baselines):** 기본 Mamba, OpenOrca로 SFT된 Mamba, DeciMamba, ReMamba.
- **모델 설정:** Mamba-2.8b를 백본으로 사용하였으며, 최대 학습 길이는 6,000 토큰으로 제한하였다.

### 주요 결과

- **장기 기억 성능:** Table 1과 2에서 볼 수 있듯, RwR은 10k 및 100k 길이 설정 모두에서 기존 baseline들을 압도한다. 특히 100k 길이에서 DeciMamba와 ReMamba의 성능이 급격히 떨어지는 반면, RwR은 SSA 전략과 결합했을 때 가장 높은 성능을 보였다.
- **단기 기억 및 기본 능력:** Table 3의 짧은 컨텍스트 작업(Dialogue, NLI, Reasoning, Open-QA) 결과, RwR은 기본 Mamba-SFT보다 약간 향상된 성능을 보였다. 반면, 토큰을 압축하는 DeciMamba와 ReMamba는 단기 기억 성능이 크게 하락하여 실용성에 문제가 있음을 보였다.
- **아키텍처 비교 (외삽 능력):** Transformer 기반의 Phi-2와 하이브리드 모델인 Hymba와 비교했을 때, 10k에서는 유사하거나 약간 낮았으나 100k 길이에서는 Phi-2와 Hymba의 성능이 거의 0에 수렴하였다. 이는 Mamba가 길이 외삽(Length Extrapolation) 측면에서 압도적인 우위에 있음을 입증한다.
- **효율성:** 100k 데이터를 처리할 때 Transformer(Phi-2)는 매우 긴 시간이 소요되었으나, Mamba(RwR)는 훨씬 빠른 처리 속도를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba가 가진 이론적 잠재력을 데이터 중심의 CoT 증류를 통해 실현할 수 있음을 보여주었다. 특히, 토큰을 직접 삭제하는 방식이 아니라 **"요약이라는 사고 과정"**을 학습시킴으로써 모델의 일반적인 언어 모델링 능력을 유지하면서도 장기 기억 능력을 비약적으로 상승시켰다는 점이 매우 고무적이다.

또한, 실험을 통해 SSM이 Transformer나 하이브리드 구조보다 극단적으로 긴 시퀀스에 대한 외삽 능력이 훨씬 뛰어남을 확인하였다. 이는 향후 초장문 컨텍스트 처리 모델 설계 시 SSM 기반 아키텍처가 매우 강력한 후보가 될 수 있음을 시사한다.

다만, 몇 가지 한계점이 존재한다. 첫째, Mamba-2.8b 모델 하나만으로 실험이 진행되어 Mamba2나 Falcon Mamba 등 다른 SSM 변형 모델에서도 동일한 효과가 나타날지는 검증되지 않았다. 둘째, 최대 100k까지는 테스트하였으나 그 이상의 극단적인 길이(예: 200k 이상)에 대한 탐색은 이루어지지 않았다. 셋째, Mamba의 사전 학습 길이(2k)가 최신 Transformer 모델(예: Llama-3의 128k)보다 훨씬 짧기 때문에, 최신 SOTA 모델들과의 절대적 성능 비교에는 제약이 있다.

## 📌 TL;DR

본 연구는 Mamba의 장기 기억 및 길이 외삽 문제를 해결하기 위해, 교사 모델로부터 쿼리 맞춤형 요약을 학습하는 **Recall with Reasoning (RwR)** 기법을 제안한다. 이 방법은 토큰 삭제 없이 CoT 증류를 통해 Mamba가 스스로 정보를 회상하도록 가이드하며, 실험 결과 100k 길이의 컨텍스트에서도 기존 SSM 기반 압축 모델 및 Transformer 모델보다 뛰어난 성능과 효율성을 보였다. 이는 SSM의 효율성을 유지하면서 장기 기억 능력을 극대화할 수 있는 실용적인 방법론을 제시한 것으로, 향후 초장문 LLM 연구에 중요한 기초가 될 것으로 보인다.
