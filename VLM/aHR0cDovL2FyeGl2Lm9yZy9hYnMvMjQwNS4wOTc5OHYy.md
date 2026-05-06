# Many-Shot In-Context Learning in Multimodal Foundation Models

Yixing Jiang, Jeremy Irvin, Ji Hun Wang, Muhammad Ahmed Chaudhry, Jonathan H. Chen, Andrew Y. Ng (2024)

## 🧩 Problem to Solve

본 연구는 멀티모달 파운데이션 모델(Multimodal Foundation Models, LMMs)에서 In-Context Learning(ICL)의 확장성, 특히 **Many-Shot ICL**의 가능성을 탐구한다.

기존의 ICL 연구는 모델의 제한된 컨텍스트 윈도우(Context Window)로 인해 소수의 예시(Few-shot, 보통 100개 미만)만을 사용하는 데 그쳤다. 하지만 최근 GPT-4o(128,000 토큰)와 Gemini 1.5 Pro(최대 100만 토큰)와 같이 매우 긴 컨텍스트 윈도우를 가진 모델들이 등장하면서, 수백에서 수천 개의 예시를 제공했을 때 모델의 성능이 어떻게 변화하는지 확인 할 수 있는 환경이 조성되었다.

논문의 목표는 다양한 도메인과 태스크에 걸쳐 예시의 수를 Few-shot에서 Many-shot으로 확장했을 때, LMM의 성능 향상 정도를 정량적으로 분석하고, 오픈 웨이트(Open-weights) 모델과의 성능 격차 및 추론 비용 최적화 방안(Batching)을 조사하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **Many-Shot ICL의 효용성 증명**: 폐쇄형(Closed-weights) 모델인 Gemini 1.5 Pro와 GPT-4o가 많은 수의 예시가 제공될 때 성능이 실질적으로 향상됨을 보였다. 특히 Gemini 1.5 Pro는 예시 수의 증가에 따라 로그-선형(log-linear)적으로 성능이 개선되는 경향을 보였다.
2. **폐쇄형 vs 오픈 웨이트 모델의 격차 확인**: Llama 3.2-Vision, InternLM-XComposer2.5와 같은 오픈 웨이트 모델들은 예시 수를 늘려도 성능 향상이 거의 없음을 발견하여, 두 진영 간의 ICL 능력에 상당한 격차가 있음을 밝혔다.
3. **ICL 데이터 효율성(Data Efficiency) 측정**: 모델이 예시로부터 얼마나 빠르게 학습하는지를 측정하는 지표를 도입하여 Gemini 1.5 Pro가 GPT-4o보다 더 높은 데이터 효율성을 가짐을 입증했다.
4. **쿼리 배칭(Batching)의 효과 분석**: 여러 개의 쿼리를 하나의 API 호출로 묶어 처리하는 배칭 기법이 추론 비용과 지연 시간(Latency)을 획기적으로 줄일 뿐만 아니라, 특히 Zero-shot 설정에서 성능을 오히려 향상시킬 수 있음을 발견했다.

## 📎 Related Works

- **Scaling ICL**: LLM 분야에서는 예시 수를 늘렸을 때 성능이 향상된다는 연구가 있었으나, 대부분 텍스트 전용 벤치마크에 국한되었으며 모델 간의 비교 분석은 부족했다.
- **Multimodal ICL**: LMM의 ICL 능력을 다룬 초기 연구들이 있었으나, 시각적 토큰의 방대한 양으로 인해 컨텍스트 윈도우의 제약이 컸으며, 이번 연구처럼 수천 개의 예시를 사용하는 Many-shot 설정은 시도되지 않았다.
- **Batch Querying**: 추론 비용 절감을 위해 여러 쿼리를 묶어 보내는 배칭 프롬프팅 연구가 존재했으나, 본 논문은 이를 Many-shot ICL 상황과 결합하여 성능 및 비용 관점에서 분석했다는 점에서 차별점이 있다.

## 🛠️ Methodology

### 전체 파이프라인

연구팀은 14개의 데이터셋(자연 이미지, 의료 이미지, 원격 탐사, 분자 이미지 등)을 사용하여 이미지 분류, 시각적 질의응답(VQA), 객체 로컬라이제이션(Object Localization) 태스크를 수행했다. 각 데이터셋에서 클래스 균형을 맞춘 Demonstration set(예시 세트)과 Test set을 구성하여 모델에 입력하였다.

### ICL 데이터 효율성 측정 방법

모델이 예시의 수 $N$에 따라 얼마나 효율적으로 성능을 올리는지 측정하기 위해 다음과 같은 선형 회귀 분석을 수행한다.

$$\text{Performance} \approx \beta \cdot \log_{10}(N + 1) + \text{Zero-shot Performance}$$

여기서 $\beta$ 값이 바로 ICL 데이터 효율성을 나타내며, 이는 예시 수가 10배 증가할 때 기대되는 성능 향상 폭을 의미한다.

### 쿼리 배칭(Batching) 절차

단일 쿼리를 반복 전송하는 대신, 최대 50개의 쿼리를 하나의 프롬프트에 포함시켜 요청한다.

- **Many-shot Batching**: [Many-shot Examples] $\rightarrow$ [Query 1, Query 2, ..., Query 50] $\rightarrow$ [Response Format]
- **Zero-shot Batching**: [Query 1, Query 2, ..., Query 50] $\rightarrow$ [Response Format]

### 추론 설정

- **모델**: GPT-4o, Gemini 1.5 Pro (주요 분석 대상), GPT-4(V)-Turbo, Llama 3.2-Vision, InternLM-XComposer2.5.
- **하이퍼파라미터**: 결정론적인 응답을 위해 $\text{temperature} = 0$으로 설정하였다.

## 📊 Results

### Many-Shot ICL 성능

- **Gemini 1.5 Pro**: 대부분의 데이터셋에서 예시 수가 늘어날수록 성능이 꾸준히 상승했다. 특히 1,000개 이상의 예시를 제공했을 때 최적의 성능을 보이는 경우가 많았으며, 로그-선형적인 개선 추세를 보였다.
- **GPT-4o**: 전반적으로 성능 향상이 있었으나 Gemini에 비해 불안정했다. 일부 데이터셋에서는 예시 수가 늘어남에 따라 성능이 급락했다가 다시 상승하는 V자형 곡선을 그리기도 했다.
- **오픈 웨이트 모델**: Llama 3.2-Vision과 InternLM-XComposer2.5는 예시 수 증가에 따른 성능 향상이 거의 나타나지 않았다.

### ICL 데이터 효율성

- Gemini 1.5 Pro가 대부분의 데이터셋에서 GPT-4o보다 높은 $\beta$ 값을 기록하여, 더 적은 수의 예시로 더 큰 성능 향상을 이끌어내는 높은 데이터 효율성을 보였다.

### 쿼리 배칭의 영향

- **비용 및 지연 시간**: Many-shot 설정에서 배칭을 사용할 경우, 단일 쿼리 방식 대비 지연 시간은 최대 35배, 비용은 최대 45배까지 절감되었다.
- **Zero-shot 성능 향상**: 놀랍게도 Zero-shot 설정에서 쿼리를 배칭했을 때 성능이 크게 향상되었다. 이에 대한 원인을 분석한 결과, 다음 세 가지 요소의 결합으로 판단된다.
    1. **Domain Calibration**: 동일 도메인의 이미지를 많이 봄으로써 모델이 도메인에 적응함.
    2. **Class Calibration**: 라벨이 없더라도 다양한 클래스의 이미지를 통해 출력을 보정함.
    3. **Self-ICL**: 자기회귀(Autoregressive) 디코딩 과정에서 모델이 앞서 생성한 정답을 예시로 활용함.

## 🧠 Insights & Discussion

### 강점 및 가능성

본 연구는 폐쇄형 LMM들이 방대한 양의 예시를 통해 별도의 파라미터 업데이트(Fine-tuning) 없이도 새로운 도메인이나 태스크에 빠르게 적응할 수 있음을 보여주었다. 이는 모델이 출시된 직후 매우 빠르게 특수 목적용으로 최적화하여 사용할 수 있다는 실무적인 이점을 제공한다.

### 한계 및 비판적 해석

1. **클래스 수의 제약**: 컨텍스트 윈도우가 비약적으로 늘어났음에도 불구하고, 수백 개 이상의 클래스를 가진 데이터셋의 경우 모든 클래스의 예시를 담기에 여전히 부족하다.
2. **데이터 오염(Data Contamination)**: 폐쇄형 모델의 학습 데이터가 공개되지 않았으므로, 사용된 벤치마크 데이터셋이 이미 학습에 포함되었을 가능성을 배제할 수 없다. 다만, Zero-shot 성능이 완벽하지 않다는 점이 이를 어느 정도 부정하는 근거가 된다.
3. **오픈 웨이트 모델의 무능력**: 오픈 웨이트 모델들이 Many-shot ICL에서 전혀 혜택을 보지 못한 점은 매우 충격적이며, 이는 단순히 모델 크기의 문제가 아니라 학습 과정에서의 컨텍스트 활용 능력(Long-context handling)에 근본적인 차이가 있음을 시사한다.

## 📌 TL;DR

이 논문은 LMM에서 예시 수를 수천 개까지 늘린 **Many-Shot ICL**의 효과를 분석했다. **Gemini 1.5 Pro**와 **GPT-4o**는 예시 수가 늘어남에 따라 성능이 크게 향상되었으며, 특히 Gemini 1.5 Pro가 더 안정적이고 효율적인 학습 능력을 보였다. 반면, **오픈 웨이트 모델들은 Many-shot ICL의 혜택을 전혀 받지 못했다.** 또한, **쿼리 배칭**을 통해 추론 비용과 시간을 획기적으로 줄이면서도 성능을 유지하거나(Many-shot), 심지어 향상(Zero-shot)시킬 수 있음을 입증했다. 이 결과는 LMM을 특정 도메인에 적응시킬 때 파인튜닝의 대안으로 Many-shot ICL이 매우 유용한 전략이 될 수 있음을 시사한다.
