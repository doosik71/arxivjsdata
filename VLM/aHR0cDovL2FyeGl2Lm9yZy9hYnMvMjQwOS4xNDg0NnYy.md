# A-VL: Adaptive Attention for Large Vision-Language Models

Junyang Zhang, Mu Yuan, Ruiguang Zhong, Puhan Luo, Huiyou Zhan, Ningkang Zhang, Chengchen Hu, Xiang-Yang Li (2025)

## 🧩 Problem to Solve

Large Vision-Language Models(LVLMs)는 컴퓨터 비전과 자연어 처리 기술을 통합하여 강력한 능력을 보여주지만, 추론 과정에서 막대한 컴퓨팅 자원을 요구한다. 특히 LLM의 자기회귀(autoregressive) 생성 특성상, 이전에 생성된 모든 토큰의 KV Cache를 유지해야 하므로 시퀀스 길이가 길어질수록 메모리 사용량과 계산 비용이 선형적으로 증가한다.

이 문제는 고해상도 이미지를 처리하는 LVLM에서 더욱 심화된다. 고해상도 이미지는 수백에서 수천 개의 시각적 토큰을 생성하며, 이는 KV Cache의 급격한 팽창을 야기하여 실시간 시스템 배포에 큰 장애물이 된다. 기존의 Adaptive Attention 기법들은 주로 단일 모달리티인 언어 모델(LLM)을 위해 설계되었으며, 시각 정보와 텍스트 정보가 공존하는 LVLM의 특수한 attention 패턴을 고려하지 않았다는 한계가 있다. 따라서 본 논문의 목표는 LVLM의 모달리티별 특성을 반영하여 성능 저하 없이 메모리와 계산 오버헤드를 줄이는 plug-and-play 방식의 적응형 어텐션 메커니즘인 A-VL을 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 직관은 LVLM 내의 서로 다른 모달리티가 서로 다른 attention 패턴을 보인다는 점이다. 연구진은 시각적 토큰과 텍스트 토큰의 attention 양상을 분석하여 다음과 같은 설계 아이디어를 도출하였다.

1. **모달리티별 분리 관리**: 시각적 입력은 시퀀스 앞부분에 위치함에도 불구하고 생성 과정 내내 지속적으로 높은 attention을 받지만, 텍스트 입력은 시간이 지남에 따라 attention이 급격히 감소하는 경향이 있다.
2. **시각적 Attention의 희소성 및 드리프트(Drift) 활용**: 시각적 attention은 매우 희소(sparse)하여 소수의 토큰이 대부분의 attention을 점유하며, 시간이 흐름에 따라 중요한 토큰의 집합이 서서히 변하는 '연속성(continuity)'을 보인다.
3. **계층적 캐시 관리**: 모든 시각적 토큰을 계산에 사용하는 대신, 잠재적으로 유용한 토큰은 메모리에 보관(Secondary)하되, 실제 계산에는 가장 핵심적인 토큰(Core)만 사용하는 구조를 제안한다.

## 📎 Related Works

기존의 추론 최적화 연구는 크게 추론 프로세스 최적화(Page Attention, Speculative Sampling 등)와 계산량 감소(Adaptive Attention, Pruning 등)로 나뉜다.

* **LLM 대상 Adaptive Attention**: StreamingLLM은 고정된 위치의 캐시를 유지하고, $H_2O$는 누적 attention 점수를 기반으로 중요한 캐시를 보존한다. 하지만 이러한 방법들은 텍스트 전용 모델을 위해 설계되어 LVLM의 시각적 토큰 특성을 반영하지 못한다.
* **Vision 모델 대상 Pruning**: SPViT나 PuMer 등은 Vision Transformer에서 토큰을 제거하여 효율성을 높였으나, 이는 주로 인코더 모델을 대상으로 한다.
* **LVLM 대상 최적화**: FastV는 두 번째 레이어 이후 일부 이미지 토큰을 제거할 수 있음을 보였으나, KV Cache를 사용하는 자기회귀 추론 환경에서는 잠재적으로 중요한 토큰이 조기에 제거되어 성능이 저하되는 문제가 있다.

A-VL은 이러한 기존 연구들과 달리, KV Cache를 유지하면서도 모달리티별 특성에 맞게 계산 대상과 보관 대상을 동적으로 분리함으로써 성능과 효율성 사이의 트레이드-오프를 최적화한다.

## 🛠️ Methodology

### 1. Attention 분석 및 지표

연구진은 먼저 attention 점수를 다음과 같이 정의하여 분석하였다.
$$s_t^l = \sum_{h=1}^H A_{t,h}^l / H$$
여기서 $A_{t,h}^l$은 $l$번째 레이어의 $h$번째 헤드에서 마지막 토큰이 이전 토큰들에 할당한 attention 가중치이다. 또한, 서로 다른 레이어나 스텝 간의 attention 일관성을 측정하기 위해 $p$-percentile concordance index(PPCI)를 도입하였다.
$$\text{p\%PPCI} = \frac{|T_1 \cap T_2|}{p\% \cdot N}$$
이 지표를 통해 시각적 attention의 **이질성(Heterogeneity, 레이어별로 주목하는 토큰이 다름)**과 **연속성(Continuity, 동일 레이어 내에서는 짧은 기간 동안 주목하는 토큰이 유지됨)**을 확인하였다.

### 2. Adaptive Text-Aware Vision Attention

시각적 토큰의 희소성과 드리프트 특성을 해결하기 위해 계층적 관리 방식을 도입한다.

* **분류 (Prefill 단계)**: 마지막 텍스트 토큰의 attention 점수를 기준으로 이미지 토큰을 세 그룹으로 나눈다.
  * **Core Tokens**: 상위 $C\%$ 토큰. 실제 계산에 직접 참여한다.
  * **Secondary Tokens**: 상위 $S\%$ 토큰 (단, $S > C$). 메모리에 저장되지만 기본적으로는 계산에서 제외된다.
  * **Minor Tokens**: 나머지 토큰. 즉시 제거되어 메모리를 절약한다.
* **동적 업데이트 (Decode 단계)**: 시각적 attention의 드리프트를 반영하기 위해, 매 $K$ 스텝마다 모든 Secondary 토큰을 사용하여 추론을 수행하고, 이를 통해 Core 토큰 집합을 업데이트한다.
* **Prefill 최적화**: FastV의 아이디어를 일부 채용하여, prefill 단계의 두 번째 레이어 이후 상위 $P\%$의 이미지 토큰만 남기고 나머지는 제거하여 초기 계산 부하를 줄인다.

### 3. Adaptive Text Attention

텍스트 토큰은 attention이 빠르게 감소하므로 $H_2O$의 방식을 적용한다. 텍스트 캐시 윈도우 크기를 전체의 $T\%$로 설정하고, 누적 attention 점수가 낮은 중복 캐시를 제거하여 지역성(Locality)을 유지한다.

### 4. 구현 및 CUDA Operator

KV Cache를 슬라이싱하여 계산하는 과정에서 메모리 비연속성으로 인해 오버헤드가 발생한다. 이를 해결하기 위해 선택된 행/열과 직접 곱셈을 수행하는 전용 **Custom CUDA Operator**를 개발하여 슬라이싱 비용을 제거하고 속도를 높였다.

## 📊 Results

### 실험 설정

* **데이터셋 및 작업**: Image Caption (Nocaps, Flickr30k), VQA (DocVQA, TextVQA, VQAv2), OCR (OCRBench) 등 다양한 입도(granularity)의 작업 수행.
* **모델**: LLaVA 1.5 (7B, 13B), LLaVA 1.6 (7B, 13B), Qwen-VL.
* **비교 대상**: Original 모델, FastV, $H_2O$. (모든 비교 방법은 KV Cache의 50%만 유지하는 조건으로 설정)

### 주요 결과

1. **성능 유지**: Table 2에 따르면, A-VL은 KV Cache를 50%만 사용함에도 불구하고 Original 모델과 거의 동일하거나(near-lossless), 오히려 일부 지표에서 더 높은 성능을 보였다. 특히 세밀한 정보가 필요한 OCR 및 VQA 작업에서 FastV나 $H_2O$보다 월등한 성능을 기록했다.
2. **메모리 절감**: Table 3에서 LLaVA-1.6 7B 모델 기준, Original의 1179 MB였던 KV Cache 메모리를 531 MB(T=70% 설정 시)까지 줄여 약 55%의 메모리 절감 효과를 거두었다.
3. **추론 속도 향상**: KV Cache를 50%로 압축함으로써 디코더 추론 속도가 1.8배 향상되었으며, 전용 CUDA Operator를 통해 추가로 1.1배의 속도 향상을 얻었다. 결과적으로 전체 디코더 지연 시간(latency)을 기존의 **50.5% 수준**으로 단축시켰다.

## 🧠 Insights & Discussion

본 논문은 LVLM의 효율적인 추론을 위해서는 단순히 토큰을 줄이는 것이 아니라, **모달리티별 attention의 동역학(dynamics)을 이해하는 것이 필수적임**을 입증하였다.

* **강점**: A-VL은 모델의 재학습이나 파인튜닝 없이 적용 가능한 plug-and-play 방식이라는 점이 매우 강력하다. 또한, 단순히 메모리 점유율만 낮춘 것이 아니라, 실제로 계산에 참여하는 토큰 수(Core)를 더 적게 유지함으로써 실질적인 연산량 감소를 이끌어냈다.
* **비판적 해석 및 한계**:
  * **파라미터 민감도**: $S, C, K, P, T$ 등 설정해야 할 하이퍼파라미터가 많다. 비록 실험을 통해 최적값을 제시했지만, 모델의 크기나 작업의 종류에 따라 이 값들이 어떻게 변해야 하는지에 대한 일반화된 가이드라인은 부족하다.
  * **CUDA Operator 의존성**: 성능 향상의 상당 부분이 하드웨어 가속(Custom CUDA Op)에 의존하고 있어, 다른 가속기나 프레임워크로 이식할 때 동일한 효율을 낼 수 있을지는 미지수이다.
  * **Prefill 단계의 한계**: Prefill 단계에서 $P\%$만 남기는 전략은 연산량을 줄여주지만, 너무 낮게 설정할 경우 정보 손실이 발생한다는 점이 확인되었다. 이는 시각 정보의 압축과 보존 사이의 근본적인 트레이드-오프가 여전히 존재함을 시사한다.

## 📌 TL;DR

A-VL은 LVLM에서 시각적 토큰(희소성, 드리프트 특성)과 텍스트 토큰(급격한 감쇠 특성)의 서로 다른 attention 패턴을 분석하여, 이를 분리 관리하는 적응형 어텐션 기법이다. 시각적 토큰을 Core/Secondary/Minor로 계층화하여 관리하고 전용 CUDA 커널을 도입함으로써, **성능 저하 없이 KV Cache 메모리 사용량을 절반으로 줄이고 추론 지연 시간을 약 50% 수준으로 단축**하였다. 이 연구는 향후 고해상도 이미지 처리가 필수적인 실시간 멀티모달 AI 서비스의 배포 효율성을 크게 높일 것으로 기대된다.
