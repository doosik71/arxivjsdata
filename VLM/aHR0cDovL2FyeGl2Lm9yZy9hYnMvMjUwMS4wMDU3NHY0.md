# VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling

Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, Limin Wang (2025)

## 🧩 Problem to Solve

본 논문은 멀티모달 거대 언어 모델(Multimodal Large Language Models, MLLMs)이 영화, 온라인 스트리밍과 같은 매우 긴 비디오 컨텍스트(long-context video)를 효율적으로 이해하도록 하는 문제를 해결하고자 한다.

기존의 MLLM들은 짧은 비디오 이해에서는 우수한 성능을 보이지만, 수 시간 분량의 긴 비디오를 처리할 때 다음과 같은 한계가 존재한다. 첫째, 비디오 프레임 수가 증가함에 따라 생성되는 비디오 토큰의 양이 방대해져 계산 복잡도와 추론 비용이 기하급수적으로 증가한다. 둘째, 이를 해결하기 위해 토큰 압축 기술을 적용할 경우, 시각적 세부 정보가 손실되어 성능이 저하되는 문제가 발생한다. 특히 일부 긴 비디오 모델은 이미지 기반 MLLM보다 낮은 성능을 보이기도 한다.

따라서 본 연구의 목표는 **효율성(Efficiency)과 성능(Performance) 사이의 균형**을 맞추어, 매우 긴 비디오 컨텍스트를 처리하면서도 세부 정보를 보존할 수 있는 모델 아키텍처, 학습 데이터, 학습 전략 및 평가 벤치마크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **계층적 비디오 토큰 압축(Hierarchical video token Compression, HiCo)**과 **단기-장기 학습(Short-to-Long Learning)** 체계이다.

1. **HiCo (Hierarchical Compression):** 비디오의 시각적 중복성을 활용하여 Clip-level과 Video-level의 2단계 압축을 수행한다. 이를 통해 성능 손실 거의 없이 약 1/50의 극단적인 압축률을 달성한다.
2. **LongVid 데이터셋:** 30만 시간의 비디오와 20억 개의 텍스트 어노테이션을 포함하는 대규모 실제 장기 비디오 데이터셋을 구축하였다.
3. **Short-to-Long Learning 전략:** 이미지 및 짧은 비디오에서 기본 시각 인지 능력을 먼저 학습한 후, 짧은 비디오와 긴 비디오 데이터를 공동 학습시키는 다단계 전략을 제안한다.
4. **Multi-Hop NIAH 벤치마크:** 기존의 단순 정보 검색(retrieval) 수준의 Needle-In-A-Video-Haystack(NIAH)을 넘어, 복잡한 추론 능력을 평가할 수 있는 "Multi-Hop NIAH" 벤치마크를 설계하였다.
5. **VideoChat-Flash 모델:** 위 요소들을 결합하여 2B 및 7B 규모에서 GPT-4o 및 Gemini-1.5 Pro와 같은 폐쇄형 모델을 능가하는 성능을 가진 Video MLLM을 구현하였다.

## 📎 Related Works

기존의 긴 비디오 이해 접근 방식은 크게 두 가지 방향으로 나뉜다.

1. **컨텍스트 윈도우 확장 (Context Window Extension):** Gemini-1.5 Pro와 같이 LLM의 컨텍스트 윈도우를 확장하여 수천 개의 프레임을 직접 입력하는 방식이다. 이는 가능성은 보여주었으나, 계산 부담과 처리 비용이 매우 높아 실용성이 떨어진다는 한계가 있다.
2. **비디오 토큰 압축 (Video Token Compression):** Llama-Vid와 같이 토큰을 매우 작게 압축하여 효율성을 높이는 방식이다. 하지만 높은 압축률은 필연적으로 세부 정보 손실을 초래하며, 때로는 이미지 전용 모델보다 성능이 낮아지는 결과가 나타났다.

본 논문은 이러한 기존 방식의 한계를 극복하기 위해, 단순한 토큰 감소가 아닌 **계층적 구조의 압축**과 **단계적 학습 전략**을 통해 효율성과 성능의 트레이드오프를 해결하고자 한다.

## 🛠️ Methodology

### 1. HiCo: Hierarchical Video Token Compression

HiCo는 비디오 컨텍스트 압축을 두 단계로 분리하여 수행한다.

**가. Clip-level Compression (Encoding 단계)**
비디오를 여러 개의 클립으로 분할한 뒤, 각 클립 내에서 시공간적 중복성을 제거한다.

- **Duration-based Sampling:** 비디오 길이 $D$에 따라 샘플링 프레임 수 $T$를 동적으로 결정한다.
  $$T = \min(T_{max}, \max(D, T_{min}))$$
  샘플링 밀도 $\phi$는 $\phi(T, D) = T/D$로 정의되어, 짧은 비디오는 조밀하게(dense), 긴 비디오는 희소하게(sparse) 샘플링한다.
- **Spatio-Temporal Encoding:** Spatio-temporal attention이 적용된 비디오 인코더를 사용하여 프레임 간 정보를 집계하고, **Similar Token Merging (ToMe)** 기법을 통해 유사한 토큰을 병합한다. 이를 통해 프레임당 평균 16개의 토큰만 남기는 고압축을 달성한다.

**나. Video-level Compression (LLM 추론 단계)**
LLM이 토큰을 처리하는 과정에서 텍스트 쿼리와 무관한 시각 토큰을 제거하는 **Progressive Visual Dropout** 전략을 사용한다.

- **Shallow Layers (얕은 층):** Uniform drop을 통해 소수의 토큰을 균일하게 제거하여 계산량을 줄이면서 구조를 유지한다.
- **Deep Layers (깊은 층):** Text-guided select를 통해 텍스트 토큰과 시각 토큰 간의 상관관계를 분석하고, 가장 중요한 정보만 남긴다.

### 2. LongVid 데이터셋 및 학습 전략

**LongVid 데이터셋:** Ego4D, HowTo100M, HD-Vila, MiraData 등에서 큐레이션하여 114,228개의 긴 비디오와 3,444,849개의 QA 쌍을 구축하였다.

**Short-to-Long Learning (4단계 학습):**

1. **Stage-1 (Alignment):** 이미지/짧은 비디오-텍스트 쌍을 통해 시각 인코더와 LLM 사이의 Projector를 학습시킨다.
2. **Stage-2 (Short Video Pre-training):** 350만 장의 이미지와 250만 개의 짧은 비디오-텍스트 쌍으로 기본 시각 인지 능력을 강화한다.
3. **Stage-3 (Joint Instruction Tuning):** 이미지, 짧은 비디오, 긴 비디오(60~3600초)를 혼합하여 다양한 길이의 비디오 작업과 지시어 이행 능력을 학습시킨다.
4. **Stage-4 (High-Resolution Post-finetuning):** 입력 해상도를 224에서 448로 높여 세부 인지 능력을 향상시킨다.

### 3. Multi-Hop NIAH-Video 벤치마크

단순히 이미지 하나를 찾는 것이 아니라, 여러 이미지로 구성된 **추론 경로(Reasoning Path)**를 따라가야 하는 벤치마크이다.

- 모델은 시작점의 단서를 통해 다음 이미지를 찾고, 최종적으로 "바늘(Needle)" 이미지에 도달하여 질문에 답해야 한다.
- 오답 경로(Wrong paths)를 함께 삽입하여 모델이 단순 암기나 정보 유출(leakage)을 통해 정답을 맞히는 것을 방지한다.

## 📊 Results

### 1. 일반 비디오 이해 성능

- **정량적 결과:** VideoChat-Flash는 2B 및 7B 모델 규모에서 MVBench, LongVideoBench, MLVU, VideoMME 등 주요 벤치마크에서 SOTA 성능을 기록하였다. 특히 7B 모델은 폐쇄형 모델인 GPT-4o 및 Gemini-1.5 Pro를 능가하는 결과를 보였다.
- **Temporal Grounding:** 단순한 텍스트 프롬프트("The video lasts for N seconds...")만으로도 타임스탬프 인지 능력을 확보하여 뛰어난 성능을 보였다.

### 2. 긴 비디오 컨텍스트 평가

- **Single-Hop NIAH:** 10,000 프레임 입력 시 **99.1%의 정확도**를 기록하며 오픈소스 MLLM 중 최상위 성능을 입증하였다. (LongVA: 3k 프레임에서 92%, Llama-VID: 10k 프레임에서 55%)
- **Multi-Hop NIAH:** 추론 경로를 따라가야 하는 고난도 작업에서도 CAP(바늘 찾기)과 QA(답변) 모두에서 기존 베이스라인 대비 약 8포인트 높은 성능을 보였다.

### 3. 효율성 분석

- **계산 비용:** 1,000 프레임 처리 시 VideoChat-Flash의 FLOPs는 LongVILA 대비 수십 배, 10,000 프레임에서는 수백 배(9,969.5 vs 1,184,250.0 TFLOPs) 낮다.
- **메모리:** 단일 A100-80G GPU에서 10,000 프레임 추론이 가능한 유일한 모델이었다.

## 🧠 Insights & Discussion

**강점 및 통찰:**

- **계층적 압축의 유효성:** Clip-level에서 중복성을 먼저 제거하고, LLM 내부에서 텍스트 가이드 기반으로 다시 한번 압축하는 방식이 효율성과 성능을 동시에 잡는 핵심이었다.
- **LLM 층별 특성 발견:** 실험을 통해 LLM의 얕은 층에서는 시각 토큰이 전반적으로 분산되어 있어 Uniform drop이 유리하고, 깊은 층으로 갈수록 특정 영역에 집중되므로 Attention-based select가 더 효과적이라는 것을 발견하였다.
- **데이터 믹스의 중요성:** 짧은 비디오와 긴 비디오를 함께 학습시키는 Short-to-Long 전략이 세부 인지와 장기 문맥 이해 사이의 균형을 맞추는 데 결정적이었다.

**한계 및 논의:**

- Video-level compression(Progressive Visual Dropout)은 추론 시에만 적용되었으며, 학습 단계의 가속화 전략(Sequence Parallelism 등)과의 호환성 문제는 여전히 과제로 남아 있다.
- 본 논문은 시각적 토큰 압축에 집중하였으나, 오디오 정보의 통합이나 더 복잡한 멀티모달 상호작용에 대한 논의는 부족하다.

## 📌 TL;DR

본 논문은 **계층적 비디오 토큰 압축(HiCo)**과 **단기-장기 학습 체계**를 통해, 매우 긴 비디오를 효율적으로 처리하는 **VideoChat-Flash**를 제안한다. 1/50이라는 극단적인 압축률을 달성하면서도 10,000 프레임의 컨텍스트에서 99.1%의 검색 정확도를 보였으며, 7B 규모로 GPT-4o 등의 상용 모델을 능가하는 성능을 기록하였다. 특히 대규모 LongVid 데이터셋과 Multi-Hop NIAH 벤치마크를 통해 장기 비디오 모델링의 새로운 기준을 제시하였으며, 이는 향후 실시간 긴 비디오 분석 시스템 구축에 중요한 기여를 할 것으로 보인다.
