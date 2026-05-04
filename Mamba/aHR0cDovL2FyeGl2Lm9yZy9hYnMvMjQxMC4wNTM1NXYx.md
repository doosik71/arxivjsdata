# Falcon Mamba: The First Competitive Attention-free 7B Language Model

Jingwei Zuo, Maksim Velikanov, Dhia Eddine Rhaiem, Ilyas Chahed, Younes Belkada, Guillaume Kunsch, Hakim Hacid (2024)

## 🧩 Problem to Solve

현대 대규모 언어 모델(LLM)의 주류인 Transformer 아키텍처는 Attention 메커니즘의 특성상 시퀀스 길이($L$)에 대해 이차 복잡도($O(L^2)$)의 계산 및 메모리 비용이 발생하는 한계가 있다. 이를 해결하기 위해 FlashAttention과 같은 최적화 기법이나 State Space Model(SSM) 기반의 Mamba, RWKV와 같은 선형 복잡도 모델들이 제안되었다.

하지만 기존의 SSM 기반 모델들은 주로 소규모 모델에서만 성능이 검증되었거나, 7B 규모 이상의 대형 모델에서는 최적화된 Transformer 모델들과 비교했을 때 여전히 성능 격차가 존재하는 문제가 있었다. 특히 최근의 Mamba 기반 모델들은 성능 향상을 위해 Attention 레이어를 섞은 하이브리드(Hybrid) 설계를 채택하는 경향이 있는데, 이는 순수 SSM 설계만으로는 고성능 Transformer를 능가하기 어렵다는 가설을 뒷받침한다. 본 논문의 목표는 순수 Mamba 아키텍처만으로도 대규모 데이터와 모델 크기에서 최신 Transformer 모델들과 경쟁 가능한 수준의 성능을 낼 수 있음을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 순수 Mamba 아키텍처를 기반으로 한 7B 규모의 언어 모델인 Falcon Mamba 7B를 개발하고, 이를 통해 Attention-free 모델이 Transformer 및 하이브리드 모델과 대등하거나 더 우수한 성능을 낼 수 있음을 보여준 것이다. 주요 직관은 하이브리드 설계가 제공하는 성능 이점이 아키텍처 자체의 필연성보다는 데이터의 품질과 학습 전략의 정교함에 의해 결정될 수 있다는 점이다. 이를 위해 5.8조 개의 토큰을 사용한 정밀한 데이터 믹스처와 학습 레시피를 적용하여, 메모리 효율성과 추론 속도라는 SSM의 장점을 유지하면서도 높은 언어 능력을 확보하였다.

## 📎 Related Works

기존의 연구들은 Transformer의 연산 효율성을 높이기 위해 Sliding Window Attention 등을 제안했거나, RWKV, Griffin, Mamba와 같은 새로운 아키텍처를 제시하였다. 특히 Mamba-7B-rw, Zamba 7B, Samba 3.8B, Mamba2 8B 등이 등장하며 규모를 확장하려는 시도가 있었다.

그러나 이러한 모델들의 대부분은 Attention 레이어를 교차 배치한 하이브리드 설계를 통해 성능을 높였으며, 순수 SSM 모델은 여전히 복제(Copying) 작업이나 In-context Learning에서 Transformer보다 뒤처진다는 평가를 받아왔다. Falcon Mamba 7B는 이러한 하이브리드 설계 없이 순수 Mamba 구조만으로 Mistral 7B, Llama 3.1 8B와 같은 강력한 Transformer 기반 모델과 경쟁하며, 순수 SSM 모델의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 아키텍처

Falcon Mamba 7B는 Mamba (Gu & Dao, 2023) 아키텍처를 기반으로 하며, Attention 레이어가 전혀 없는 순수 SSM 구조이다. 모델의 주요 파라미터는 다음과 같다.

- **레이어 수**: 64개
- **모델 차원 ($d_{model}$)**: 4096
- **어휘 사전 크기 ($vocab\_size$)**: 65,024
- **특이 사항**: 모델의 유연성을 높이기 위해 입력 임베딩과 출력 가중치를 분리(Untied)하여 학습하였다.

### 모델 안정화 기법

학습 중 무작위로 발생하는 Loss Spike 현상을 해결하기 위해, B, C, $\Delta$ 파라미터 이후에 RMSNorm 레이어를 추가하였다. 이는 Mamba 아키텍처가 Transformer보다 학습률(Learning Rate)에 더 민감하다는 관찰 결과에 따른 조치이며, 이를 통해 학습 안정성을 확보하였다.

### 학습 절차 및 전략

1. **최적화 및 스케줄링**:
   - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}, \text{weight decay}=0.1$)
   - LR Schedule: Warmup-Stable-Decay (WSD) 스케줄을 적용하였다.
   - 최대 학습률 $\eta_{max} = 6.4 \times 10^{-4}$에서 시작하여, decay 단계에서는 $\eta_{min} = \frac{\eta_{max}}{256}$까지 지수적으로 감소시켰다.

2. **Batch Scaling**:
   - 배치 사이즈를 128에서 2048까지 선형적으로 증가시키는 Ramp-up 과정을 거쳤다.
   - 이때 Adam 옵티마이저의 Noise Temperature $T_{noise}$를 일정하게 유지하기 위해 다음과 같은 방정식을 사용하여 학습률 $\eta$를 조정하는 Batch Scaling을 적용하였다.
   $$T_{noise} = \frac{\eta}{\sqrt{b}}$$
   여기서 $b$는 배치 사이즈를 의미한다.

3. **데이터 구성 및 커리큘럼**:
   - 총 5.8조 개의 토큰을 사용하였으며, RefinedWeb(웹 데이터), Curated(도서, 논문 등), Code, Math 데이터를 혼합하였다.
   - **커리큘럼 학습**: 시퀀스 길이를 $2\text{k} \to 4\text{k} \to 8\text{k}$로 점진적으로 늘리며 4단계의 학습 스테이지를 거쳤다.
   - **Decay Stage**: 마지막 단계에서는 FineWeb-Edu, Cosmopedia와 같은 고품질 데이터와 소량의 Instruction 데이터를 추가하여 제로샷 및 퓨샷 학습 능력을 강화하였다.

## 📊 Results

### 벤치마크 성능

Falcon Mamba 7B는 HF Leaderboard v1 및 v2의 다양한 지표에서 Transformer 기반 모델들과 대등하거나 상회하는 결과를 보였다.

- **비교 대상**: Llama 3.1 8B, Mistral 7B, Falcon2 11B, RecurrentGemma 9B, RWKV-v6 Finch 7B/14B 등.
- **정량적 결과**:
  - **HF Leaderboard v1**: Llama 3.1 8B 및 Mistral 7B보다 높은 평균 성능을 기록하며, 특히 순수 SSM 모델 중에서는 최상위 성능을 보였다.
  - **HF Leaderboard v2**: IFEval, MMLU-PRO 등에서 경쟁력 있는 수치를 기록하였으며, 특히 긴 문맥 추론 능력을 측정하는 MuSR 벤치마크에서 강점을 보였다.

### 추론 효율성 및 메모리 분석

- **메모리 사용량**: Transformer 모델은 문맥 길이가 길어질수록 KV 캐시로 인해 메모리 사용량이 선형적으로 증가하지만, Falcon Mamba 7B는 상태(State)만 저장하므로 생성 단계에서 메모리 사용량이 일정하게 유지된다.
- **처리량 (Throughput)**: 생성 토큰 수가 130k까지 증가하더라도 처리 속도가 저하되지 않고 일정하게 유지됨을 확인하였다.
- **Prefill 전략**: Parallel Prefill 방식에서는 메모리 제약이 존재하지만, Sequential Prefill 방식을 사용할 경우 이론적으로 무제한의 프롬프트 길이를 처리할 수 있음을 보였다.

## 🧠 Insights & Discussion

본 연구는 순수 Mamba 설계가 적절한 데이터 믹스처와 학습 전략(특히 WSD 스케줄과 Batch Scaling)이 뒷받침된다면, Attention 메커니즘 없이도 최신 Transformer 모델과 경쟁할 수 있음을 증명하였다. 이는 "Attention is all you need"라는 기존의 지배적인 패러다임에 도전하는 결과이다.

**강점 및 한계**:

- **강점**: 추론 시 메모리 비용이 일정하며, 극도로 긴 시퀀스 생성에서 압도적인 효율성을 가진다.
- **한계**: Transformer에 비해 In-context Learning 능력이 다소 부족할 가능성이 제기되었으나, 고품질 CoT(Chain-of-Thought) 데이터와 정밀한 튜닝을 통해 이를 어느 정도 완화할 수 있음을 확인하였다.
- **향후 과제**: 본 모델은 8k 문맥 길이로 학습되었으나, SSM의 잠재력을 완전히 활용하기 위해 초거대 문맥(Extra-large context)에 특화된 학습 전략과 검증이 필요하다.

## 📌 TL;DR

Falcon Mamba 7B는 Attention 메커니즘을 완전히 제거한 순수 Mamba 아키텍처 기반의 7B 모델로, 5.8조 개의 토큰 학습을 통해 Llama 3.1 8B 및 Mistral 7B와 같은 최신 Transformer 모델에 필적하는 성능을 달성하였다. 특히 생성 시 메모리 사용량과 추론 속도가 시퀀스 길이에 관계없이 일정하게 유지되는 SSM의 이점을 극대화하였으며, 이는 향후 저지연·대규모 생성 작업(오디오, 비디오 등)을 위한 효율적인 모델 설계에 중요한 이정표가 될 것으로 기대된다.
