# MoFE: Mixture of Frozen Experts Architecture

Jean Seo, Jaeyoon Kim, Hyopil Shin (2025)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(Large Language Models, LLMs)의 거대한 파라미터 규모로 인해 발생하는 막대한 계산 비용과 메모리 요구량, 그리고 데이터 확보의 어려움이라는 자원 제약 문제를 해결하고자 한다. 일반적으로 모델의 성능은 모델 크기, 데이터 양, 계산 능력에 따라 예측 가능하게 향상되는 Scaling Law를 따르지만, 이는 실제 환경에서 모델의 개발과 배포를 매우 어렵게 만든다.

따라서 연구의 목표는 높은 성능을 유지하면서도 훈련 효율성을 극대화할 수 있는 효율적인 LLM 학습 방법론을 제시하는 것이다. 특히, 기존의 파라미터 효율적 미세 조정(Parameter-efficient Fine-tuning, PEFT)의 효율성과 Mixture of Experts (MoE) 아키텍처의 확장성(Scalability)이라는 두 가지 장점을 결합하여, 적은 자원으로도 다중 도메인에 능숙한 모델을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MoE 아키텍처 내의 Feed Forward Network (FFN) 레이어를 동결(Freeze)시키는 **Mixture of Frozen Experts (MoFE)** 구조를 제안하는 것이다.

전통적인 MoE는 전문가(Expert)의 수가 늘어남에 따라 전체 파라미터 수가 증가하여 학습 부담이 커지지만, MoFE는 전문가 모델의 핵심인 FFN 블록을 동결함으로써 학습 가능한 파라미터 수를 획기적으로 줄인다. 이를 통해 모델의 전체 규모(전문가 수)를 확장하더라도 학습 비용을 일정하게 유지하면서, 이미 사전 학습된 전문가 모델의 지식을 효과적으로 전이(Knowledge Transfer)받을 수 있도록 설계하였다.

## 📎 Related Works

논문에서는 효율적인 모델 학습을 위한 두 가지 주요 흐름을 소개한다.

1.  **PEFT 및 양자화**: Prompt-tuning, Adapters, LoRA, DoRA와 같은 PEFT 기법들은 학습 가능한 파라미터 수를 줄여 계산 요구량을 낮춘다. 또한, QLoRA와 같이 양자화(Quantization)를 결합하여 효율성을 더욱 높이는 시도가 있었다.
2.  **Mixture of Experts (MoE)**: Mixtral 8x7B와 같은 모델은 여러 개의 전문가 모델을 통합하여 모델 크기를 키우면서도 추론 시 일부 전문가만 활성화함으로써 효율적인 스케일링을 달성하였다.

기존 접근 방식과의 차별점은, 일반적인 MoE가 전문가 레이어를 함께 학습시키거나 PEFT가 모델의 일부 가중치만을 업데이트하는 것과 달리, MoFE는 MoE의 구조적 이점을 취하면서도 전문가의 FFN 레이어를 완전히 동결함으로써 PEFT 수준의 효율성과 MoE 수준의 확장성을 동시에 확보하려 했다는 점이다.

## 🛠️ Methodology

### 전체 시스템 구조
MoFE는 Mixtral 아키텍처를 기반으로 하며, `mergekit`을 통해 구축된다. 시스템은 크게 **Base Model**, **Expert Model**, **Router**의 세 가지 구성 요소로 이루어진다.

### 주요 구성 요소 및 역할
1.  **Base Model (학습 가능)**:
    - 전체 아키텍처의 Embedding 레이어와 Self-attention 레이어를 제공한다.
    - MoFE 구조에서 이 부분의 파라미터들은 학습 과정에서 업데이트된다.
    - 본 실험에서는 1.1B 파라미터 규모의 TinyLlama를 기반 모델로 사용하였다.
2.  **Expert Model (동결)**:
    - Transformer 구조에서 Attention 레이어 다음에 위치하는 FFN 레이어를 제공한다.
    - FFN은 토큰 임베딩의 등방성(Isotropy)을 유지하는 역할을 수행한다.
    - **핵심 설계**: MoFE에서는 모든 전문가 모델의 FFN 블록을 동결한다. 따라서 전문가의 수가 늘어나더라도 학습해야 할 파라미터는 증가하지 않는다.
3.  **Router (학습 가능)**:
    - 각 타임 스텝에서 입력 토큰에 대해 어떤 FFN 블록(전문가)을 활성화할지 결정하는 Gate 역할을 한다.
    - 선형 레이어로 구성되며, 입력 벡터와 모델의 은닉 상태(Hidden states) 간의 내적(Dot product)을 계산하여 점수가 가장 높은 상위 $m$개의 전문가를 선택한다. 본 실험에서는 $m=2$로 설정하였다.

### 학습 절차 및 목표
- **학습 범위**: Router와 Base Model(Embedding, Attention)의 파라미터만 업데이트하고, Expert Model의 FFN 블록은 고정한다.
- **추론 절차**: 입력 토큰 $\rightarrow$ Base Model(Attention) $\rightarrow$ Router(전문가 선택) $\rightarrow$ 선택된 $m$개의 Frozen FFNs $\rightarrow$ 최종 출력 순으로 진행된다.

## 📊 Results

### 실험 설정
- **하드웨어**: NVIDIA A100 80GB GPU 3장.
- **하이퍼파라미터**: Batch size 4, Learning rate $3 \times 10^{-5}$, Gradient accumulation 512, Weight decay 0.01.
- **데이터셋 및 지표**: 일반 도메인(MMLU), 의료 도메인(MedMCQA)을 사용하였으며, `lm-evaluation-harness`를 통해 성능을 측정하였다.

### 주요 결과
1.  **효율성 및 성능 트레이드오프**:
    - 전문가 수(Small 2, Medium 4, Large 8)에 관계없이 MoFE의 학습 가능 파라미터는 $0.34\text{B}$로 일정하며, 학습 시간 또한 약 6시간으로 고정된다.
    - 반면, 전체 미세 조정(Full Fine-tuning)은 전문가 수에 따라 학습 시간이 14시간에서 26시간까지 선형적으로 증가한다.
    - 성능 면에서는 Full Fine-tuning이 약간 우세하지만, MoFE는 매우 적은 비용으로 이에 근접하는 성능을 보였다.

2.  **타 PEFT 방법론과의 비교**:
    - LoRA, QLoRA, DoRA와 비교했을 때, MoFE는 학습 시간은 가장 짧으면서도 MMLU와 MedMCQA 모두에서 가장 높은 성능을 기록하였다.

3.  **도메인 전문성 전이 (Knowledge Transfer)**:
    - **전문가 구성**: 의료 전문가 모델의 수를 늘릴수록 MedMCQA 성능이 향상됨을 확인하여, 동결된 FFN을 통해서도 도메인 지식이 전이됨을 입증하였다.
    - **다중 도메인**: 의료 및 금융 전문가를 섞어서 구성했을 때 두 도메인 모두에서 성능 향상이 있었으나, 전문가 수와 성능이 반드시 선형적으로 비례하지는 않았다.
    - **Base Model 영향**: 다중 도메인 모델을 구축할 때, 특정 도메인 전문가보다는 일반(General) 모델을 Base Model로 사용하는 것이 가장 효율적이었다.

4.  **최적 학습 전략**:
    - Post-pretraining(추가 사전 학습) 후 미세 조정을 진행하는 전략은 오히려 성능을 크게 떨어뜨렸다. 이는 FFN이 동결된 상태에서 다른 레이어만 업데이트될 경우 레이어 간 미정렬(Misalignment)이 발생하기 때문으로 분석된다. 따라서 직접적인 Instruction-tuning이 가장 효과적이다.

## 🧠 Insights & Discussion

### 강점
MoFE는 MoE의 확장성과 PEFT의 효율성을 동시에 잡은 구조이다. 특히 전문가 모델의 FFN을 동결함으로써, 기존에 구축된 다양한 도메인 전문가 모델들을 최소한의 추가 학습만으로 통합하여 다중 도메인에 능숙한 모델을 만들 수 있다는 점이 매우 강력한 실용적 이점으로 작용한다.

### 한계 및 가정
- **모델 규모의 제한**: 실험이 1.1B 규모의 소형 모델(TinyLlama)에서만 진행되어, 초거대 모델에서도 동일한 효율성과 성능 경향이 나타날지는 명시되지 않았다.
- **데이터의 제한**: 소수의 도메인과 제한된 데이터셋을 사용하여 일반화 성능에 대한 추가 검증이 필요하다.
- **FFN 동결의 영향**: FFN이 지식 저장의 핵심 역할을 한다는 가정하에 동결을 진행하였으나, 새로운 지식을 완전히 학습시키기에는 한계가 있으며 이는 Post-pretraining 시의 성능 저하로 나타났다.

### 비판적 해석
본 연구는 "Frozen FFN"이 지식의 저장소 역할을 하며, Router와 Attention 레이어의 튜닝만으로도 충분히 경로 최적화가 가능하다는 점을 시사한다. 하지만 성능-효율성 트레이드오프 표에서 Full Fine-tuning 대비 약간의 성능 하락이 관찰되는 점은, 전문가 모델 간의 유기적인 통합을 위해서는 일부 파라미터의 업데이트가 불가피함을 보여준다. 그럼에도 불구하고 학습 시간을 1/4 수준으로 줄이면서 PEFT보다 높은 성능을 낸 것은 실무적으로 매우 유의미한 결과이다.

## 📌 TL;DR

본 논문은 MoE 아키텍처에서 FFN 레이어를 동결시켜 학습 효율을 극대화한 **MoFE(Mixture of Frozen Experts)**를 제안한다. 이를 통해 전문가 수와 무관하게 학습 파라미터와 시간을 일정하게 유지하면서도, 기존 도메인 전문가 모델들의 지식을 효율적으로 전이받을 수 있음을 증명하였다. 특히 LoRA 등 기존 PEFT보다 우수한 성능과 빠른 속도를 보여, 자원이 제한된 환경에서 다중 도메인 특화 모델을 구축하는 데 매우 유용한 전략이 될 것으로 기대된다.