# LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models

Yanwei Li, Chengyao Wang, Jiaya Jia (2023)

## 🧩 Problem to Solve

본 논문은 Vision Language Models(VLMs)가 비디오 및 이미지 이해 작업에서 직면하는 **토큰 생성 오버헤드(token generation challenge)** 문제를 해결하고자 한다. 현재의 VLM들은 이미지 캡셔닝이나 시각적 질의응답(VQA)에서는 뛰어난 성능을 보이지만, 긴 비디오를 처리할 때는 막대한 양의 시각적 토큰이 생성되어 계산 비용이 기하급수적으로 증가하는 문제가 발생한다.

예를 들어, BLIP은 이미지 한 장당 32개, LLaVA는 256개 이상의 토큰을 사용한다. 만약 10,000프레임으로 구성된 비디오를 처리한다면 320,000개 이상의 토큰이 필요하게 되며, 이는 현재 LLM의 컨텍스트 창(context window) 용량을 훨씬 초과하는 수치이다. 기존의 단순한 시간적 압축(temporal compression) 방식은 장기적인 시간 간격의 표현력을 심각하게 훼손한다는 한계가 있다. 따라서 본 연구의 목표는 핵심 정보를 보존하면서도 토큰 수를 획기적으로 줄여, 수 시간 분량의 긴 비디오를 처리할 수 있는 효율적인 VLM 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 각 비디오 프레임을 두 개의 서로 다른 토큰, 즉 **Context Token**과 **Content Token**으로 표현하는 **Dual-token strategy**이다.

1. **Context Token**: 사용자의 입력(instruction)에 기반하여 이미지의 전반적인 문맥을 인코딩한다. 이는 사용자의 질문과 가장 관련이 깊은 시각적 정보를 단 하나의 토큰으로 응축하여 효율성을 극대화한다.
2. **Content Token**: 각 프레임의 세부적인 시각적 단서를 캡처한다. 계산 자원 상황에 따라 토큰 수를 유연하게 조절할 수 있으며, 비디오 입력 시에는 프레임당 1개로 압축하고 단일 이미지 입력 시에는 더 많은 토큰을 할당하여 세부 정보를 유지한다.

이러한 설계를 통해 LLaMA-VID는 긴 비디오의 토큰 과부하 문제를 해결하는 동시에, Context Token이라는 추가적인 정보를 통해 모델의 성능 상한선을 높였다.

## 📎 Related Works

### Large Language Models (LLMs)

Transformer의 등장 이후 GPT, LLaMA와 같은 LLM들은 방대한 텍스트 데이터를 통해 복잡한 언어 작업에서 탁월한 능력을 보여주었다. 특히 Alpaca나 Vicuna와 같은 Instruction Tuning 방식은 LLM이 사용자의 지시사항을 더 잘 따르도록 개선하였다.

### Vision Language Models (VLMs)

CLIP, ALIGN과 같은 초기 모델에서 시작하여, 최근에는 BLIP-2, Flamingo와 같이 LLM의 능력을 시각 데이터로 확장하려는 시도가 이어지고 있다. LLaVA는 단순한 선형 프로젝터(linear projector)를 통해 이미지와 텍스트 공간을 정렬하여 강력한 성능을 보였다.

### 기존 비디오 처리 방식의 한계

Video-LLaMA나 VideoChat은 BLIP-2를 통해 비디오 임베딩을 추출하고, Video-ChatGPT는 공간 및 시간적 풀링(pooling)을 제안하였다. 그러나 이러한 방식들은 여전히 프레임당 필요한 토큰 수가 너무 많아, 1시간이 넘는 매우 긴 비디오 시퀀스를 처리하는 데에는 한계가 있다. LLaMA-VID는 프레임당 단 2개의 토큰만으로 인코딩함으로써 이 문제를 해결하며 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

LLaMA-VID의 프레임워크는 크게 **Visual Encoder**, **Text Decoder**, **Token Generation Module**, 그리고 **LLM**으로 구성된다. 전체 파이프라인은 입력 이미지/비디오를 시각적 임베딩으로 변환하고, 사용자의 지시어에 맞게 Context 및 Content 토큰을 생성하여 LLM에 전달하는 구조이다.

### 주요 구성 요소 및 절차

1. **Encoder and Decoder**:
    * **Visual Encoder**: 사전 학습된 ViT(예: EVA-G)를 사용하여 비디오 프레임 $V_t$로부터 시각적 임베딩 $X_t \in \mathbb{R}^{N \times C}$를 추출한다.
    * **Text Decoder**: 사용자의 지시어를 입력받아 텍스트 가이드 쿼리 $Q_t \in \mathbb{R}^{M \times C}$를 생성한다. 이 모듈은 BERT나 QFormer로 구현될 수 있으며, 사용자의 의도를 시각적 쿼리로 변환하는 역할을 한다.

2. **Token Generation**:
    * **Context Token 생성**: Context Attention 메커니즘을 통해 텍스트 쿼리와 시각적 임베딩을 결합하여 단 하나의 Context-related embedding $E_t$를 생성한다. 방정식은 다음과 같다.
    $$E_t = \text{Mean}(\text{Softmax}(Q_t \times X_t^T) \times X_t)$$
    여기서 $\text{Softmax}$와 $\text{Mean}$ 연산은 각각 $N$과 $M$ 차원을 따라 수행된다. 이후 선형 프로젝터를 통해 LLM 공간에 정렬된 Context Token $E_t^T$가 된다.
    * **Content Token 생성**: 적응형 풀링(Adaptive Pooling) 전략을 사용한다. 비디오의 경우 효율성을 위해 글로벌 풀링을 통해 1개의 토큰 $E_t^V$로 압축하며, 단일 이미지의 경우 더 많은 토큰을 유지하여 디테일을 보존한다.
    * **최종 결합**: 생성된 $E_t^T$와 $E_t^V$를 결합(concatenate)하여 해당 프레임을 대표하는 토큰 셋을 구성한다.

### 학습 전략 (Training Strategy)

학습은 총 3단계로 진행된다.

* **Stage 1: Modality Alignment**: 790K개의 이미지-비디오 캡션 쌍을 사용하여 시각적 특징을 언어 공간에 정렬한다. 이때 Visual Encoder와 Text Decoder는 동결(freeze)하고 프로젝터와 Context Attention만 최적화한다.
* **Stage 2: Instruction Tuning**: 텍스트, 이미지, 비디오가 포함된 다중 모달 지시어 데이터셋을 사용하여 LLM의 이해 능력을 높인다. Visual Encoder를 제외한 모든 모듈을 최적화한다.
* **Stage 3: Long Video Tuning (Optional)**: 15K개의 긴 비디오 QA 쌍(MovieNet 기반)을 사용하여 수 시간 분량의 비디오 처리 능력을 학습시킨다. 이 과정에서 **Position Interpolation** 기술을 사용하여 LLM의 컨텍스트 창을 4K에서 64K로 확장한다.

## 📊 Results

### 실험 설정

* **모델**: Visual Encoder(EVA-G), Text Decoder(QFormer), LLM(Vicuna-7B/13B).
* **데이터셋**: 비디오(MSVD, MSRVTT, ActivityNet), 이미지(GQA, MMB, MME, POPE, SEED, SQA, VizWiz, VQA v2).

### 정량적 결과

1. **비디오 기반 벤치마크**:
    * MSVD-QA, MSRVTT-QA, ActivityNet-QA에서 기존 SOTA 모델인 BT-Adapter 등을 제치고 가장 높은 정확도를 기록하였다. 특히 Vicuna-7B 기반 모델로도 기존의 더 큰 모델들보다 우수한 성능을 보였다.
    * 비디오 생성 성능 벤치마크에서도 Correctness, Detail, Context, Temporal, Consistency 등 모든 지표에서 압도적인 성능 향상을 달성하였다.

2. **이미지 기반 벤치마크**:
    * LLaVA-1.5와 동일한 학습 데이터 및 해상도를 사용했을 때, LLaMA-VID가 대부분의 벤치마크에서 더 높은 성능을 보였다. 특히 GQA, MME, VizWiz에서 유의미한 상승이 있었으며, 이는 Context Token이 추가적인 정보를 효과적으로 제공함을 시사한다.

### 정성적 결과

* 단일 이미지에 대해서는 세부 특징을 정확히 인식하고 복잡한 대화를 수행하는 능력을 보였다.
* 3시간 분량의 장편 영화(예: Avatar, Titanic)에 대해 줄거리 요약, 인물 관계 추론, 세부 장면 묘사 등 고차원적인 추론 작업을 성공적으로 수행하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 **효율성과 성능의 트레이드-오프를 성공적으로 해결**했다는 점이다. 실험을 통해 Context Token이 단 1개의 토큰만으로도 모델 성능을 크게 향상시킨다는 것을 입증하였다. 이는 사용자의 질문에 따라 이미지의 특정 영역에 동적으로 집중하는 Context Attention의 효과로 분석된다. 또한, Content Token의 수를 유연하게 조절함으로써 자원 제한이 심한 긴 비디오 처리와 디테일이 중요한 이미지 처리 모두에 대응할 수 있는 범용성을 확보하였다.

### 한계 및 논의사항

논문에서는 긴 비디오 처리를 위해 Position Interpolation을 사용하였으나, 64K 이상의 극단적으로 긴 컨텍스트를 처리할 때의 안정성이나 계산 효율성에 대한 심층적인 분석은 부족하다. 또한, 데이터셋 구축 과정에서 GPT-4와 Claude-2와 같은 외부 LLM에 의존하여 지시어 쌍을 생성했으므로, 생성된 데이터의 품질이 모델 성능에 직접적인 영향을 미쳤을 가능성이 크다.

## 📌 TL;DR

LLaMA-VID는 비디오 프레임을 **사용자 지시어 기반의 Context Token 1개**와 **시각 정보 기반의 Content Token $n$개**로 표현하는 효율적인 토큰 생성 전략을 제안한다. 이를 통해 기존 VLM의 고질적인 문제였던 토큰 과부하를 해결하여 수 시간 분량의 긴 비디오를 처리할 수 있게 되었으며, 동시에 이미지 이해 작업에서도 기존 SOTA 모델인 LLaVA-1.5를 능가하는 성능을 보였다. 이 연구는 효율적인 시각적 표현 방식이 미래의 초거대 멀티모달 모델 설계에 중요한 기준이 될 것임을 시사한다.
