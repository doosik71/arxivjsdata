# Tell What You Hear From What You See - Video to Audio Generation Through Text

Xiulong Liu, Kun Su, Eli Shlizerman (2024)

## 🧩 Problem to Solve

본 논문은 무음 비디오에서 적절한 오디오를 생성하는 **Video-to-Audio (V2A) generation** 작업에서 발생하는 **제어 가능성(Controllability)의 부재** 문제를 해결하고자 한다.

일반적인 V2A 작업은 시각적 정보만을 입력으로 받아 오디오를 생성한다. 그러나 동일한 시각적 대상이라도 맥락에 따라 생성되어야 할 소리는 매우 다양하다. 예를 들어, 두 마리의 고양이가 등장하는 영상에서 이들이 친밀하게 지내는지 혹은 영역 다툼을 벌이는지에 따라 생성되는 소리는 완전히 달라져야 한다. 기존의 vision encoder 기반 방식은 이러한 미세한 맥락적 차이를 구분하는 데 한계가 있으며, 결과적으로 시각적 장면과 어울리지 않는 부조화스러운 오디오가 생성되는 문제가 발생한다.

반면, Text-to-Audio (T2A) 모델은 텍스트를 통해 세밀한 제어가 가능하지만, 비디오의 역동적인 시각적 맥락을 반영하지 못해 시간적 정렬(Temporal alignment)이나 시각-청각 간의 시맨틱 일치도가 떨어진다는 단점이 있다. 따라서 본 연구의 목표는 비디오의 시각적 정보와 사용자의 텍스트 프롬프트를 동시에 활용하여, 시각적으로 정렬되면서도 사용자가 의도한 맥락을 반영할 수 있는 제어 가능한 오디오 생성 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Large Language Model (LLM)을 시각-청각 정보의 중간 매개체로 활용**하여, 비디오 특징을 오디오 관련 텍스트 표현으로 변환하고 이를 통해 오디오 생성을 가이드하는 것이다.

1. **VATT (Video-to-Audio Through Text) 프레임워크 제안**: LLM을 통합하여 텍스트 가이드 기반의 V2A 생성과 비디오-오디오 캡셔닝(Video-to-Audio Captioning)을 단일 모델에서 수행할 수 있는 구조를 설계하였다.
2. **V2A Instruction 데이터셋 구축**: 기존 Audio LLM인 LTU-13B를 활용하여 VGGSound 및 AudioSet-2M 데이터셋에 대한 대규모 합성 오디오 캡션 코퍼스를 생성함으로써, 텍스트 조건부 학습을 가능하게 하였다.
3. **효율적인 오디오 생성 메커니즘**: Masked Token Modeling과 Iterative Parallel Decoding을 도입하여, 기존의 자기회귀(Auto-regressive) 방식이나 Diffusion 방식보다 훨씬 빠른 생성 속도를 달성하였다.
4. **성능 입증**: 텍스트 프롬프트가 제공될 때 KLD 점수 1.41이라는 매우 낮은 수치를 기록하며, 기존 SOTA 모델 대비 뛰어난 시각-청각 일치성과 제어 가능성을 입증하였다.

## 📎 Related Works

### 1. Visual-to-Audio Generation

기존 연구는 주로 특정 카테고리(음악 또는 자연음)에 집중해 왔으며, 최근에는 Diffusion 기반 모델들이 등장하여 직접적인 파형(Waveform) 생성을 시도하고 있다. 그러나 이러한 방법들은 세밀한 소리 제어가 어렵고 추론 시간이 매우 길다는 한계가 있다.

### 2. Text-to-Audio Generation

AudioLDM, AudioGen 등 T2A 모델들은 latent diffusion이나 Transformer 기반의 토큰 모델링을 통해 고품질 오디오를 생성한다. 하지만 이들은 시각적 입력에 최적화되지 않았기 때문에, 비디오에 적용했을 때 시간적 정렬이나 시각적 역동성 반영이 부족하다.

### 3. Multi-modal Large Language Models (MLLMs)

LLaMA, Vicuna와 같은 LLM을 확장하여 시각/청각 특징을 텍스트 임베딩 공간으로 투영(Projection)함으로써 이해 능력을 높인 MLLM 연구가 활발하다. VATT는 이러한 MLLM의 추론 능력을 '이해'를 넘어 '오디오 생성의 조건화(Conditioning)' 단계까지 확장했다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

VATT는 크게 두 가지 단계로 구성된 파이프라인을 가진다.

### 1. Video-to-Caption Stage (VATT Converter)

이 단계의 목적은 비디오 특징에서 오디오 생성에 필요한 핵심 시맨틱 정보를 추출하여 텍스트 형태로 변환하는 것이다.

- **구조**: `eva-CLIP` 이미지 엔코더로 추출된 비디오 특징 $V_f = \{v_1, v_2, ..., v_T\}$를 입력으로 받는다.
- **VATT Projector**: 선형 층(Linear layer)을 통해 시각 특징의 차원 $d_v$를 LLM의 임베딩 차원 $d_{lm}$으로 투영하여 $V_{lm}$을 생성한다.
  $$V_{lm} = V_f W_l + b_l$$
- **Instruction Tuning**: 투영된 특징 $V_{lm}$과 지시문 $T_i$를 Vicuna-7B 또는 Gemma-2B와 같은 LLM에 입력하여 오디오 캡션 $T_a$를 생성하도록 학습한다. 이때 효율적인 학습을 위해 LoRA (Low-Rank Adaptation)를 적용하며, 손실 함수로는 Negative Log-Likelihood를 사용한다.
  $$L_{v2t}(T_a | T_i, V_{lm}) = -\sum_{l=1}^{N} \log P_\theta(\hat{t}_{al} = t_{al} | T_i, V_{lm})$$

### 2. Video + Text to Audio Stage (VATT Audio)

첫 번째 단계에서 정렬된 LLM의 은닉 상태(Hidden states)를 조건으로 하여 실제 오디오 토큰을 생성하는 단계이다.

- **Audio Token Representation**: `Encodec` 신경망 코덱을 사용하여 오디오 파형을 4개의 코드북 레벨($L=4$)을 가진 이산 토큰 $A_{tok} \in \mathbb{N}^{L \times T_c}$로 표현한다.
- **VATT Audio Decoder**: Bi-directional Transformer 구조를 사용한다. LLM의 마지막 층 은닉 상태 $H_{lm}$을 $E_{lm}$으로 투영하고, 마스킹 처리된 오디오 토큰 임베딩 $E^M_a$와 결합(Concatenate)하여 입력으로 넣는다.
  $$E_{mm} = \text{Concat}([E_{lm}, E^M_a])$$
- **Masked Token Modeling (MTM)**: 학습 시 일부 오디오 토큰을 $\langle \text{MASK} \rangle$ 토큰으로 교체하고, 이를 예측하도록 학습하는 Cross-entropy 손실 함수를 사용한다.
  $$L_{VATT} = -\sum_{a_{tok} \in A^M_{tok}} I(a_{tok} = \langle \text{MASK} \rangle) \log P_\phi(\hat{a}_{tok} = a_{gt\_tok} | A^M_{tok}; H_{lm})$$

### 3. Inference 및 Decoding 절차

- **Iterative Parallel Decoding**: 추론 시에는 모든 토큰을 $\langle \text{MASK} \rangle$ 상태에서 시작하여, 점진적으로 마스크를 해제하는 방식을 취한다.
- **Cosine Scheduling**: $\cos(\frac{\pi}{2} \cdot \frac{t}{T})$ 스케줄에 따라 마스킹 비율을 조절하며, Gumbel-top-k trick을 통해 신뢰도가 높은 토큰부터 확정적으로 복원한다.
- **최종 변환**: 생성된 오디오 토큰은 Encodec 디코더를 통해 최종 Waveform으로 복원된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: VGGSound 및 AudioSet-2M을 사용하였으며, 합성 캡션 데이터셋인 "V2A Instruction"을 구축하였다.
- **평가 지표**:
  - **KLD (Kullback-Leibler Divergence)**: PaSST 분류기를 통해 생성 오디오와 실제 오디오의 시맨틱 일치도를 측정 (낮을수록 좋음).
  - **FAD (Fréchet Audio Distance)**: 오디오 분포의 전반적인 품질 측정 (낮을수록 좋음).
  - **Align Acc (Alignment Accuracy)**: CAVP 모델을 통해 시각-청각 간의 시간적 정렬도 측정 (높을수록 좋음).
  - **Speed**: 샘플당 생성 시간 측정.

### 2. 주요 결과

- **정량적 성능**: VATT-LLama-T(텍스트 프롬프트 사용) 모델은 **KLD 1.41**을 기록하며 기존 모든 V2A 방법론을 압도하였다. Align Acc 또한 80% 이상의 높은 수치를 보였다.
- **생성 속도**: Iterative Parallel Decoding 덕분에 기존 방법들보다 약 10배 이상 빠른 속도(VATT-Gemma 기준 0.65s)를 기록하였다.
- **텍스트 제어 가능성**: 동일한 비디오에 서로 다른 텍스트 프롬프트를 입력했을 때, 각 프롬프트에 부합하는 서로 다른 오디오가 생성됨을 확인하여 제어 가능성을 입증하였다.
- **비디오-오디오 캡셔닝**: VATT Converter는 LLAVA나 Video-LLAMA 등 기존 MLLM보다 오디오 관련 캡션 생성 능력(CLAP Score 및 NLG 지표)에서 더 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 기여

본 연구는 LLM을 단순히 텍스트 생성기가 아닌, **시각 특징을 오디오 도메인의 시맨틱 공간으로 매핑하는 '컨버터'**로 활용했다는 점이 매우 혁신적이다. 이를 통해 비디오-오디오 캡셔닝과 가이드 기반 오디오 생성을 단일 프레임워크 내에서 통합 구현하였다. 또한, Bi-directional Transformer와 MTM 방식을 통해 생성 효율성을 극대화한 점이 돋보인다.

### 2. 한계 및 비판적 해석

- **Fidelity vs. Relevance**: 정성 평가 결과, VATT는 시맨틱 일치도(Relevance)에서는 매우 뛰어나지만, 절대적인 음질(Fidelity) 측면에서는 Diffusion 기반 모델(예: V2A-Mapper)보다 약간 낮게 평가되었다. 이는 토큰 기반 모델의 고유한 한계일 수 있으며, 향후 더 큰 규모의 코덱 모델이나 하이브리드 구조가 필요함을 시사한다.
- **합성 데이터 의존성**: 모델 학습에 LTU-13B로 생성한 합성 캡션을 사용하였다. 합성 데이터의 품질이 모델의 상한선을 결정하므로, 실제 인간이 작성한 고품질 데이터셋이 추가된다면 성능이 더 향상될 가능성이 크다.
- **텍스트 스타일의 다양성**: 결론 부분에서 언급되었듯, 사용자가 입력하는 텍스트의 스타일이 매우 다양할 경우 모델이 이를 일관되게 처리하는 데 어려움이 있을 수 있다.

## 📌 TL;DR

VATT는 **LLM을 중간 매개체로 사용하여 비디오를 오디오 캡션으로 변환하고, 이를 다시 오디오 생성의 가이드로 활용**하는 제어 가능한 Video-to-Audio 생성 프레임워크이다. 텍스트 프롬프트를 통해 생성되는 소리를 정밀하게 조정할 수 있으며, Masked Token Modeling 기반의 병렬 디코딩을 통해 기존 모델보다 훨씬 빠른 생성 속도를 구현하였다. 이 연구는 향후 사용자와의 대화형 인터페이스를 통한 인터랙티브 오디오 편집 및 생성 도구로 확장될 가능성이 매우 높다.
