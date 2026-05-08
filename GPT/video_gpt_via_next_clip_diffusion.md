# Video-GPT via Next Clip Diffusion

Shaobin Zhuang et al. (2025)

## 🧩 Problem to Solve

본 논문은 시각적 세계 모델링(Visual World Modeling)을 위해 비디오 데이터를 어떻게 효과적으로 학습할 것인가에 대한 문제를 다룬다.

기존의 대규모 언어 모델(LLM)인 GPT 시리즈는 자연어 처리에서 뛰어난 성과를 거두었으나, 언어 시퀀스만으로는 현실 세계의 복잡한 시공간적 세부 사항(spatial-temporal details)을 모두 묘사하기에는 한계가 있다. 예를 들어, 매듭을 묶는 방법과 같은 구체적인 물리적 동작은 텍스트보다 비디오 시퀀스를 통해 더 정확하게 캡처될 수 있다.

비디오 생성 분야에서는 크게 두 가지 접근 방식이 존재해 왔다. 첫째는 **Video Diffusion 모델**로, 생성 품질은 매우 뛰어나지만 장기적인 미래 예측(long-term future prediction)에 어려움을 겪는다. 둘째는 **Autoregressive(자기회귀) 모델**로, 긴 컨텍스트 모델링에는 유리하지만 생성된 결과물의 품질이 디퓨전 모델에 비해 낮다는 단점이 있다. 따라서 본 연구의 목표는 GPT의 자기회귀적 특성과 디퓨전 모델의 고품질 생성 능력을 결합하여, 단기 생성과 장기 예측을 모두 수행할 수 있는 효율적인 비디오 세계 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **비디오 클립(Video Clip)을 언어의 단어(Word)와 같은 기본 단위로 취급**하는 것이다. 이를 통해 GPT의 '다음 토큰 예측(Next Token Prediction)' 개념을 비디오 도메인으로 확장한 **Next Clip Diffusion** 패러다임을 제안한다.

핵심 직관은 비디오를 연속적인 클립의 시퀀스로 보고, 이전의 깨끗한 클립들(clean clips)을 컨텍스트로 삼아 다음의 노이즈가 섞인 클립(noisy clip)을 디노이징(denoising)하여 예측하는 것이다. 이러한 설계를 통해 모델은 GPT의 확장 가능한 자기회귀 구조와 디퓨전의 정교한 합성 능력을 동시에 확보할 수 있다.

## 📎 Related Works

논문에서는 기존 연구를 크게 두 가지 흐름으로 구분하여 설명한다.

1. **Video Diffusion Models**: Sora, Kling, Wan 등 최신 모델들은 텍스트-비디오 생성에서 압도적인 성능을 보이지만, 텍스트 어노테이션에 크게 의존하며 데이터셋의 규모가 LLM에 비해 작고, 텍스트가 비디오의 모든 의미를 완전히 캡처하지 못한다는 한계가 있다.
2. **Autoregressive Video Models**: ImageTransformer, LVM, VideoWorld 등은 픽셀 또는 잠재 공간에서 다음 토큰을 예측하는 방식을 사용한다. 이들은 세계 지식을 학습하는 능력이 입증되었으나, 생성 품질 면에서는 여전히 최신 디퓨전 모델에 뒤처진다.

**Video-GPT와의 차별점**: 기존의 하이브리드 방식들이 주로 이미지 도메인에서 작동하거나 단순한 결합에 그쳤다면, Video-GPT는 언어의 '단어' 개념을 '비디오 클립'으로 치환하여 비디오 클립 수준에서 디퓨전과 자기회귀 모델링을 통합했다는 점에서 독창성을 가진다.

## 🛠️ Methodology

### 1. 클립 시퀀스 구성 (Input Construction)

먼저 학습 비디오에서 $N$개의 프레임을 균일하게 샘플링하고, 이를 $K$개의 클립으로 무작위 분할한다 ($K \sim \text{Uniform}\{2, 3, \dots, N\}$).

**Forward Diffusion Process**: 각 클립의 프레임들에 대해 VAE를 통해 잠재 특징(latent feature)을 추출하고, 효율적인 학습을 위해 **Flow Matching** 방식을 채택하여 노이즈를 추가한다. $k$번째 클립의 $i$번째 프레임에 대한 노이즈 추가 식은 다음과 같다.
$$\Psi(k,i) = \alpha^k \Phi(k,i) + (1-\alpha^k)\epsilon^{k,i}$$
여기서 $\Phi(k,i)$는 원본 잠재 특징, $\Psi(k,i)$는 노이즈가 추가된 특징, $\alpha^k \sim \text{Uniform}[0,1]$는 노이즈 비율, $\epsilon^{k,i} \sim \mathcal{N}(0,I)$는 가우시안 노이즈이다. 특히 클립 내 모든 프레임에 동일한 $\alpha^k$를 적용하여 병렬 추론이 가능하게 한다.

**Noise-Clean Interleaved Sequence**: 모델은 노이즈 클립과 깨끗한 클립이 교차로 배치된 시퀀스를 입력으로 받는다.
$$\text{Input} = [\text{NS}(1,:), \text{CL}(1,:), \dots, \text{NS}(k,:), \text{CL}(k,:), \dots, \text{NS}(K,:]$$

- **Clean Clip ($\text{CL}$)**: $[\langle \text{img} \rangle, \Phi(k,i), \langle /\text{img} \rangle]$ 형태로 구성된다.
- **Noisy Clip ($\text{NS}$)**: $[\langle \text{diff} \rangle, \alpha^k, \Psi(k,i)]$ 형태로 구성되며, $\langle \text{diff} \rangle$ 토큰은 디노이징 힌트를 제공한다.

### 2. 사전 학습: Next Clip Diffusion

모델 아키텍처는 단순함을 위해 **Vanilla Transformer**(Phi-3-mini 기반, 3.8B 파라미터)를 사용하며, 핵심은 **계층적 마스킹(Hierarchical Masking)** 전략에 있다.

- **Clip-Level Mask**: $\text{CL}(k)$는 자신과 이전의 $\text{CL}(1 \dots k-1)$에 의존하며, $\text{NS}(k)$는 자신과 이전의 **깨끗한 클립들($\text{CL}(1 \dots k-1)$)**에만 의존한다. 이는 정확한 컨텍스트를 바탕으로 다음 클립을 예측하기 위함이다.
- **Frame-Level Mask**: $\text{CL}$ 내의 프레임은 인과적(causal) 관계를 가지나, $\text{NS}$ 내의 프레임들은 서로 양방향(bidirectional) 어텐션을 가진다. 이는 디노이징 과정에서 생성 품질을 높이기 위함이다.
- **Patch-Level Mask**: 프레임 내의 패치 토큰들은 공간적 관계를 나타내므로 Full Attention을 적용한다.

**학습 목표**: 모델이 예측한 디노이즈된 클립 특징과 실제 깨끗한 클립 특징 사이의 $L_2$ 손실 함수를 최소화하는 방향으로 학습한다.

**점진적 학습(Progressive Training)**: 계산 비용을 줄이기 위해 16프레임(단순 프레임 예측) $\to$ 48프레임 $\to$ 80프레임으로 점차 비디오 길이를 늘려가며 학습하는 전략을 사용한다.

### 3. 추론: 자기회귀 비디오 예측 (Inference)

추론 단계에서는 이전에 생성된(디노이즈된) 클립들을 깨끗한 컨텍스트로 사용하여 다음 노이즈 클립을 반복적으로 디노이징한다.
$$\text{DN}_{\text{S}}(k+1,:) = \text{Video-GPT}(\text{DN}_{\text{S}}(1,:), \dots, \text{DN}_{\text{S}}(k,:), \text{NS}(k+1,:))$$

## 📊 Results

### 1. 정량적 평가

- **Physics-IQ Benchmark**: 물리 법칙 준수 여부를 측정하는 벤치마크에서 **34.97점**을 기록하며, 2위인 Seine(29.13점) 및 Kling(23.64점)을 크게 상회하는 SOTA 성능을 달성하였다.
- **Kinetics-600**: 예측 불가능한 인간 동작 예측 작업에서 Vanilla Transformer 구조 모델 중 가장 낮은 FVD(315.40)를 기록하여 뛰어난 생성 능력을 입증하였다.

### 2. 다운스트림 태스크 일반화

사전 학습된 Video-GPT를 파인튜닝하여 6가지 작업에서 성능을 검증하였다.

- **생성 작업**: Class-to-Video(UCF-101 SOTA), Text-to-Video, Image Animation에서 높은 품질의 결과물을 생성하였다. 특히 매우 적은 데이터셋으로도 효과적인 전이가 가능함을 보였다.
- **이해 작업**: Video Classification(UCF-101에서 VideoMAEv2보다 높은 58.9% Top-1 정확도), Video Retrieval(MSR-VTT 제로샷 성능 우수), Video Object Segmentation에서 강점을 보였다.

### 3. 절제 연구 (Ablation Study)

- **클립 프레임 수**: 추론 시 클립당 프레임 수가 증가할수록(최대 32프레임까지) 생성 품질이 향상됨을 확인하였다.
- **데이터 규모**: Panda-70M과 같은 대규모 데이터셋을 사용할 때 물리적 세계 모델링 능력이 비약적으로 상승함을 확인하여, LLM처럼 데이터 스케일링의 가능성을 보여주었다.
- **노이즈 추가**: 학습 시 깨끗한 클립에도 약간의 노이즈를 추가하는 것이 추론 시의 괴리를 줄여 성능을 향상시켰다.

## 🧠 Insights & Discussion

**강점**:
본 논문은 비디오 생성의 고질적인 문제였던 '품질'과 '장기 예측 능력' 사이의 트레이드오프를 '클립 단위의 자기회귀 디퓨전'이라는 단순하고 명확한 아이디어로 해결하였다. 특히 Physics-IQ 벤치마크에서의 압도적인 성과는 모델이 단순한 픽셀 복제를 넘어 물리적 세계의 규칙을 어느 정도 학습했음을 시사한다.

**한계 및 논의**:

- **계산 자원**: 3.8B 파라미터 모델을 사용하였으나, 저자들은 가용한 자원의 한계로 더 큰 모델을 실험하지 못했음을 명시하였다. 모델 규모를 더 키운다면 성능이 추가로 향상될 가능성이 매우 높다.
- **가정**: 비디오를 클립으로 나누는 기준이나 Flow Matching의 효율성에 의존하고 있으며, 매우 복잡한 물리적 상호작용이 포함된 영상에서의 한계는 명확히 제시되지 않았다.
- **해석**: 텍스트 없이 비디오 데이터만으로 세계 모델을 구축할 수 있다는 점은 향후 멀티모달 학습의 방향성을 제시한다.

## 📌 TL;DR

Video-GPT는 비디오 클립을 언어의 단어처럼 취급하여, 이전 클립들을 조건으로 다음 클립을 디노이징하는 **Next Clip Diffusion** 패러다임을 제안한다. 이 모델은 70M개의 무라벨 비디오로 사전 학습되어 물리 법칙 예측(Physics-IQ)과 비디오 생성/이해 등 다양한 다운스트림 작업에서 SOTA급 성능을 보였다. 이는 비디오 데이터만으로도 강력한 시각적 세계 모델(Visual World Model)을 구축할 수 있음을 증명하며, 향후 비디오 생성 및 자율 주행, 로보틱스 등의 분야에 중요한 기초 모델이 될 가능성이 크다.
