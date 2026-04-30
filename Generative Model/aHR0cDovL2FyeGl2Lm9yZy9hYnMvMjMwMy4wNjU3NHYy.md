# Diffusion Models for Non-autoregressive Text Generation: A Survey

Yifan Li, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen (2023)

## 🧩 Problem to Solve

본 논문은 자연어 처리 분야의 Non-autoregressive (NAR) 텍스트 생성 모델이 가진 근본적인 한계를 해결하기 위해 등장한 Diffusion 모델들의 연구 동향을 분석한다. 일반적으로 텍스트 생성은 토큰을 순차적으로 생성하는 Autoregressive (AR) 방식을 사용하며, 이는 토큰 간의 의존성을 잘 포착하지만 생성 속도가 매우 느리다는 단점이 있다. 이를 해결하기 위해 모든 토큰을 병렬로 생성하는 NAR 방식이 제안되었으나, 토큰 간의 복잡한 의존 관계를 충분히 학습하지 못해 생성 정확도가 AR 모델에 비해 크게 떨어진다는 문제점이 존재한다.

따라서 본 연구의 목표는 최근 이미지 및 오디오 생성에서 뛰어난 성능을 보인 Diffusion 모델을 NAR 텍스트 생성에 도입하여, 추론 속도의 이점을 유지하면서도 생성 품질을 AR 수준으로 끌어올리기 위한 방법론들을 체계적으로 정리하고 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 NAR 텍스트 생성을 위한 Diffusion 모델의 최신 연구들을 종합적으로 검토하고, 이를 체계적인 프레임워크로 분류하여 제시했다는 점이다. 특히 텍스트의 이산적(discrete) 특성으로 인해 발생하는 어려움을 극복하기 위해 제안된 Continuous Diffusion과 Discrete Diffusion의 두 가지 주류 접근 방식을 상세히 분석한다. 또한, Denoising Network, Noise Schedule, Objective Function, Conditioning Strategy라는 네 가지 핵심 설계 요소를 중심으로 각 모델의 기술적 차별점을 분석하여 연구자들에게 체계적인 참조 가이드를 제공한다.

## 📎 Related Works

기존의 NAR 생성 모델들은 성능 격차를 줄이기 위해 Knowledge Distillation이나 대규모 사전 학습(Large-scale pre-training)과 같은 기법들을 도입해 왔다. 하지만 이러한 방법들은 여전히 토큰 간의 의존성을 완벽하게 캡처하는 데 한계가 있었다. 한편, Diffusion 모델은 연속적인 데이터 생성에서 탁월한 성능을 입증했으나, 텍스트는 이산적인 토큰으로 구성되어 있어 표준적인 Gaussian noise를 직접 적용하기 어렵다는 특성이 있다. 본 논문은 이러한 기존 NAR 모델의 한계와 일반적인 Diffusion 모델의 적용 제약 사항을 지적하며, 이를 해결하기 위해 텍스트 전용으로 설계된 Diffusion 모델들의 필요성을 강조한다.

## 🛠️ Methodology

본 논문은 텍스트 Diffusion 모델을 크게 두 가지 경로로 구분하여 설명한다.

### 1. Discrete Text Diffusion Model
이산적 공간에서 직접 Diffusion 과정을 수행하는 방식으로, 주로 전이 행렬(Transition Matrix) $Q_t$를 사용하여 데이터를 오염시킨다.
- **Forward Process**: $q(x_t | x_{t-1}) = \text{Cat}(x_t; p = x_{t-1} Q_t)$와 같이 정의되며, $x_0$에서 $x_t$까지의 상태 변화를 Categorical 분포로 모델링한다.
- **특징**: D3PM과 같은 모델은 특정 토큰을 $[MASK]$ 토큰으로 변환하는 Absorbing state 방식을 사용하여, 추론 시 모든 토큰이 $[MASK]$인 상태에서 시작해 점진적으로 실제 토큰으로 복구한다.

### 2. Continuous Text Diffusion Model
이산 토큰을 연속적인 임베딩 공간으로 매핑한 후 Diffusion을 수행하는 방식이다.
- **Pipeline**: $\text{Tokens} \rightarrow \text{Embedding} \rightarrow \text{Diffusion Process} \rightarrow \text{Denoising} \rightarrow \text{Rounding (Mapping back to Tokens)}$의 과정을 거친다.
- **핵심 모델**: Diffusion-LM은 임베딩 단계에 가우시안 노이즈를 추가하며, 최종 단계에서 $\text{softmax}$ 함수를 이용한 Rounding 단계를 통해 다시 이산 토큰으로 변환한다. SSD-LM은 Simplex representation을 활용하여 거의 원-핫(almost-one-hot) 형태의 표현법을 도입함으로써 제어 능력을 높였다.

### 3. 핵심 설계 요소 (Key Designs)
- **Denoising Network**: 이미지 모델의 U-Net 대신 텍스트의 순차적 의존성을 포착하기 위해 Transformer 아키텍처를 주로 사용한다.
- **Noise Schedule**: $\beta_t$의 변화를 결정하는 스케줄로, Linear, Cosine, Sqrt, Spindle(정보량이 많은 토큰을 먼저 오염시킴) 등 다양한 방식이 제안되었다.
- **Objective Function**: 단순한 $\mu_t$ 예측보다는 원본 데이터 $x_0$를 직접 예측하는 $x_0$-parameterized loss가 수렴 성능이 더 좋다고 분석하며, 다음과 같은 손실 함수를 사용한다:
  $$L_{\text{simple}} = \sum_{t=1}^{T} E_q [ \| f_\theta(x_t, t) - x_0 \|^2 ]$$
- **Conditioning Strategy**: 무조건 생성(Unconditional), 속성 기반 생성(Attribute-to-text), 텍스트 기반 생성(Text-to-text)으로 구분된다. Classifier-guidance나 Classifier-free guidance를 통해 생성 과정에 외부 제어 조건을 주입한다.

## 📊 Results

본 논문은 서베이 논문으로서 개별 실험 결과보다는 기존 모델들의 특성을 비교 분석한 결과를 제시한다. Table 1을 통해 다양한 모델(D3PM, Diffusion-LM, SSD-LM 등)이 사용하는 Diffusion 공간, 노이즈 스케줄, 적용 작업(UCG, A2T, T2T) 및 PLM 활용 여부를 정량적으로 비교하고 있다.

분석 결과, Diffusion 모델은 기존 NAR 방식보다 다음과 같은 이점을 가진다고 명시한다:
1. **제약된 반복 정제 (Constrained Iterative Refinement)**: 사전 정의된 변동 폭 내에서 점진적으로 품질을 높여 AR과의 성능 격차를 줄인다.
2. **중간 제어 가능성 (Intermediate Control)**: 생성 중간 단계에 Classifier 등을 통해 복잡한 제어 조건(예: 구문 분석 트리)을 효과적으로 주입할 수 있다.
3. **속도와 품질의 트레이드-오프**: DDIM과 같은 가속 샘플링 기법을 통해 추론 시간을 조절하면서 품질 손실을 최소화할 수 있다.

## 🧠 Insights & Discussion

본 논문은 텍스트 Diffusion 모델이 가진 잠재력과 동시에 해결해야 할 과제들을 다음과 같이 논의한다.

**강점 및 가능성**: Diffusion 모델은 NAR의 고질적인 문제인 '토큰 간 의존성 부족'을 반복적인 정제 과정을 통해 해결할 수 있으며, 특히 제어 가능한 텍스트 생성(Controllable Text Generation)에서 매우 강력한 도구가 될 수 있음을 시사한다.

**한계 및 비판적 해석**:
- **PLM 활용의 효율성**: 많은 모델이 BERT나 BART 같은 PLM을 Denoising Network로 사용하지만, 정작 PLM의 사전 학습 목표(AR 또는 MLM)와 Diffusion의 학습 목표 간의 괴리가 있어 이를 완전히 활용하지 못하고 있다.
- **토큰별 특성 무시**: 현재 대부분의 Noise Schedule은 모든 토큰을 동일하게 처리한다. 하지만 실제 언어에서는 희귀 단어와 일반 단어의 정보량이 다르므로, 토큰의 중요도에 따른 맞춤형 스케줄링이 필요하다.

**향후 연구 방향**:
- 텍스트 특성에 맞춘 맞춤형 Noise Schedule 개발.
- PLM의 지식을 Diffusion 과정에 더 효율적으로 전이시키는 방법 연구.
- 텍스트와 이미지의 잠재 공간을 통합하는 Unified Multimodal Diffusion 모델로의 확장.
- 생성된 텍스트의 윤리적 가치 정렬(Human Value Alignment) 및 독성 제거(Detoxification).

## 📌 TL;DR

본 논문은 추론 속도가 빠른 Non-autoregressive (NAR) 텍스트 생성의 품질 저하 문제를 해결하기 위해 도입된 Diffusion 모델들을 체계적으로 정리한 서베이 논문이다. 텍스트의 이산성을 해결하기 위한 연속적/이산적 Diffusion 접근법과 핵심 설계 요소(네트워크, 스케줄, 손실 함수, 조건화)를 분석하였다. 이 연구는 향후 고품질·고속 텍스트 생성 모델 설계 및 멀티모달 통합 모델 연구에 중요한 이론적 토대를 제공할 것으로 기대된다.