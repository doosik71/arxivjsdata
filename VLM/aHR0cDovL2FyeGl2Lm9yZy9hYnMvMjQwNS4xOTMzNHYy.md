# LLMs Meet Multimodal Generation and Editing: A Survey

Yingqing He et al. (2024)

## 🧩 Problem to Solve

본 논문은 최근 급격히 발전한 대규모 언어 모델(Large Language Models, LLMs)을 다양한 멀티모달 생성 및 편집 작업에 결합하려는 시도를 체계적으로 분석한다. 기존의 멀티모달 대규모 언어 모델(MLLMs)에 관한 서베이들은 주로 멀티모달 '이해(Understanding)' 능력에 초점을 맞추어 왔으며, 생성 및 편집 분야에 대한 포괄적인 분석은 부족한 상태였다.

특히, OpenAI의 Sora와 같은 모델이 '세계 시뮬레이터(World Simulator)'로서의 가능성을 보여주었으나, 여전히 텍스트, 3D, 오디오 등 다양한 모달리티를 동시에 인지하고 생성하는 통합적인 능력은 부족하다. 따라서 본 연구의 목표는 이미지, 비디오, 3D, 오디오 등 다양한 도메인에서 LLM이 어떻게 생성 품질을 높이고, 제어 가능성을 향상시키며, 통합된 시스템을 구축하는지에 대한 기술적 방법론을 정리하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM이 멀티모달 생성 과정에서 수행하는 역할을 구체적으로 정의하고, 이를 기반으로 최신 연구들을 분류했다는 점이다. 중심적인 설계 아이디어는 LLM을 단순한 텍스트 처리기가 아니라, 다음과 같은 다양한 역할의 핵심 구성 요소로 활용하는 것이다.

- **Backbone Architecture**: 오디오나 이미지 토큰을 직접 처리하는 통합 신경망으로 활용.
- **Planner**: 생성될 콘텐츠의 레이아웃이나 시퀀스를 설계하는 계획자로 활용.
- **Instruction Processor**: 사용자의 복잡한 요구사항을 생성 모델이 이해할 수 있는 상세 프롬프트로 변환.
- **Semantic Guidance Provider**: 생성 모델에 고차원적인 의미론적 가이드를 제공.
- **Evaluator & Labeller**: 생성된 결과물의 품질을 평가하거나, 학습 데이터에 고품질의 캡션을 생성.

또한, LLM 도입 이전(Pre-LLM)과 이후(Post-LLM)의 생성 패러다임 변화를 비교 분석하여, LLM이 어떻게 AIGC(AI Generated Content)의 인터랙티브한 특성과 정밀한 제어 능력을 강화했는지 제시한다.

## 📎 Related Works

본 논문은 기존 관련 연구를 크게 두 가지 방향으로 구분하여 그 한계를 지적한다.

1. **모달리티별 생성 연구 (Modality-specific Generation)**: 이미지, 비디오, 3D, 오디오 각각의 생성 모델에 집중한 연구들이다. 이들은 주로 CLIP나 T5와 같은 인코더를 사용하여 텍스트-이미지 정렬을 구현했으나, LLM과 같은 강력한 추론 능력을 갖춘 모델의 통합 활용은 미흡했다.
2. **MLLM 이해 중심 연구 (Understanding-focused MLLMs)**: LLM에 시각적/청각적 인코더를 결합하여 질문-답변(VQA)이나 캡셔닝 등을 수행하는 연구들이다. 이들은 이해 능력에는 탁월하지만, 고품질의 콘텐츠를 생성하거나 정밀하게 편집하는 생성적 측면의 분석은 부족했다.

본 논문은 이러한 한계를 극복하기 위해 '생성(Generation)'과 '편집(Editing)'에 특화된 LLM 활용 방안을 집중적으로 다루며 차별성을 둔다.

## 🛠️ Methodology

본 논문은 멀티모달 생성을 위한 이론적 배경과 LLM의 통합 방법론을 상세히 설명한다.

### 1. 생성 모델의 기초 원리

논문은 다섯 가지 주요 생성 모델의 수학적/구조적 원리를 설명한다.

- **GAN (Generative Adversarial Networks)**: 생성자($G$)와 판별자($D$)가 서로 경쟁하는 minimax 게임 구조이다.
  $$\min_{G} \max_{D} V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]$$
- **VAE (Variational Auto-Encoder)**: 인코더를 통해 잠재 공간 $z$를 학습하고 디코더로 복원한다. 목표는 ELBO(Evidence Lower Bound)를 최대화하는 것이다.
  $$L(\theta; X) = - [KL(q(z|x; \theta) || p(z)) - \mathbb{E}_{q(z|x; \theta)} [\log p(x|z; \theta)]]$$
- **Flow-based Model**: 가역(Invertible) 변환을 통해 단순한 분포를 복잡한 데이터 분포로 매핑하며, 로그 가능도(Log-likelihood)를 직접 최대화한다.
  $$L(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}(x)} [\log p_{\text{model}}(x; \theta)]$$
- **Diffusion Model**: 데이터에 노이즈를 점진적으로 추가하는 Forward 과정과, 이를 다시 제거하는 Reverse(Denoising) 과정을 학습한다.
  - Forward: $x_{t+1} = \sqrt{1-\alpha_t^2}x_t + \alpha_t \zeta_t$ (Gaussian noise $\zeta_t$ 추가)
  - Reverse: $\hat{\zeta}_t = D_\theta(x_t, t)$ (노이즈 $\zeta_t$를 예측하여 제거)
- **Autoregressive Model**: 이전 토큰들을 조건으로 다음 토큰을 예측하는 방식으로, 음성 및 텍스트 생성의 핵심이다.

### 2. LLM의 모달리티별 통합 방법론

#### (1) 이미지 및 비디오 생성

- **Layout Planning**: LLM이 객체의 좌표, 수량, 크기를 포함한 레이아웃(Bounding Box)을 먼저 설계하고, 이를 GLIGEN과 같은 모델에 전달하여 정밀한 배치를 구현한다.
- **Prompt Synthesis**: LLM이 사용자의 짧은 프롬프트를 매우 상세한 묘사로 확장하여 생성 모델의 품질을 높인다.
- **MLLM Integrated**: 시각적 토큰(voken)을 도입하여 LLM이 텍스트와 이미지를 동시에 생성/이해하는 통합 구조를 구축한다.

#### (2) 3D 생성

- **CLIP/T5-based**: SDS(Score Distillation Sampling) 손실 함수를 통해 2D 확산 모델의 지식을 3D 표현(NeRF, Mesh)으로 증류(Distill)한다.
- **LLM-based**: LLM이 Blender 코드나 3D 파라미터를 직접 생성하여 절차적 모델링(Procedural Modeling)을 수행한다.

#### (3) 오디오 생성

- **LLM as Backbone**: 오디오를 이산적 토큰(Discrete Tokens)으로 변환하여 LLM이 언어 모델링 방식으로 오디오를 생성하게 한다.
- **LLM as Conditioner**: LLM이 텍스트 프롬프트를 고차원 임베딩으로 변환하여 오디오 확산 모델의 조건부 입력으로 제공한다.

#### (4) Tool-augmented Multimodal Agents

LLM을 컨트롤러로 사용하여 외부 전문 도구(Expert Tools)를 호출하는 파이프라인이다.

- **Task Planning $\to$ Task Execution $\to$ Response Generation** 순으로 진행되며, LLM은 ReAct나 Chain-of-Thought 기법을 통해 어떤 도구를 어떤 순서로 사용할지 결정한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 수많은 최신 연구들을 체계적으로 분류한 분석 결과를 제시한다.

- **데이터셋 분석**: LAION-5B(이미지), WebVid-10M(비디오), Objaverse(3D), AudioSet(오디오) 등 각 모달리티별 핵심 데이터셋의 특성과 규모를 정리하였다.
- **성능 향상 지점**: LLM을 도입함으로써 기존 CLIP 기반 모델들이 해결하지 못했던 **텍스트 렌더링(이미지 내 글자 쓰기), 복잡한 공간 관계 묘사, 일관성 있는 긴 시퀀스 생성** 능력이 유의미하게 향상되었음을 확인하였다.
- **에이전트 효율성**: Training-free 방법(Prompt Engineering 기반)보다 Instruction-tuning 방법(LoRA 등을 이용한 미세 조정)이 도구 호출의 정확도와 복잡한 작업 수행 능력이 더 높음을 분석하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 파편화되어 있던 멀티모달 생성 연구들을 'LLM의 역할'이라는 관점에서 통합하여 정리함으로써, 향후 연구자들이 어떤 방향으로 모델을 설계해야 할지에 대한 기술적 로드맵을 제공한다. 특히, 단순한 생성 모델의 나열이 아니라 세계 모델(World Model)로 나아가기 위한 필수 구성 요소들을 짚어낸 점이 훌륭하다.

### 한계 및 미해결 과제

- **고해상도 및 장기 일관성**: 여전히 비디오나 오디오에서 매우 긴 시퀀스를 생성할 때 일관성을 유지하는 것이 어렵다.
- **계산 비용**: LLM과 대형 생성 모델을 결합할 때 발생하는 막대한 연산 비용과 추론 속도 문제는 실시간 서비스 적용의 걸림돌이다.
- **안전성(Safety)**: Deepfake와 같은 오남용 문제, 저작권 침해 문제에 대한 방어 기제(Watermarking 등)가 생성 속도를 따라가지 못하고 있다.

### 비판적 해석

논문은 LLM의 능력을 높게 평가하고 있으나, 실제로는 LLM이 생성한 '상세 프롬프트'가 단순히 생성 모델의 내부 분포를 더 잘 자극하는 것인지, 아니면 실제 '추론'을 통해 구조적 개선을 이룬 것인지에 대한 깊은 분석은 부족하다. 또한, Unified Training(통합 학습)의 가능성을 언급했지만, 각 모달리티의 통계적 특성 차이로 인한 상충(Interference) 문제를 해결할 구체적인 방법론 제시는 미흡하다.

## 📌 TL;DR

이 논문은 LLM을 이미지, 비디오, 3D, 오디오 생성 및 편집에 통합하는 최신 기술들을 집대성한 서베이 보고서이다. LLM을 **계획자(Planner), 가이드(Guidance), 백본(Backbone), 평가자(Evaluator)** 등으로 정의하여 분석하였으며, 이를 통해 단순한 콘텐츠 생성을 넘어 물리 세계를 시뮬레이션하는 **'세계 모델(World Model)'**로 진화하기 위한 기술적 방향성을 제시한다. 향후 AIGC 분야에서 정밀한 제어 가능성과 멀티모달 통합 생성 능력을 확보하는 데 핵심적인 지침서가 될 것으로 보인다.
