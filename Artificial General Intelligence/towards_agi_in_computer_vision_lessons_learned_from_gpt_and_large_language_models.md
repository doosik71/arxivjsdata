# Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models

Lingxi Xie, Longhui Wei, Xiaopeng Zhang, Kaifeng Bi, Xiaotao Gu, Jianlong Chang, and Qi Tian (2023)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(Computer Vision, CV) 분야에서 인공 일반 지능(Artificial General Intelligence, AGI)을 달성하기 위한 경로를 탐색하는 것을 목표로 한다. 자연어 처리(NLP) 분야에서는 대규모 언어 모델(Large Language Models, LLMs)과 GPT와 같은 시스템을 통해 다양한 작업을 통합하여 해결하는 AGI의 초기 징후가 나타나고 있다. 반면, CV 분야에서는 여전히 이미지 분류, 객체 탐지, 세그멘테이션 등 개별 작업마다 서로 다른 네트워크 구조나 파이프라인을 사용하는 파편화된 상태에 머물러 있다.

저자들은 CV 분야에서 통합(Unification)이 어려운 근본적인 이유가 '환경으로부터 학습하는 패러다임'의 부재에 있다고 진단한다. 기존 CV 연구들은 실제 세계를 시뮬레이션하는 대신, 데이터셋이라는 표본을 통해 성능을 높이려는 '대리 작업(Proxy Tasks)'에 매몰되어 있으며, 이는 결국 AGI라는 궁극적인 목표에서 멀어지게 만드는 결과를 초래한다고 주장한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 NLP의 성공 사례인 GPT의 메커니즘을 CV에 이식하여 AGI로 나아가기 위한 개념적 파이프라인을 제안한 것이다. 중심 아이디어는 CV 알고리즘을 단순한 인식기가 아닌, 상호작용 가능한 환경 속의 '에이전트(Agent)'로 정의하는 것이다.

구체적으로는 상호작용 가능한 고품질 환경을 구축하고, 에이전트가 자신의 행동에 따른 미래 프레임을 예측하도록 사전 학습(Pre-training)시킨 뒤, 인간의 지시(Instruction)를 통해 다양한 작업을 수행하도록 미세 조정(Fine-tuning)하는 3단계 전략을 제시한다.

## 📎 Related Works

논문은 CV의 통합을 위한 최근의 시도들을 다섯 가지 범주로 분류하여 설명한다.

1. **Open-world Visual Recognition**: CLIP과 같은 Vision-Language Alignment 모델을 통해 학습 데이터에 없는 개념도 인식하려는 시도이다. 하지만 이는 주로 사전 학습된 모델의 지식에 의존하는 것이며, 복잡한 장면 내의 세부 의미를 지칭하는 데 한계가 있다.
2. **Segment Anything (SAM)**: 다양한 프롬프트를 통해 이미지 픽셀을 그룹화하는 일반화된 모듈을 제안하였다. 많은 작업에 전이 가능하지만, 색상과 같은 픽셀 레벨의 외형에 과적합되어 분류 능력이 약화될 수 있다는 한계가 있다.
3. **Generalized Visual Encoding**: $\text{pix2seq}$나 $\text{Painter}$와 같이 다양한 CV 작업을 단일한 출력 형태(예: 토큰 시퀀스 또는 컬러 패치)로 통일하려는 시도이다. $\text{Gato}$와 같은 일반 목적 에이전트가 이에 해당한다.
4. **LLM-guided Visual Understanding**: $\text{ViperGPT}$나 $\text{HuggingGPT}$처럼 LLM이 논리적 제어기 역할을 하여 CV 모듈을 호출하는 방식이다. 논리적 추론은 가능하지만, 기초 인식 모듈의 결과가 틀리면 최종 답변도 틀리는 의존성 문제가 존재한다.
5. **Multimodal Dialog**: $\text{LLaVA}$나 $\text{MiniGPT-4}$와 같이 Vision-Language 모델을 Instruction Tuning하여 대화형 시스템으로 확장한 연구들이다.

저자들은 이러한 연구들이 통합을 향한 진전이지만, 여전히 실제 환경에서의 상호작용을 통한 일반화라는 AGI의 핵심에는 도달하지 못했다고 분석한다.

## 🛠️ Methodology

본 논문은 AGI를 '환경 내에서 보상을 최대화하는 알고리즘'으로 정의하며, 이를 수학적으로 다음과 같이 정식화한다. 에이전트가 상태 시퀀스 $S=\{s_1, \dots, s_T\}$를 관찰하고 행동 집합 $A=\{a_1, \dots, a_M\}$ 중에서 행동을 선택할 때, 목표는 기대 누적 보상(Expected Cumulative Reward) $R$을 최대화하는 정책 $\pi: S \to A$를 학습하는 것이다.

$$R = \sum_{t=1}^{T} r(s_t, a_t)$$

이 관점에서 제안하는 AGI 달성을 위한 가상 파이프라인(Imaginary Pipeline)은 다음과 같은 3단계로 구성된다.

**Stage 0: 환경 구축 (Establishing Environments)**
다양성(Abundance), 충실도(Fidelity), 상호작용 가능성(Interactability)을 갖춘 고품질의 가상 환경을 구축한다. 이는 에이전트가 학습할 수 있는 '세계'를 만드는 단계이다.

**Stage 1: 생성적 사전 학습 (Generative Pre-training)**
에이전트가 환경을 탐색하며 자신의 행동에 따른 미래 프레임을 예측하도록 학습한다. 이는 NLP의 다음 토큰 예측(Next Token Prediction)과 유사하지만, CV에서는 미래 상태가 에이전트의 행동($a_t$)에 따라 달라지므로 상태와 행동의 결합 분포를 학습하게 된다.

**Stage 2: 지시 미세 조정 (Instruct Fine-tuning)**
사전 학습된 모델에 인간의 지시를 입력하여 실제 세계의 구체적인 작업(예: "커피 한 잔 사 와")을 수행하도록 학습시킨다. 이 과정에서 탐색, 내비게이션, 언어 소통, 물리적 조작 등이 통합적으로 학습된다.

## 📊 Results

본 논문은 특정 알고리즘의 성능을 측정하는 실험 논문이 아니라, 향후 연구 방향을 제시하는 관점(Perspective) 논문이다. 따라서 저자가 직접 수행한 정량적 실험 결과는 제시되지 않았다.

대신, GPT-4의 사례를 통해 텍스트 환경에서의 AGI 가능성을 분석하고, SAM이나 CLIP 등의 기존 모델들이 가진 통합의 한계를 정성적으로 논의한다. 또한, $\text{Habitat}$이나 $\text{ProcTHOR}$와 같은 기존 3D 환경들이 현실 세계의 복잡성을 담아내기에 아직 부족하며, 데이터 양을 늘려도 성능이 빠르게 포화되는 현상이 있음을 지적하며 환경 구축의 중요성을 강조한다.

## 🧠 Insights & Discussion

저자들은 **"Proxy is Dying!"**이라는 강렬한 메시지를 통해, ImageNet과 같은 고정된 데이터셋에서 정확도를 0.5% 높이려는 기존의 연구 방식이 더 이상 AGI로 가는 길이 아니라고 주장한다. 딥러닝의 발전으로 대리 작업(Proxy Tasks)의 성능은 이미 매우 높아졌으며, 이제는 데이터 샘플링이 아닌 세계 시뮬레이션(World Simulation)으로 패러다임을 전환해야 한다는 것이다.

**강점 및 제안:**

- CV를 NLP의 하위 집합으로 보고, '환경-에이전트-보상'의 프레임워크를 도입하여 CV AGI의 명확한 경로를 제시하였다.
- 단순한 인식 모델을 넘어 Embodied AI의 관점에서 통합을 논의하였다.

**한계 및 미해결 질문:**

- 제안한 파이프라인을 실제로 구현하기 위해서는 엄청난 규모의 컴퓨팅 자원과 정교한 시뮬레이터가 필요하며, 가상 환경과 실제 환경 사이의 도메인 간극(Domain Gap)을 어떻게 해결할 것인지에 대한 구체적인 방법론은 명시되지 않았다.
- 시각 데이터의 엄청난 중복성(Redundancy)을 해결하기 위한 데이터 압축(Data Compression)의 중요성을 언급했으나, 이에 대한 구체적인 아키텍처는 제안되지 않았다.

## 📌 TL;DR

본 논문은 컴퓨터 비전(CV) 분야가 AGI로 나아가기 위해서는 개별 작업의 정확도를 높이는 '대리 작업(Proxy Tasks)' 중심의 연구에서 벗어나, 상호작용 가능한 '환경(Environments)'에서의 학습 패러다임으로 전환해야 한다고 주장한다. 이를 위해 **환경 구축 $\rightarrow$ 미래 프레임 예측 사전 학습 $\rightarrow$ 지시 미세 조정**으로 이어지는 3단계 파이프라인을 제안하며, 이는 CV를 단순한 이미지 분석 도구가 아닌 세계를 이해하고 상호작용하는 에이전트로 진화시키는 방향성을 제시한다.
