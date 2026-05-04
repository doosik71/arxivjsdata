# A Survey for Foundation Models in Autonomous Driving

Haoxiang Gao, Zhongruo Wang, Yaqian Li, Kaiwen Long, Ming Yang, Yiqing Shen (2024)

## 🧩 Problem to Solve

자율주행(Autonomous Driving, AD) 시스템은 전통적으로 인지(Perception), 예측(Prediction), 계획(Planning)의 세 가지 핵심 단계로 구성된다. 기존의 딥러닝 기반 AD 모델들은 주로 수동으로 레이블링 된 데이터셋을 이용한 지도 학습(Supervised Learning)에 의존해 왔으나, 이는 데이터의 다양성 부족으로 인해 일반화 능력이 떨어지는 한계가 있다. 또한, 계획 단계에서는 복잡한 휴리스틱 규칙 기반 시스템(Heuristic rule-based systems)을 사용하는데, 이는 수많은 코너 케이스(Corner cases)를 처리하기 위해 막대한 엔지니어링 노력과 디버깅 시간이 소요된다는 문제가 있다.

본 논문의 목표는 최근 자연어 처리(NLP)와 컴퓨터 비전(CV) 분야에서 혁신을 일으킨 Foundation Models(기초 모델)를 자율주행 분야에 어떻게 적용할 수 있는지 체계적으로 분석하는 것이다. 특히 방대한 데이터로 사전 학습된 모델의 일반화 능력과 추론 능력을 활용하여, 자율주행의 안전성, 설명 가능성, 그리고 유연성을 높이는 방안을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 자율주행을 위한 Foundation Models의 적용 현황을 분석하고 이를 체계화한 Taxonomy(분류 체계)를 제안한 점이다. 주요 기여 사항은 다음과 같다.

1. **포괄적인 문헌 조사**: 40편 이상의 최신 연구 논문을 분석하여 Large Language Models(LLM), Vision Foundation Models(VFM), Multi-modal Foundation Models가 자율주행의 각 단계(인지, 예측, 계획, 시뮬레이션)에 어떻게 기여하는지 상세히 검토하였다.
2. **분류 체계(Taxonomy) 제안**: Foundation Models를 모달리티(Modality)와 자율주행 내 기능(Function)에 따라 분류하여, 어떤 모델이 어떤 작업에 적합한지에 대한 구조적인 프레임워크를 제공하였다.
3. **기술적 방법론 분석**: Prompt Engineering, In-context Learning, Fine-tuning, RLHF 등 Foundation Models를 자율주행 도메인에 적응시키기 위한 구체적인 기법들을 분석하였다.
4. **미래 연구 로드맵 제시**: 현재 모델들이 가진 한계(환각 현상, 지연 시간, 데이터 갭 등)를 정의하고, 이를 해결하기 위한 단계적인 연구 방향(도메인 특화 사전 학습 $\rightarrow$ 3D 적응 $\rightarrow$ 인간 피드백 정렬)을 제안하였다.

## 📎 Related Works

본 논문은 기존의 자율주행 관련 기초 모델 서베이 논문들과 차별점을 둔다. 기존의 LLM4Drive와 같은 연구들은 주로 LLM의 적용에 집중하였으며, 일부 연구들은 시뮬레이션, 데이터 어노테이션, 계획 단계에 국한된 요약을 제공하였다.

반면, 본 논문은 LLM뿐만 아니라 Vision Foundation Models와 Multi-modal Foundation Models까지 범위를 확장하였다. 특히 기존 연구들이 간과했던 예측(Prediction)과 인지(Perception) 작업에서의 기초 모델 활용 가능성을 심층적으로 다루며, 기술적인 세부 사항(사전 학습 모델 및 방법론)과 향후 연구 기회를 더 구체적으로 명시하였다.

## 🛠️ Methodology

본 논문은 Foundation Models를 세 가지 주요 모달리티로 나누어 자율주행에 적용하는 방법론을 설명한다.

### 1. Large Language Models (LLM)

LLM은 주로 텍스트 기반의 추론과 계획 단계에 활용된다.

- **Reasoning and Planning**: 차량의 상태, 주변 에이전트의 좌표, 속도, 지도 정보 등을 텍스트 설명으로 변환하여 LLM에 입력한다. LLM은 사전 학습된 상식(Common-sense)을 바탕으로 주행 결정을 내리며, 결정의 근거를 함께 제시함으로써 설명 가능성(Explainability)을 높인다.
- **Prediction**: 장면의 텍스트 표현을 사용하여 도로 권한(Right-of-ways)이나 보행자의 제스처와 같은 시맨틱 정보를 활용해 미래 궤적을 예측한다.
- **Simulation**: 사고 보고서 등의 텍스트에서 핵심 정보를 추출하여 시뮬레이션을 위한 시나리오 코드나 도메인 특화 언어(DSL)로 변환한다.

### 2. Vision Foundation Models (VFM)

VFM은 주로 고수준의 시각적 이해와 장면 생성에 활용된다.

- **Segment Anything Model (SAM)**: 2D 이미지 세그멘테이션 능력을 3D 객체 검출(SAM3D)이나 도메인 적응형 3D 세그멘테이션에 활용한다.
- **Diffusion Models**: 가우시안 노이즈에서 이미지를 복원하는 역확산 과정(Reverse diffusion process)을 통해 현실적인 주행 영상을 생성한다. 특히 Latent Diffusion Models(LDM)는 다음과 같은 구조를 가진다.
  - **Autoencoder**: 이미지를 저차원 잠재 공간(Latent space)으로 압축/복원한다.
  - **Diffusion Model**: 잠재 공간 상에서 데이터 분포를 학습하여 새로운 샘플을 생성한다.
- **World Models**: 현재 상태와 제어 신호를 입력받아 다음 프레임을 예측하는 방식으로, GAIA-1과 같은 모델은 DINO 임베딩과 코사인 유사도 손실 함수를 사용하여 시맨틱 지식을 증류(Distill)한다.

### 3. Multi-modal Foundation Models

여러 모달리티를 통합하여 복잡한 공간 추론과 인지를 수행한다.

- **Contrastive Learning (CLIP)**: 이미지 인코더와 텍스트 인코더의 임베딩 간 코사인 유사도를 최대화하는 방식으로 학습하여 Zero-shot 전이 능력을 갖춘다.
- **Visual Reasoning**: HiLM-D와 같이 고해상도 인지 브랜치와 저해상도 추론 브랜치를 결합하여 위험 객체를 식별하고 제안을 생성한다.
- **Unified Perception and Planning**: DriveGPT4와 같이 CLIP 인코더와 LLM을 결합하고 자율주행 특화 지시어 데이터셋으로 Fine-tuning 하여, 주변 환경 이해부터 제어 명령 생성까지 End-to-End로 수행한다.

## 📊 Results

본 논문은 개별 연구들의 결과를 종합하여 다음과 같은 정량적, 정성적 분석을 제시한다.

1. **LLM의 성능과 한계**: LLM 기반 드라이버(GPT-Driver)는 추론 능력이 뛰어나지만, 기존 방법론보다 높은 $0.44\%$의 충돌률을 보였다. 이는 LLM의 환각(Hallucination) 현상이 안전 중심의 자율주행에서 치명적일 수 있음을 시사한다.
2. **VFM의 3D 적용**: SAM을 3D 객체 검출에 적용한 결과, Waymo Open Dataset에서 기존 SOTA 모델들에 비해 낮은 성능(Average Precision)을 보였다. 이는 SAM이 웹 데이터로 학습되어 LiDAR의 희소하고 노이즈가 많은 포인트 클라우드 데이터를 처리하는 데 한계가 있기 때문이다.
3. **영상 생성 능력**: GAIA-1 및 DriveDreamer와 같은 World Model들은 텍스트 설명과 제어 신호에 따라 물리적으로 일관성 있고 현실적인 주행 시나리오를 생성할 수 있음을 입증하였다.
4. **멀티모달 모델의 강점**: GPT-4V와 같은 모델은 일반적인 객체 검출 모델이 놓치기 쉬운 '특수 차량'이나 '경찰관의 수신호'와 같은 롱테일(Long-tail) 시나리오에서 상식에 기반한 뛰어난 이해도를 보였다.

## 🧠 Insights & Discussion

### 강점

Foundation Models는 방대한 사전 학습 데이터를 통해 인간과 유사한 상식적 추론 능력을 제공한다. 특히 롱테일 시나리오에 대한 대응 능력이 뛰어나며, 결정 과정을 텍스트로 출력함으로써 자율주행 시스템의 블랙박스 문제를 해결하고 설명 가능성을 부여할 수 있다.

### 한계 및 비판적 해석

1. **안전성 및 신뢰성**: LLM의 환각 현상은 안전이 최우선인 자율주행에서 심각한 리스크이다. 단순히 텍스트 응답을 파싱하는 방식으로는 실시간 안전을 보장할 수 없다.
2. **실시간성(Latency)**: 수십억 개의 파라미터를 가진 모델은 추론에 수 초가 소요되며, 이는 차량 내 제한된 컴퓨팅 자원에서 실시간 제어를 수행하기에 부적합하다.
3. **인지 시스템 의존성**: LLM의 추론 능력은 뛰어나지만, 입력으로 들어오는 환경 설명(Environmental description)이 상위 인지 모듈의 오류에 민감하게 반응한다. 즉, 인지 단계의 작은 에러가 치명적인 계획 오류로 이어지는 전파 문제가 존재한다.
4. **Sim-to-Real Gap**: 대부분의 연구가 시뮬레이션 환경에서 진행되었으며, 실제 도로의 복잡성과 다양성을 완전히 반영하지 못하고 있다.

## 📌 TL;DR

본 논문은 LLM, VFM, Multi-modal Foundation Models를 자율주행의 인지, 예측, 계획, 시뮬레이션 단계에 적용하는 방법론과 현황을 체계적으로 정리한 서베이 논문이다. Foundation Models는 특히 설명 가능한 계획 수립과 롱테일 시나리오 대응에서 강력한 잠재력을 보이지만, 환각 현상, 높은 지연 시간, 3D 데이터 적응 문제라는 명확한 한계를 가지고 있다. 향후 연구는 도메인 특화 데이터셋 구축, 3D 센서 융합을 위한 모델 적응, 그리고 인간 피드백을 통한 안전성 정렬(Alignment) 방향으로 나아가야 하며, 이는 완전 자율주행 구현을 위한 핵심적인 경로가 될 것이다.
