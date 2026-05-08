# Human-Centric Foundation Models: Perception, Generation and Agentic Modeling

Shixiang Tang, Yizhou Wang, Lu Chen, Yuan Wang, Sida Peng, Dan Xu, Wanli Ouyang (2025)

## 🧩 Problem to Solve

본 논문은 인간의 외형, 정체성, 동작, 의도를 이해하고 이를 디지털 휴먼이나 휴머노이드 형태로 구현하기 위한 **Human-centric Foundation Models (HcFMs)**의 최신 연구 동향을 분석한다.

기존의 인간 중심 연구들은 특정 작업(Task-specific)에 특화된 파이프라인을 구축하는 방향으로 발전해 왔다. 예를 들어 2D 키포인트 검출, 신체 부위 세그멘테이션, 3D 메쉬 복원 등이 각각 개별적으로 연구되었다. 그러나 이러한 방식은 다음과 같은 한계점을 가진다.

1. **높은 비용**: 각 작업마다 별도의 네트워크 설계, 사전 학습(Pretraining), 파라미터 튜닝 및 데이터 어노테이션 작업이 필요하여 자원 소모가 매우 크다.
2. **전체론적 이해 부족**: 인간을 하나의 통합된 복잡한 시스템으로 파악하지 못하고 파편화된 정보로 처리함으로써, 정교하고 지능적인 디지털 휴먼을 생성하는 데 한계가 있다.

따라서 본 논문의 목표는 일반화(Generalization), 광범위한 적용 가능성(Broad applicability), 높은 충실도(High fidelity)를 갖춘 통합된 기반 모델(Foundation Model)의 필요성을 제기하고, 현재까지의 연구를 체계적으로 분류한 Taxonomy를 제시하여 향후 연구의 로드맵을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 빠르게 성장하는 인간 중심 기반 모델 분야를 체계적으로 정리하기 위해 다음과 같은 새로운 **Taxonomy(분류 체계)**를 제안한 것이다.

1. **Human-centric Perception Foundation Models**: 다중 모달(Multi-modal) 2D/3D 이해를 위해 세밀한 특징을 캡처하는 인식 모델.
2. **Human-centric AIGC Foundation Models**: 고충실도의 다양하고 사실적인 인간 관련 콘텐츠를 생성하는 생성 모델.
3. **Unified Perception and Generation Models**: 인식과 생성 능력을 통합하여 상호 보완적인 시너지를 내는 통합 모델.
4. **Human-centric Agentic Foundation Models**: 인식을 넘어 휴머노이드 로봇 등의 물리적 구현체(Embodiment)를 위해 인간 수준의 지능과 상호작용 행동을 학습하는 에이전트 모델.

## 📎 Related Works

본 연구는 대규모 언어 모델(LLM), 대규모 시각 모델(LVM), 그리고 텍스트-이미지 생성 모델과 같은 일반 목적의 기반 모델(Generalist Models)의 성공에서 영감을 받았다. 기존 연구들은 다음과 같은 흐름을 보였다.

- **인식 분야**: Vision Transformer(ViT) 등을 활용하여 인간의 신체 구조를 파악하려는 시도가 있었으나, 대부분 특정 데이터셋에 과적합된 경향이 있었다.
- **생성 분야**: Diffusion Model과 GAN을 통해 사실적인 인간 이미지를 생성했으나, 3D 일관성(3D Consistency)이나 정교한 제어(Control) 능력이 부족했다.
- **차별점**: 기존 연구들이 단일 작업의 성능 향상에 집중했다면, HcFMs는 여러 인간 중심 작업(Human-centric tasks)을 하나의 프레임워크로 통합하여 범용성을 확보하려는 패러다임의 전환을 꾀한다.

## 🛠️ Methodology

논문은 제안한 네 가지 카테고리별로 세부 방법론과 학습 프레임워크를 설명한다.

### 1. Human-centric Perception Foundation Models

인간 신체의 구조적 특성(Human Priors)을 활용하여 일반적인 표현(Representation)을 학습한다.

- **Unsupervised Learning**: 레이블 없이 학습하며, Contrastive Learning을 통해 다양한 모달리티(RGB, Depth, 2D Keypoints) 간의 특징을 정렬하거나, Mask Image Modeling을 통해 마스킹된 신체 부위를 복원하며 구조적 지식을 학습한다.
- **Multitask Supervised Learning**: 여러 작업의 감독 신호를 동시에 사용하여 인코더가 범용적인 특징을 학습하게 한다. 특히 **Unified Modeling** 방식은 Dynamic Query를 사용하는 인코더-디코더 구조를 통해 단일 모델에서 다양한 인식 작업을 수행한다.

### 2. Human-centric AIGC Foundation Models

- **Unsupervised (GAN-based)**:
  - **Style Modulation**: 잠재 공간(Latent Space)을 분리하여 스타일 벡터를 통해 외형을 제어한다.
  - **Neural Renderer**: 3D 표현과 뉴럴 렌더러를 결합하여 3D 기하학적 일관성을 강제한다.
- **Multi-modal Supervised (Diffusion-based)**:
  - **Conditional Latent Diffusion**: 텍스트, 포즈, 세그멘테이션 맵 등을 조건으로 입력받아 잠재 공간에서 디노이징을 수행한다.
  - **Spatial-Temporal Diffusion Transformers**: Transformer 구조를 도입하여 시간적 의존성과 공간적 구조를 동시에 캡처함으로써 고품질의 인간 비디오를 생성한다.

### 3. Unified Perception and Generation Foundation Models

인간 중심의 신호를 '외국어'처럼 취급하여 LLM/MLLM에 통합한다.

- **Fixed Vocabulary**: 기존 LLM의 어휘집을 유지하며, 프로젝션 레이어(Projection Layer)를 통해 인간 중심 신호를 LLM의 특징 공간으로 매핑하거나 외부 툴(Tool)을 호출한다.
- **Extended Vocabulary**: LLM의 임베딩 공간과 어휘집을 직접 확장하여 포즈 파라미터, SMPL 표현, 모션 시퀀스 등을 직접 토큰화하여 학습시킨다.

### 4. Human-centric Agentic Foundation Models

물리적 환경과의 상호작용을 위해 VLM을 제어 정책(Control Policy)과 결합한다.

- **VL-based Models**: 사전 학습된 VLM을 사용하여 고수준의 세만틱 이해를 수행하고, 이를 로봇의 센서모터 제어 정책과 정렬한다.
- **VLA-based Models (Vision-Language-Action)**: 로봇의 관찰값과 동작(Action) 자체를 LLM의 토큰으로 취급하여, 텍스트-이미지-동작을 하나의 시퀀스로 학습하고 직접 동작 토큰을 생성한다.

## 📊 Results

본 논문은 서베이 논문으로서 개별 실험 결과보다는 각 카테고리별 대표 모델들의 성과와 역할에 집중하여 설명한다.

- **인식 모델**: SOLIDER, Sapiens 등의 모델이 적은 데이터 환경(Low-data regimes)에서도 ImageNet 사전 학습 모델보다 인간 중심 작업에서 우수한 성능을 보임을 확인하였다.
- **생성 모델**: HumanSD, CosmicMan 등은 텍스트 및 포즈 제어를 통해 매우 높은 충실도의 이미지와 비디오를 생성하며, 특히 Human4DiT는 360도 일관성을 가진 4D 비디오 생성이 가능함을 보여주었다.
- **통합 모델**: MotionGPT, UniPose 등은 모션 이해와 생성을 하나의 언어 모델 프레임워크에서 처리함으로써 모션 캡셔닝, 모션 생성, 모션 편집 등 다양한 작업을 유연하게 수행할 수 있음을 증명하였다.
- **에이전트 모델**: Project GR00T와 같은 VLA 모델은 인터넷 규모의 데이터와 시뮬레이션 데이터를 통해 휴머노이드 로봇이 복잡한 물리적 작업을 수행할 수 있는 가능성을 제시하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 파편화되어 있던 인간 중심 연구들을 '기반 모델(Foundation Model)'이라는 관점에서 통합하여 체계적인 분류 체계를 세웠다. 특히 인식 $\rightarrow$ 생성 $\rightarrow$ 통합 $\rightarrow$ 에이전트로 이어지는 발전 단계는 향후 디지털 휴먼 및 휴머노이드 로봇 연구의 방향성을 명확히 제시한다.

### 한계 및 향후 과제

1. **데이터 수집의 민감성**: 인간 데이터는 개인정보 보호 및 윤리적 문제로 인해 일반 이미지/비디오 데이터보다 수집 비용이 높고 양이 제한적이다.
2. **전체론적 표현의 부재**: 현재 모델들은 얼굴, 손, 신체 중 일부에 치중되어 있으며, 이를 동시에 완벽하게 통합하여 캡처하는 단일 기반 모델은 아직 부족하다.
3. **상호작용성(Interactivity) 부족**: 대부분의 모델이 정적인 상태나 고립된 개체에 집중하고 있으며, 실제 환경에서의 복잡한 다중 에이전트 상호작용 및 맥락적 역동성을 모델링하는 데 한계가 있다.
4. **윤리적 이슈**: 딥페이크와 같은 오남용 가능성이 크므로, 개인 식별 정보를 제거하는 익명화 기술과 개인정보 보호 강화 기술(Privacy-enhancing technologies)의 도입이 필수적이다.

## 📌 TL;DR

본 논문은 인간의 인식, 생성, 그리고 행동 제어를 통합하는 **Human-centric Foundation Models (HcFMs)**의 최신 연구를 분석한 종합 서베이 보고서이다. 연구자들은 HcFMs를 **인식(Perception), 생성(AIGC), 통합(Unified), 에이전트(Agentic)**의 네 가지 범주로 분류하고, 각 분야의 핵심 방법론(Contrastive Learning, Diffusion, LLM Integration, VLA)을 상세히 분석하였다. 이 연구는 향후 지능형 디지털 휴먼과 고도화된 휴머노이드 로봇 구현을 위한 통합적인 기술적 토대를 제공하며, 데이터 확보와 윤리적 가이드라인 수립이 핵심 과제임을 시사한다.
