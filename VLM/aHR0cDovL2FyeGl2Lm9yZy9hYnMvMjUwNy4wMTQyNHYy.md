# TriVLA: A Triple-System-Based Unified Vision-Language-Action Model for General Robot Control

Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, Yanwei Fu (2025)

## 🧩 Problem to Solve

본 논문은 일반적인 로봇 제어를 위해 시각-언어-행동(Vision-Language-Action, VLA) 모델이 해결해야 할 핵심적인 한계점을 다룬다. 기존의 VLA 모델들은 주로 대규모 사전 학습된 지식을 활용하는 Dual-system 아키텍처를 채택하고 있으나, 대부분 현재 시점의 한두 장의 이미지와 같은 정적인 정보(Static Information)에 의존하는 경향이 있다.

이러한 접근 방식은 로봇이 복잡하고 역동적인 환경에서 수행해야 하는 Embodied Task에서 필수적인 '동적 특성(Dynamic Aspects)'을 간과하게 만든다. 즉, 단순한 시각적 인식과 고수준의 추론만으로는 물리적 세계의 변화를 예측하고 정밀하게 제어하는 데 한계가 있으며, 이는 특히 장기적인 목표를 달성해야 하는 Long-horizon task에서 성능 저하로 이어진다. 따라서 본 논문의 목표는 고수준의 추론 능력과 물리적 동역학에 대한 예측 능력을 동시에 갖춘 통합 VLA 모델을 구축하여 일반적인 로봇 제어 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 인지 구조에서 영감을 얻어 VLM과 VDM을 결합한 **Triple-system Compositional Architecture**를 제안한 것이다.

1. **Triple-system 구조의 도입**: 기존의 Dual-system(VLM + Policy) 구조에 Video Diffusion Model(VDM) 기반의 동역학 인식 모듈을 추가하여, '세계 지식(World Knowledge)'과 '세계 모델(World Model)'을 동시에 활용한다.
2. **동적 예측 표현의 통합**: VDM을 통해 현재 상태뿐만 아니라 미래의 예상 궤적을 시각적 표현(Visual Representation) 형태로 생성하고, 이를 정책 학습에 제공함으로써 로봇이 물리적 동역학을 암시적으로 학습하게 한다.
3. **효율적인 추론 메커니즘**: VDM의 완전한 디노이징 과정 대신, 단 한 번의 forward pass만으로도 유효한 미래 궤적 정보를 추출하여 제어 빈도(Control Frequency)를 36Hz 수준으로 유지하면서도 예측 성능을 확보하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **VLA Models**: RT-2, OpenVLA 등은 VLM의 추론 능력을 활용해 로봇의 동작을 예측한다. 하지만 이들은 정적 이미지에 의존하여 동적인 환경 변화에 취약하다.
- **Future Prediction in Robotics**: UniPi, Susie 등은 미래의 키프레임을 예측하여 정책 학습에 활용한다. 그러나 대개 단일 스텝의 미래 예측에 그치거나, 전체 비디오를 디노이징하는 과정이 너무 느려 실시간 제어(Real-time control)에 부적합하며 Open-loop 제어 문제가 발생한다.

### TriVLA의 차별점

TriVLA는 고수준의 상식 추론(System 2)과 저수준의 물리적 동역학 예측(System 3)을 분리하여 처리하고, 이를 정책 모듈(System 1)에서 통합한다. 특히 VDM을 세계 모델로 사용하여 단순한 이미지 생성이 아닌, 정책 학습을 가이드하기 위한 '예측적 특성(Predictive Features)'을 추출한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

TriVLA는 그림 3에서 보여주듯 세 가지 시스템이 유기적으로 연결된 파이프라인을 가진다.

### 1. Vision-Language Module (System 2)

환경 해석과 작업 목표 이해를 담당한다. 사전 학습된 **Eagle-2 VLM**을 백본으로 사용하며, 시각적 입력과 언어 지시어를 처리한다.

- **특징 추출**: LLM의 최종 층이 아닌 12번째 레이어에서 임베딩을 추출하며, 이는 추론 속도를 높이고 하위 정책의 성공률을 개선한다.
- **출력**: 비전-언어 토큰 $Q_{vl}$을 생성하며, 로봇의 상태 정보(Robot State)는 별도의 MLP를 통해 상태 토큰 $Q_s$로 투영된다.

### 2. Dynamics Perception Module (System 3)

물리적 동역학 지식을 제공하는 세계 모델 역할을 수행한다. **Stable Video Diffusion (SVD)** 모델을 기반으로 하며, 인터넷의 인간/로봇 조작 데이터로 파인튜닝되었다.

- **학습 목표**: 노이즈가 섞인 샘플 $x_t$로부터 원본 비디오 시퀀스 $x_0$를 복원하는 diffusion 목적 함수를 사용한다.
$$L_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2$$
- **효율적 특징 추출**: 실시간성을 위해 전체 디노이징을 수행하지 않고, 단 한 번의 forward pass를 통해 얻은 초기 특징맵을 사용한다. 여러 up-sampling 레이어($m$)에서 나오는 특징맵 $L_m$을 보간(Interpolation)하고 연결(Concatenation)하여 최종 예측 시각 표현 $F_p$를 생성한다.
$$F_p = \text{concate}((L'_0, L'_1, \dots, L'_m), \text{dim}=1)$$

### 3. Policy Learning Module (System 1)

최종적인 모터 액션을 생성하는 제어기이다.

- **입력 통합**: System 2의 $Q_{vl}$과 System 3의 고차원 특징 $F_p$를 입력으로 받는다. $F_p$는 학습 가능한 토큰 $Q$와 Spatio-temporal Attention을 통해 압축된 예측 토큰 $Q_p$로 변환된다.
- **액션 생성**: **Diffusion Transformer (DiT)**를 기반으로 하며, Action Flow-matching 기법을 통해 액션 시퀀스를 생성한다.- **손실 함수**: 노이즈가 섞인 액션 $a_k$에서 원본 액션 $a_0$를 복원하는 디노이저 $D_\psi$를 학습한다.
$$L_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| A_d(D_\psi(a_k, Q_{vl}, Q_p)) - a_0 \|^2$$
- **Action Chunking**: 매 타임스텝마다 단일 액션이 아닌 10단계의 액션 시퀀스를 한 번에 예측하여 제어 효율을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋 및 벤치마크**: CALVIN (ABC$\rightarrow$D 시나리오), LIBERO (Spatial, Object, Goal, Long), MetaWorld (60개 작업).
- **비교 대상**: RT-1, Diffusion Policy, Robo-Flamingo, UniPi, GR-1, VPP 등.
- **측정 지표**: 작업 성공률(Success Rate), 평균 완료 길이(Avg. Length).

### 주요 결과

1. **CALVIN 벤치마크**: ABC 환경에서 학습하고 처음 보는 D 환경에서 테스트한 결과, TriVLA는 평균 완료 길이(Avg. Len)에서 기존 SOTA를 뛰어넘는 성능을 보였다. 특히 전체 데이터의 10%만 사용했을 때의 성능(3.46)이 타 모델의 전체 데이터 학습 결과보다 높게 나타나 매우 높은 데이터 효율성을 입증하였다.
2. **LIBERO 벤치마크**: 모든 태스크 세트(Spatial, Object, Goal, Long)에서 가장 높거나 경쟁력 있는 성공률을 기록하였다. (평균 87.0%)
3. **MetaWorld 벤치마크**: 난이도별(Easy, Middle, Hard) 분석에서 모두 강세를 보였으며, 특히 Hard 작업에서도 높은 성공률을 기록하여 정밀 제어 능력을 입증하였다.
4. **추론 속도**: NVIDIA H100 GPU 기준 약 34~36 Hz의 제어 빈도를 달성하여 실시간 로봇 제어가 가능함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

TriVLA의 가장 큰 강점은 **'상식적 추론'과 '물리적 예측'의 성공적인 결합**에 있다. System 2는 "사과를 집어서 선반에 놓으라"는 복잡한 지시어의 의도를 파악하고, System 3은 이를 위해 로봇 팔이 어떻게 움직여야 하는지에 대한 물리적 가이드를 제공한다. 이러한 구조 덕분에 모델은 학습 데이터에 없는 새로운 환경(Unseen environment)에서도 일반화 능력이 뛰어나며, Long-horizon task를 안정적으로 수행할 수 있다.

### 한계 및 비판적 해석

논문에서는 매우 긍정적인 결과가 제시되었으나, 몇 가지 고려사항이 있다.

- **의존성**: Eagle-2 VLM과 SVD와 같은 거대 사전 학습 모델에 크게 의존하고 있다. 이러한 기반 모델의 성능이 전체 시스템의 상한선(Upper bound)을 결정하게 된다.
- **단일 포워드 패스의 정당성**: VDM의 단 한 번의 forward pass가 제공하는 특징이 구체적으로 어떤 정보를 담고 있는지에 대한 정성적 분석이 더 필요하다. 비록 그림 5에서 대략적인 궤적을 보여주지만, 이것이 정밀한 조작(Precision manipulation)에 얼마나 기여하는지는 추가 검증이 필요해 보인다.

## 📌 TL;DR

TriVLA는 **VLM(추론) + VDM(동역학 예측) + Diffusion Policy(제어)**라는 Triple-system 아키텍처를 통해 로봇의 일반 제어 능력을 극대화한 모델이다. 정적 정보에만 의존하던 기존 VLA의 한계를 극복하여, 세계 모델을 통한 미래 상태 예측 정보를 정책 학습에 통합함으로써 Long-horizon task 수행 능력을 비약적으로 향상시켰다. 특히 높은 데이터 효율성과 36Hz의 실시간 제어 속도를 동시에 확보하여, 향후 범용 로봇 컨트롤러 설계에 중요한 이정표가 될 것으로 기대된다.
