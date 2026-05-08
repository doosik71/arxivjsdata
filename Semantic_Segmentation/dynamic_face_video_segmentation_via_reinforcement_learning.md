# Dynamic Face Video Segmentation via Reinforcement Learning

Yujiang Wang, Mingzhi Dong, Jie Shen, Yang Wu, Shiyang Cheng, Maja Pantic (2021)

## 🧩 Problem to Solve

실시간 시맨틱 비디오 세그멘테이션(Semantic Video Segmentation)은 매우 높은 계산 비용을 요구하는 작업이다. 최근의 연구들은 모든 프레임을 처리하는 대신, 중요 프레임(Key frame)과 비중요 프레임(Non-key frame)을 구분하여 처리하는 다이내믹 프레임워크(Dynamic framework)를 사용함으로써 속도를 개선하고 있다.

그러나 기존의 Key scheduler(중요 프레임 결정기)들은 고정된 주기(Fixed policy)를 사용하거나, 단순히 두 프레임 간의 유사도나 편차를 계산하는 휴리스틱한 전략(Heuristic strategies)에 의존한다. 이러한 방식은 비디오 전체의 글로벌 컨텍스트(Global context)를 고려하지 못하므로, 결과적으로 전역적인 최적 성능(Global performance)을 달성하는 데 한계가 있다.

또한, 얼굴 비디오 세그멘테이션(Face video segmentation) 분야는 이미지 기반의 연구나 단순한 프레임별 처리 연구는 존재하지만, 다이내믹 가속 메커니즘을 적용한 실시간 시스템에 대한 연구는 거의 이루어지지 않았다. 따라서 본 논문의 목표는 심층 강화학습(Deep Reinforcement Learning, RL)을 통해 효율적이고 효과적인 Key scheduling 정책을 학습하여 실시간 얼굴 비디오 세그멘테이션 성능을 최적화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Key frame 결정 과정을 강화학습 문제로 모델링하여, 전역적인 보상(Global return)을 최대화하는 최적의 정책을 학습하는 것이다.

단순히 현재 프레임과 이전 Key frame 간의 차이만을 보는 것이 아니라, 과거의 결정 이력(Decision history)이라는 전문가 정보(Expert information)를 상태(State)에 포함함으로써 에이전트가 비디오의 흐름을 더 잘 이해하게 한다. 이를 통해 제한된 계산 자원(Key budget)을 가장 필요한 시점에 적절히 배분하여, 전역적인 세그멘테이션 정확도를 극대화하는 전략을 구축하였다.

## 📎 Related Works

**1. Semantic Image/Video Segmentation**
FCN 이후 Deeplab 시리즈와 같은 고성능 모델들이 등장했으나, 이들은 무거운 아키텍처로 인해 실시간 비디오 적용 시 지연 시간이 너무 길다는 문제가 있다. 이를 해결하기 위해 Deep Feature Flow(DFF)와 같은 다이내믹 프레임워크가 제안되었으며, Key frame의 특징(Feature)을 Optical flow를 이용해 비중요 프레임으로 전파(Warping)함으로써 연산량을 줄이는 방식이 사용되었다.

**2. Key Scheduler**
기존의 Key scheduler들은 주로 고정된 간격을 사용하거나(Fixed), DVSNet과 같이 두 프레임 간의 유사도 점수를 기반으로 한 적응형(Adaptive) 방식을 사용했다. 하지만 이러한 방식은 국소적인 변화만 감지할 뿐, 비디오 전체의 맥락을 파악하여 전략적으로 Key frame을 배치하는 능력이 부족하다.

**3. Semantic Face Segmentation**
얼굴 세그멘테이션은 주로 정적 이미지나 단순한 프레임별 처리(Per-frame basis)에 집중되어 있었으며, 비디오의 동역학(Dynamics)을 이용해 가속화를 시도한 연구는 본 논문이 처음이다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 Deep Feature Flow(DFF)를 기반으로 하며, 전체 파이프라인은 다음과 같다.

1. 이미지 세그멘테이션 모델 $N$을 무거운 특징 추출부 $N_{feat}$와 가벼운 작업 수행부 $N_{task}$로 나눈다.
2. **Key frame ($I_i$)**: $N_{feat}$와 $N_{task}$를 모두 통과하여 세그멘테이션 마스크 $y_i$를 생성한다.
3. **Non-key frame ($I_i$)**: $N_{feat}$를 생략하고, 마지막 Key frame($I_k$)의 특징 $f_k$를 Optical flow $M_{i \to k}$를 이용해 워핑(Warping)하여 특징 $f_i$를 얻은 후, $N_{task}$만 통과시켜 $y_i$를 생성한다.
4. **Key Scheduler**: 강화학습 기반의 정책 네트워크 $\pi_\theta$가 현재 프레임을 Key로 만들지(Action $a_1$) 아니면 Non-key로 유지할지(Action $a_0$)를 결정한다.

### 정책 네트워크 (Policy Network)

정책 네트워크는 하나의 Convolution layer와 4개의 Fully Connected(FC) layer로 구성된다.

- **입력 상태 ($s_i$)**: 두 가지 정보의 조합이다.
  - $D_{i \to k}$: FlowNet2-s 모델에서 추출한 현재 프레임과 마지막 Key frame 간의 편차 정보.
  - $E_i$: 전문가 정보인 $\text{KAR(Key All Ratio, Key frame 선택 비율)}$와 $\text{LKD(Last Key Distance, 마지막 Key frame 이후 경과 시간)}$.
- **출력**: $a_0$(Non-key)와 $a_1$(Key)에 대한 확률 $\pi_\theta(a_j|s_i)$.

### 학습 목표 및 손실 함수

에이전트는 누적 보상 $J$를 최대화하도록 학습된다.

- **보상 정의 ($r_i$)**: mIoU(mean Intersection-over-Union)를 지표로 사용한다.
$$r_i = \begin{cases} 0, & a_j = a_0 \\ U_i^{a_1} - U_i^{a_0}, & a_j = a_1 \end{cases}$$
여기서 $U_i^{a_1}$은 Key action을 취했을 때의 mIoU, $U_i^{a_0}$는 Non-key action을 취했을 때의 mIoU이다. 즉, Key frame으로 지정함으로써 얻는 성능 향상분이 클수록 더 큰 보상을 준다.

- **최종 목적 함수**: 누적 보상 $J(\theta)$에 과도한 확신을 방지하기 위한 엔트로피 손실(Entropy loss) $H$를 추가하여 최적화한다.
$$L = J(\theta) + \lambda \frac{1}{H(\pi_\theta(a|s))}$$

### Key selection 빈도 제약 (Constraint)

모든 프레임을 Key로 지정하면 보상이 최대화되므로, 이를 방지하기 위해 $\text{KAR}$ 제한치 $\eta$를 도입한다. 에피소드 도중 $\text{KAR}$가 $\eta$를 초과하면 즉시 에피소드를 종료(Stop immediately)시킨다. 이를 통해 에이전트는 한정된 Key budget 내에서 보상을 극대화할 수 있는 최적의 시점을 찾는 법을 배우게 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: 300VW(얼굴 비디오), Cityscapes(도시 도로 장면).
- **모델**: $N$은 Deeplab-V3+ (ResNet-50 backbone), $F$는 FlowNet2-s를 사용하였다.
- **비교 대상**: DVSNet, Flow magnitude 기반 적응형 스케줄러, 고정 주기 DFF.
- **지표**: $\text{AKI(Average Key Interval)}$ 대비 mIoU, $\text{FPS}$ 대비 mIoU.

### 주요 결과

- **성능 우위**: 300VW와 Cityscapes 데이터셋 모두에서 제안 방법이 동일한 $\text{AKI}$ 조건 하에 가장 높은 mIoU를 기록하였다. 특히 $\text{AKI}$가 25 이상으로 커질수록(즉, Key frame을 적게 사용할수록) 타 방법론 대비 성능 저하가 훨씬 완만하게 나타났다.
- **실행 속도**: $\text{FPS}$ 대비 mIoU 곡선에서도 제안 방법이 가장 효율적인 지점에 위치하여, 더 적은 계산 비용으로 더 높은 정확도를 달성함을 보였다.
- **일반화 능력**: 얼굴 비디오뿐만 아니라 Cityscapes와 같은 일반 도시 장면에서도 유사한 성능 향상 추세가 나타나, RL 기반 스케줄러의 범용성을 입증하였다.

## 🧠 Insights & Discussion

**1. 전략적 Key 배치**
시각화 결과(Consecutive Key Intervals, CKI), 제안 방법은 Key frame 사이의 간격이 매우 불규칙하게 분포되어 있었다. 이는 단순히 주기적으로 혹은 단순 차이에 따라 Key를 잡는 것이 아니라, 비디오의 동역학을 파악하여 성능 하락이 예상되는 '결정적인 시점'에 Key frame을 집중 배치한다는 것을 의미한다.

**2. 글로벌 컨텍스트의 중요성**
$\text{LKD}$와 $\text{KAR}$ 정보의 가중치를 분석한 결과, 학습이 진행될수록 이 정보들이 결정에 중요한 영향을 미쳤다. 특히 $\text{LKD}$가 커질수록(Key frame이 오랫동안 없었을 때) Key action을 취할 확률이 높아지는 경향을 보였으며, 이는 강화학습이 전역적인 예산 관리 전략을 학습했음을 시사한다.

**3. 한계 및 비판적 해석**
본 논문은 RL을 통해 효율적인 스케줄링을 구현했지만, 학습 과정에서 $\text{KAR}$ 제한치 $\eta$라는 하이퍼파라미터에 의존한다. $\eta$ 값에 따라 모델의 성향이 결정되므로, 실제 적용 시 적절한 $\eta$를 설정하기 위한 기준이 명확히 제시되지 않은 점은 아쉽다. 또한, 에피소드 길이를 270프레임 등으로 제한했는데, 훨씬 긴 비디오에서의 장기적 의존성 해결 능력에 대해서는 추가 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 실시간 비디오 세그멘테이션의 연산 비용을 줄이기 위해, **심층 강화학습을 이용한 최적의 Key frame 스케줄러**를 제안하였다. 과거의 결정 이력($\text{KAR}, \text{LKD}$)을 상태 정보로 활용하고, mIoU 향상분을 보상으로 정의하여 학습함으로써, **한정된 계산 자원을 최적의 시점에 배치하는 전략**을 학습했다. 특히 얼굴 비디오 세그멘테이션에 이를 처음 적용하여 기존의 고정/휴리스틱 방식보다 빠른 속도와 높은 정확도를 동시에 달성했으며, 이는 일반적인 시맨틱 세그멘테이션 작업으로도 확장 가능하다.
