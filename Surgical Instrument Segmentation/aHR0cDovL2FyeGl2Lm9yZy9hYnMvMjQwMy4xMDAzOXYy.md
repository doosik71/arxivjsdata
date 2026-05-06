# Motion-Boundary-Driven Unsupervised Surgical Instrument Segmentation in Low-Quality Optical Flow

Yang Liu, Peiran Wu, Jiayu Huo, Gongyu Zhang, Zhen Yuan, Christos Bergeles, Rachel Sparks, Prokar Dasgupta, Alejandro Granados, and Sebastien Ourselin (2025)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술 비디오에서 수동 어노테이션(manual annotation) 없이 수술 도구를 분할하는 비지도 학습 기반의 세그멘테이션 문제를 다룬다. 수술 도구 세그멘테이션은 워크플로우 인식, 동작 식별 및 추적과 같은 AI 기반 수술 작업의 핵심 구성 요소이지만, 딥러닝 기반의 기존 지도 학습 및 준지도 학습 방식은 막대한 양의 수동 어노테이션 작업이 필요하다는 한계가 있다.

특히 비지도 학습 방식은 주로 Motion cue, 즉 Optical flow에 크게 의존하는데, 내시경 영상의 특성상 어두운 영역, 급격한 카메라/도구 움직임, 또는 도구가 정지해 있는 상황 등으로 인해 Optical flow의 품질이 매우 낮게 형성되는 경우가 많다. 이러한 저품질의 Optical flow는 비지도 학습 모델의 학습 신호를 왜곡시켜 전체적인 세그멘테이션 성능을 저하시키는 주요 원인이 된다. 따라서 본 논문의 목표는 저품질의 Optical flow 환경에서도 강건하게 작동하는 비지도 수술 도구 세그멘테이션 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Optical flow의 전체 영역을 신뢰하는 대신, 움직임의 변화가 급격하게 일어나는 '경계 영역(motion boundaries)'에 집중하여 신뢰할 수 있는 학습 신호를 추출하는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계 아이디어를 제안한다.

1. **High-Quality Area Matching (HQAM)**: Optical flow에서 급격한 방향 변화가 발생하는 영역을 탐지하여 마스크를 생성하고, 이 신뢰할 수 있는 영역에 대해서만 학습 감독(supervision)을 집중시킨다.
2. **Low-Quality Case Dropping (LQCD)**: 로컬 영역뿐만 아니라 프레임 전체의 flow 품질이 현저히 낮은 'Hard Case'들을 배치(batch) 단위에서 식별하여 학습에서 제외함으로써 에러 전파를 방지한다.
3. **Variable Frame-Rate Training**: 고정된 프레임 간격이 아니라 랜덤한 간격($r \in \{1, 2, 3\}$)으로 프레임 쌍을 입력하여, 매우 느리거나 미세한 도구의 움직임까지 포착할 수 있도록 학습 감도를 높인다.

## 📎 Related Works

기존의 수술 도구 세그멘테이션 연구는 주로 완전 지도 학습(Fully-supervised) 또는 준지도 학습(Semi-supervised) 방식에 치중되어 있었으며, 이는 데이터 구축 비용이 매우 높다는 단점이 있다. 비지도 학습 분야에서는 다음과 같은 시도들이 있었다.

- **AGSD**: 비지도 방식이지만 수작업으로 설계된 큐(handcrafted cues)에 의존하여 다양한 수술 환경에 대한 적응력이 떨어진다.
- **FUN-SIS**: Optical flow와 함께 외부 데이터셋에서 얻은 Shape-priors(도구 모양에 대한 사전 정보)를 통합하여 사용한다. 그러나 Shape-priors에 대한 의존도가 높고 end-to-end 구조가 아니라는 한계가 있다.
- **RCF**: 완전 비지도 방식으로 비디오 객체 세그멘테이션(VOS)을 수행하며 Motion의 중요성을 입증하였다. 본 논문은 RCF를 백본으로 채택하되, 수술 영상 특유의 저품질 Optical flow 문제를 해결하는 모듈을 추가하여 성능을 개선하였다.

## 🛠️ Methodology

본 연구는 RCF(Relaxed Common Fate) 모델의 첫 번째 단계를 백본으로 사용하며, 사전 학습된 RAFT 모델을 통해 Pseudo flow map을 생성한다. 전체 시스템은 입력 영상 쌍을 받아 세그멘테이션 마스크를 출력하며, 학습 과정에서 제안된 HQAM과 LQCD 모듈이 flow 기반의 감독 신호를 정제한다.

### 1. High-Quality Area Matching (HQAM)

Optical flow의 내부 영역은 조명이나 정지 상태로 인해 신호가 누락되기 쉽지만, 움직임의 경계(motion boundary)는 상대적으로 신뢰도가 높다. HQAM은 이를 다음과 같은 절차로 구현한다.

먼저, 수평 및 수직 성분을 가진 Optical flow $\mathbf{o}_i \in \mathbb{R}^{H \times W \times 2}$를 방향각 $\theta$로 변환한다. 픽셀 $p=(j,k)$에서의 각도 $\theta_p^i$는 다음과 같이 계산된다.
$$\theta_p^i = \arctan 2(o_{j,k,x}^i, o_{j,k,y}^i)$$

그 다음, 픽셀 $p$와 그 주변 4-이웃(4-neighbourhood) $N_p^1$ 간의 최대 각도 차이 $\delta_p^i$를 계산하여 경계 영역을 식별한다.
$$\delta_p^i = \max_{n_p \in N_p^1} |\theta_p^i - \theta_{n_p}^i|$$

임계값 $\alpha = \pi/12$를 기준으로 $\delta_p^i > \alpha$인 영역을 1, 그렇지 않은 영역을 0으로 하는 경계 마스크 $M_i$를 생성한다. 또한, 감독 영역을 확장하기 위해 팽창(dilation) 기술을 적용한다. 최종적으로 인스턴스 수준의 손실 함수 $L_{ins}$는 마스크 $M_i$가 1인 영역에 대해서만 가중 평균을 수행한다.
$$L_{ins} = \frac{\sum_{p \in \Omega} M_i(p) \|o_i(p) - \hat{o}_i(p)\|^2}{\sum_{p \in \Omega} M_i(p)}$$

### 2. Low-Quality Case Dropping (LQCD)

특정 프레임 전체가 저품질 flow를 가질 경우, 로컬 마스킹만으로는 부족하다. 따라서 배치 크기 $B$ 내에서 각 프레임의 손실 값 $L_{ins}$를 계산하고, 손실이 가장 높은 상위 $h$개의 'Hard Case'를 제거한 나머지 집합 $S_{remain}$에 대해서만 최종 배치 손실을 계산한다.
$$L_{batch} = \frac{1}{|S_{remain}|} \sum_{x_i \in S_{remain}} L_{ins}(x_i)$$

### 3. Variable Frame Rates

도구가 거의 움직이지 않는 경우 Optical flow 값이 0에 가까워 학습 신호가 부족해진다. 이를 해결하기 위해 인접 프레임($r=1$)뿐만 아니라 $r \in \{1, 2, 3\}$ 범위의 랜덤한 간격을 가진 프레임 쌍 $(x_i, x_{i+r})$을 입력으로 제공하여 움직임의 포착 확률을 높였다.

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI EndoVis 2017 Robotic Instrument Segmentation Challenge (VOS 버전 및 Challenge 버전).
- **평가 지표**: mean Intersection-over-Union (mIoU).
- **구현 세부사항**: ResNet50 기반의 RCF 백본, RAFT(FlyingChairs, FlyingThings로 사전 학습됨)를 통한 Optical flow 생성, NVIDIA A100 GPU 사용.

### 정량적 결과

본 방법은 기존 비지도 학습 모델 대비 괄목할 만한 성능 향상을 보였다.

- **EndoVis 2017 VOS**: mIoU $0.75$ 달성 (Baseline RCF 대비 $28.98$ pp 증가).
- **EndoVis 2017 Challenge**: mIoU $0.72$ 달성 (Baseline RCF 대비 $25.93$ pp 증가).

특히, 완전 비지도 방식임에도 불구하고 수작업 기반의 pseudo-label을 사용한 AGSD보다 높은 성능을 기록하였으며, Shape-priors를 사용하는 FUN-SIS(Stage 2)보다도 우수한 결과를 보였다.

### Ablation Study

각 구성 요소의 기여도는 다음과 같다 (VOS 데이터셋 기준).

- **Baseline (RCF)**: $46.09\%$
- **Baseline + LQCD**: $47.15\%$ (노이즈 제거 효과)
- **Baseline + LQCD + HQAM**: $74.47\%$ (경계 큐 활용의 결정적 영향)
- **Baseline + LQCD + HQAM + Variable**: $75.07\%$ (미세 움직임 포착을 통한 최종 최적화)

## 🧠 Insights & Discussion

본 논문은 수술 영상의 저품질 Optical flow 문제를 해결하기 위해 '선택적 집중'과 '과감한 제거'라는 전략을 취했다. 특히 HQAM 모듈이 성능 향상에 가장 크게 기여했다는 점은, 비지도 학습에서 모든 데이터를 활용하는 것보다 신뢰도가 높은 고품질 영역(High-contrast motion areas)만을 선별하여 학습하는 것이 훨씬 효과적임을 시사한다.

또한, 제안된 모듈들이 Plug-and-play 형태로 설계되어 있어, 다른 motion-driven 작업(예: 비지도 깊이 추정)으로 확장 가능하다는 잠재력을 가지고 있다. 다만, 저자들도 언급했듯이 비지도 학습 특성상 $\alpha$나 $d$와 같은 하이퍼파라미터 설정에 따른 성능 변동이 존재한다. 이는 향후 하이퍼파라미터에 덜 민감하고 더 일반화된 프레임워크를 구축해야 할 필요성을 제기한다.

## 📌 TL;DR

본 연구는 저품질 Optical flow로 인해 어려움을 겪는 비지도 수술 도구 세그멘테이션을 위해, **신뢰할 수 있는 움직임 경계 영역만을 추출(HQAM)**하고, **품질이 낮은 프레임을 제거(LQCD)**하며, **가변 프레임 레이트로 미세 움직임을 포착**하는 방법을 제안하였다. 이를 통해 어노테이션 없이도 SOTA 수준의 성능(mIoU 0.75/0.72)을 달성하였으며, 이는 수술 영상 분석의 확장성과 강건성을 크게 높이는 결과이다.
