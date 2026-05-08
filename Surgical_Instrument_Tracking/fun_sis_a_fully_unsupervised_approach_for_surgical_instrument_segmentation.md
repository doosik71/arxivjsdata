# FUN-SIS: a Fully UNsupervised approach for Surgical Instrument Segmentation

Luca Sestini, Benoit Rosa, Elena De Momi, Giancarlo Ferrigno, Nicolas Padoy (2022)

## 🧩 Problem to Solve

최소 침습 수술(Minimally Invasive Surgery, MIS)에서 내시경 영상 내 수술 도구를 자동으로 분할(Segmentation)하는 것은 수술 기술 분석, 단계 분할, 도구-조직 상호작용 추정 등 다양한 컴퓨터 보조 응용 프로그램의 핵심 구성 요소이다. 현재까지의 최첨단(State-of-the-art) 접근 방식들은 대부분 수동 주석(Manual annotation)을 통해 얻은 Ground-truth(GT) 감독 신호에 전적으로 의존하고 있다. 하지만 수술 영상의 특성상 대규모의 정교한 주석 데이터를 수집하는 것은 비용이 매우 많이 들며, 이는 모델의 일반화 성능을 제한하는 주요 원인이 된다.

또한, 일반적인 비디오 객체 분할(Video Object Segmentation, VOS) 기법들은 배경의 움직임이 전경과 무관하다는 '비일관적 배경 움직임(Incoherent background motion)' 가설에 의존한다. 그러나 수술 환경에서는 수술 도구(전경)와 조직(배경)이 강하게 상호작용하며 함께 움직이는 '일관적 움직임'이 빈번하게 발생하므로, 기존의 무감독 VOS 방식들을 그대로 적용하기 어렵다. 본 논문의 목표는 수동 주석 없이, 오직 내재된 움직임 정보(Implicit motion information)와 도구의 형상 사전 정보(Instrument shape-priors)만을 활용하여 수술 도구를 분할하는 완전 무감독(Fully-unsupervised) 학습 체계를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 움직임 정보에서 생성된 노이즈가 섞인 가상 라벨(Pseudo-labels)을 단계적으로 정제하여 깨끗한 감독 신호를 추출하는 것이다. 주요 기여 사항은 다음과 같다.

1. **무감독 Optical-flow 분할 모델(Teacher) 제안**: Shape-priors를 활용하여 Optical-flow 영상을 생성하고 동시에 분할하는 Generative-adversarial 접근 방식을 제안한다. 이는 기존 VOS의 배경 움직임 가설을 완화하여 수술 도메인의 특성에 적응하도록 설계되었다.
2. **노이즈 특성 분석 (Unpredictability & Polarization)**: Optical-flow 기반 가상 라벨에 포함된 노이즈의 두 가지 핵심 성질, 즉 개별 프레임만으로는 노이즈를 예측할 수 없다는 '예측 불가능성(Unpredictability)'과 도구가 완전히 분할되거나 완전히 누락되는 '양극화(Polarization)' 특성을 이론적, 실험적으로 분석하였다.
3. **Learning-from-noisy-labels 전략**: 확장된 Teacher-Student 구조를 통해, Teacher-Proxy 모델 간의 지역적 합의(Local agreement)를 이용해 가상 라벨의 신뢰할 수 있는 영역만을 선택적으로 학습하는 Student 모델 학습 전략을 제안한다.

## 📎 Related Works

**수술 도구 분할(Surgical Tool Segmentation)** 분야에서는 초기에는 수작업 특징(Hand-crafted features)을 사용했으나, 현재는 U-Net, VGG 기반의 완전 감독 학습(Fully-supervised) CNN 모델들이 주류를 이루고 있다. 최근에는 주석 비용을 줄이기 위해 반합성 데이터(Semi-synthetic data)를 생성하거나 Cycle-GAN을 이용해 도메인 간 변환을 시도하는 연구들이 진행되었으나, 여전히 어느 정도의 감독 신호가 필요하다는 한계가 있다.

**비디오 객체 분할(VOS)** 연구에서는 무감독 방식으로 전경 객체를 분할하려는 시도가 있었으나, 앞서 언급한 대로 수술 영상의 도구-조직 간 상관관계가 높은 움직임 때문에 기존 방식들이 실패하는 경향이 있다.

**노이즈 라벨 학습(Learning from Noisy Labels)**에서는 Robust Architecture, Robust Regularization, Robust Loss Design, 그리고 Sample Selection 방식들이 연구되어 왔다. 특히 Sample Selection은 깨끗한 라벨로 학습된 Teacher 모델이 필요한 경우가 많으나, 본 논문은 완전 무감독 환경에서 이를 구현하기 위해 노이즈의 통계적 특성을 이용하는 차별점을 가진다.

## 🛠️ Methodology

FUN-SIS는 크게 세 단계의 파이프라인으로 구성된다.

### Step I: Teacher 모델의 무감독 Optical-flow 분할

Optical-flow 도메인과 Shape-priors(현실적인 도구의 이진 마스크) 도메인 간의 매핑을 학습한다. Cycle-GAN 구조에서 영감을 얻었으나, 복잡도 불균형 문제를 해결하기 위해 다음과 같은 수정 사항을 적용하였다.

- **단일 Cycle-consistency loss**: Shape-priors 도메인에 대해서만 재구성 손실을 적용하여 Steganography 문제를 방지한다.
- **Noise vector 결합**: Generator에 랜덤 노이즈 $n$을 함께 입력하여 동일한 마스크에서도 다양한 움직임의 Optical-flow를 생성하게 함으로써 도구의 실루엣과 움직임을 분리한다.
- **Optical-flow 증강**: 랜덤 회전 행렬 $R$을 적용하여 생성자가 모든 방향의 흐름을 생성해야 하는 부담을 줄인다.

손실 함수는 다음과 같이 구성된다.

- **Cycle-consistency loss**: $\mathcal{L}_{cycle} = -\sum [m \log(\hat{m}) + (1-m) \log(1-\hat{m})]$
- **Adversarial loss**: $\mathcal{L}_{G}^{adv} = -\log(D(m_{OF}))$, $\mathcal{L}_{D}^{adv} = -\log(1-D(m_{OF})) - \log(D(y_{OF}^{t}))$
- 최종 Generator 손실: $\mathcal{L}_{G} = \mathcal{L}_{G}^{adv} + \mathcal{L}_{cycle}$

### Step II: Proxy 모델과 예측 불가능성(Unpredictability)

Teacher 모델이 생성한 가상 라벨 $y_t^T$를 사용하여 개별 프레임 $x_t$를 분할하는 Proxy 모델을 학습시킨다. 이때 **Unpredictability** 성질을 이용한다. 가상 라벨의 노이즈는 Optical-flow 추정 단계나 분할 단계에서 발생하며, 이는 개별 프레임 $x_t$만으로는 예측할 수 없다. 따라서 모델은 노이즈를 외우기보다 가장 쉬운 패턴인 '도구와 조직의 분리'라는 일반적인 특징을 먼저 학습하게 된다.
학습 손실 함수 $\mathcal{L}_{P}$는 Binary Cross Entropy($\mathcal{L}_{P}^{CE}$)와 Log IoU loss($\mathcal{L}_{P}^{IoU}$)의 가중 합으로 정의된다.
$$\mathcal{L}_{P} = \alpha_{P} \mathcal{L}_{P}^{IoU} + (1-\alpha_{P}) \mathcal{L}_{P}^{CE}$$

### Step III: Student 모델과 양극화(Polarization)

가상 라벨의 **Polarization** 성질(도구가 거의 완벽하게 분할되거나, 거의 완전히 누락됨)을 이용하여 고품질 영역만을 추출한다.

- **Local IoU ($\text{IoU}_{loc}$)**: Proxy 모델의 예측 $y_t^P$와 Teacher의 가상 라벨 $y_t^T$ 사이의 지역적 합의도를 계산한다. 윈도우 크기 $w \times h$를 슬라이딩하며 IoU를 계산하여 맵을 생성하고, 이를 임계값 $\epsilon_{IoU}$로 이진화한다.
- **Masked Loss**: $\text{IoU}_{loc}$가 높은 영역(합의가 이루어진 영역)에 대해서만 Student 모델의 손실을 계산하여 전파한다.
$$\mathcal{L}_{S}^{CE} = \frac{1}{\sum \text{IoU}_{loc}} \sum \text{IoU}_{loc} \cdot \text{BCE}(y_t^S, y_t^T)$$
이를 통해 노이즈가 심한 영역은 학습에서 배제하고 깨끗한 영역의 신호만으로 Student 모델을 정교화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2017 (로봇 수술), STRAS (유연 내시경 수술), Cholec80 (수동 복강경 수술), DAVIS 2016 (일반 객체 분할).
- **Shape-priors**: RoboTool, GrScreenTool(그린스크린 촬영), STRASMasks(CAD 모델 투영) 등을 사용하였다.
- **지표**: Mean Intersection-over-Union (mIoU).

### 주요 결과

1. **Optical-flow 분할 성능**: Teacher 모델은 EndoVis 2017 및 DAVIS 2016 데이터셋 모두에서 최신 무감독 VOS 방식인 CIS(Contextual Information Separation)보다 높은 성능을 보였다. 특히 수술 영상에서 CIS 대비 $\sim 16.32\%$ 높은 IoU 향상을 보였는데, 이는 배경 움직임 가설에 의존하지 않은 덕분이다.
2. **개별 프레임 분할 성능**: EndoVis 2017 VOS 데이터셋에서 Student 모델은 **83.77% IoU**를 달성하였다. 이는 무감독 방식인 AGSD(+12.30% 향상)보다 월등히 높으며, 완전 감독 학습 기반의 Baseline FS(88.99%)와도 근소한 차이(5.22%p)만을 보였다.
3. **전이 가능성**: 로봇 수술 데이터로 학습한 모델을 수동 복강경 수술 영상(Cholec80)에 적용했을 때도 정성적으로 유효한 분할 결과를 보여, 도메인 간 일반화 능력을 입증하였다.
4. **데이터 효율성**: Shape-priors의 양을 전체의 $1\%$ (단 5개 마스크)까지 줄여도 증강 기법을 통해 성능 하락을 최소화하며 동작함을 확인하였다.

## 🧠 Insights & Discussion

**강점 및 기여**:
본 연구는 수동 주석이 전혀 없는 상태에서 움직임 정보와 최소한의 형상 정보만으로 감독 학습에 근접한 성능을 냈다는 점에서 매우 고무적이다. 특히 노이즈의 통계적 성질(Unpredictability, Polarization)을 딥러닝 학습 프로세스(Proxy $\to$ Student)에 직접적으로 녹여내어 가상 라벨의 한계를 극복한 점이 학술적으로 가치가 높다.

**한계 및 비판적 해석**:

1. **데이터 손실**: Student 학습 시 Local IoU 마스킹을 통해 약 $49.52\%$의 픽셀 데이터를 버리게 된다. 이 불확실한 영역을 완전히 버리기보다 반지도 학습(Semi-supervised) 방식으로 활용했다면 성능을 더 높일 수 있었을 것이다.
2. **Optical-flow 의존성**: 전체 파이프라인이 Optical-flow의 품질에 크게 의존한다. 영상의 해상도가 낮거나 블러(Blur)가 심한 경우(예: STRAS 데이터셋) 성능이 저하되는 모습이 관찰되었다.
3. **단일 클래스의 한계**: 움직임 정보에만 의존하므로, 서로 다른 도구들을 구분하는 다중 클래스(Multi-class) 분할은 불가능하다. 이를 위해서는 도구별 고유의 움직임 패턴 분석이나 제한적인 시맨틱 감독 신호가 추가되어야 할 것이다.

## 📌 TL;DR

FUN-SIS는 수술 영상의 **Optical-flow(움직임)**와 **Shape-priors(형상 정보)**만을 이용해 수술 도구를 분할하는 완전 무감독 학습 프레임워크이다. **Teacher(가상 라벨 생성) $\to$ Proxy(일반 패턴 학습) $\to$ Student(고품질 영역 정제)**로 이어지는 3단계 학습 전략을 통해 노이즈 섞인 가상 라벨에서 깨끗한 신호를 추출한다. 실험 결과, 무감독 방식임에도 불구하고 완전 감독 학습 모델에 근접한 성능을 보였으며, 다양한 수술 도메인에 적용 가능한 범용성을 입증하였다. 이는 향후 라벨링되지 않은 방대한 수술 영상을 활용한 의료 AI 연구에 중요한 이정표가 될 가능성이 높다.
