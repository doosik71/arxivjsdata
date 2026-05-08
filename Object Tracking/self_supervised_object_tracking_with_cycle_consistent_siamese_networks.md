# Self-supervised Object Tracking with Cycle-consistent Siamese Networks

Weihao Yuan, Michael Yu Wang, and Qifeng Chen (2020)

## 🧩 Problem to Solve

본 논문은 비디오 내의 객체 추적(Visual Object Tracking)과 비디오 객체 분할 전파(Video Object Segmentation Propagation)를 위한 자기지도 학습(Self-supervised Learning) 방법을 제안한다.

기존의 딥러닝 기반 추적기들은 높은 성능을 보이지만, 학습을 위해 매 프레임마다 정교하게 작성된 정답 라벨(Ground-truth trajectories)이 필요하다는 치명적인 단점이 있다. 이러한 수동 어노테이션 작업은 매우 많은 비용과 시간이 소요되며, 이는 학습 데이터의 크기를 제한하고 새로운 환경(unseen scenarios)에 대한 적용력을 떨어뜨린다. 따라서 본 연구의 목표는 인간의 개입 없이 데이터 자체의 일관성을 이용하여 학습할 수 있는 end-to-end Siamese 네트워크 기반의 자기지도 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Cycle-consistency(사이클 일관성)**를 이용한 자기지도 학습이다. 구체적으로, 첫 번째 프레임에서 타겟 객체를 추적하여 이후 프레임으로 이동(Forward Tracking)시킨 뒤, 다시 역순으로 첫 번째 프레임으로 돌아오게(Backward Tracking) 하는 과정을 설계하였다.

만약 추적 네트워크가 정확하다면, 사이클을 돌아온 최종 예측 위치는 처음 시작했던 위치와 일치해야 한다. 이 두 위치 사이의 차이를 손실 함수(Loss function)로 활용함으로써, 별도의 정답 라벨 없이도 네트워크를 최적화할 수 있는 강력한 감독 신호를 생성한다. 또한, 이를 위해 Siamese Region Proposal Network(RPN)와 Mask Regression 네트워크를 통합하여 추적의 정확도와 속도를 동시에 향상시켰다.

## 📎 Related Works

기존의 객체 추적 연구는 크게 Correlation Filter 기반 방법과 Siamese 네트워크 기반 방법으로 나뉜다. Correlation Filter는 속도가 빠르지만 성능에 한계가 있으며, Siamese 네트워크는 템플릿과 검색 영역 간의 상호 상관관계(Cross-correlation)를 통해 높은 정확도를 보이지만 대규모의 지도 학습 데이터에 의존한다.

최근 UDT와 같은 자기지도 학습 기반 추적기들이 등장하였으나, 이들은 단순한 Discriminative Correlation Filter를 사용하여 딥러닝 네트워크의 end-to-end 학습 능력을 충분히 활용하지 못했다. 한편, 비디오 객체 분할 전파(VOS Propagation) 분야에서는 사이클 일관성을 이용한 연구들이 있었으나, 대부분 시각적 표현(Representation)을 먼저 학습한 뒤 별도의 대응 매칭(Correspondence matching)을 수행하는 방식이었다. 본 논문은 이를 end-to-end 구조로 통합하여 실시간 성능과 정확도를 동시에 확보함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

제안된 프레임워크는 크게 **Cycle Object Tracking**과 **Siamese Tracking Network**로 구성된다. 전체 파이프라인은 다음과 같다.

1. **Forward Tracking**: $t_1$ 시점의 프레임 $I_1$에서 타겟 패치 $p_1$을 $t_2$ 시점의 프레임 $I_2$로 추적하여 $\tilde{p}_2$를 얻는다.
2. **Backward Tracking**: 예측된 $\tilde{p}_2$를 다시 $I_1$ 프레임으로 역추적하여 $\tilde{p}_1$을 얻는다.
3. **Self-supervision**: 초기 위치 $p_1$과 최종 복귀 위치 $\tilde{p}_1$ 사이의 불일치를 계산하여 네트워크를 학습시킨다.

### 2. Siamese Tracking Network

네트워크는 템플릿 이미지 패치 $z$와 검색 이미지 $x$를 입력으로 받는 Siamese 구조를 가진다.

- **Feature-Net**: 두 입력을 동일한 가중치를 공유하는 Fully Convolutional Subnetwork에 통과시켜 특징을 추출한다.
- **Cross-correlation**: 추출된 특징들 간의 depth-wise cross-correlation을 수행하여 dense response map을 생성한다.
- **Box-Net & Score-Net**:
  - **Box-Net**: 각 위치에서 $k$개의 Bounding Box 후보군을 회귀(Regression)한다.
  - **Score-Net**: 각 후보 박스가 객체인지 배경인지에 대한 점수를 분류한다.

박스 좌표는 R-CNN 방식을 따라 다음과 같이 정규화된 좌표 $t$로 인코딩된다.
$$t_x = \frac{x-x_a}{w_a}, \quad t_y = \frac{y-y_a}{h_a}, \quad t_w = \log \frac{w}{w_a}, \quad t_h = \log \frac{h}{h_a}$$
여기서 $(x, y, w, h)$는 예측 박스의 좌표이며, $(x_a, y_a, w_a, h_a)$는 앵커 박스의 좌표이다.

### 3. 손실 함수 및 학습 절차

학습은 사이클이 완전히 종료된 후, 초기 타겟과 최종 예측값 사이의 오차를 계산하는 방식으로 이루어진다.

- **Box Localization Loss**: Smooth $L_1$ 손실을 사용하여 좌표 오차를 계산한다.
$$\mathcal{L}_{box} = l_1(t_x - t^*_x) + l_1(t_y - t^*_y) + l_1(t_w - t^*_w) + l_1(t_h - t^*_h)$$
- **Object Score Loss**: Cross-entropy 손실을 사용하여 객체/배경 분류 성능을 최적화한다.
$$\mathcal{L}_{sco} = -[y_o \log(p_{obj}) + (1-y_o) \log(1-p_{obj}) + y_b \log(p_{back}) + (1-y_b) \log(1-p_{back})]$$
- **최종 손실 함수**:
$$\mathcal{L} = \mathcal{L}_{sco} + \lambda_1 \mathcal{L}_{box}$$

### 4. Self-supervised Segmentation Propagation

추적 기능을 확장하여 픽셀 수준의 마스크를 예측하는 **Mask-Net** 브랜치를 추가하였다.

- **구조**: Cross-correlation 이후 Mask-Net이 각 위치에 대해 평탄화된(flattened) 벡터 형태의 마스크를 예측한다.
- **마스크 손실 함수**: 예측된 마스크 $m_{ij}^n$과 실제 라벨 $c_{ij}^n$ 사이의 일관성을 계산한다.
$$\mathcal{L}_{mask} = \sum_n \left( 1 + y_n \frac{2}{w_m h_m} \sum_{i,j} \log(1 + e^{-c_{ij}^n m_{ij}^n}) \right)$$
- **최종 통합 손실**: $\mathcal{L} = \mathcal{L}_{sco} + \lambda_1 \mathcal{L}_{box} + \lambda_2 \mathcal{L}_{mask}$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 시각적 객체 추적은 VOT-2016, VOT-2018에서 평가하였고, 분할 전파는 DAVIS-2016, DAVIS-2017에서 평가하였다. 학습은 ILSVRC-2015 및 YouTube-VOS 데이터셋을 사용하였다.
- **지표**: 추적 작업에서는 Accuracy, Robustness, EAO(Expected Average Overlap) 및 속도(fps)를 측정하였고, 분할 전파에서는 Jaccard index($J$)와 Contour F-measure($F$)를 사용하였다.

### 2. 정량적 결과

- **객체 추적 (VOT-2016)**: 제안된 CycleSiam은 EAO 0.371을 달성하여, 기존 최신 자기지도 학습 방법인 UDT(0.226~0.301)를 큰 차이로 앞질렀다.
- **분할 전파 (DAVIS-2017)**: CycleSiam+는 $J=50.9, F=56.8$을 기록하며 기존 자기지도 학습 방법들(Wang et al., Lai et al.)보다 우수한 성능을 보였다. 특히, 이전 프레임 여러 개를 사용하는 기존 방식과 달리 단일 프레임만으로 실시간 추론이 가능하다는 점이 강조되었다.

### 3. 분석 결과

- **Ablation Study**: 타겟을 무작위로 초기화하여 학습시켰을 때도 어느 정도 작동함을 확인하였으나, 정확한 객체 초기화 시보다 성능이 하락하였다. 이는 배경이나 불필요한 객체가 포함될 경우 네트워크가 혼란을 겪기 때문으로 분석된다.
- **정성적 결과**: Basketball, Motocross와 같이 객체가 뭉치거나 영상이 흐릿한 상황에서도 안정적인 추적 성능을 보였으며, 일부 사례에서는 정답 라벨보다 더 합리적인 예측 결과를 나타내기도 하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 정답 라벨 없이도 사이클 일관성이라는 기하학적 제약 조건을 통해 딥러닝 네트워크를 효과적으로 학습시킬 수 있음을 증명하였다. 특히, 단순한 필터 기반의 자기지도 학습에서 벗어나 Siamese RPN이라는 강력한 아키텍처를 end-to-end로 학습시킨 점이 성능 향상의 핵심 요인이다. 또한, 추적과 분할 전파라는 두 가지 다른 태스크를 하나의 프레임워크 내에서 통합적으로 해결하였으며, 실시간 동작이 가능하다는 실용적인 이점을 가진다.

### 한계 및 비판적 해석

- **초기화 의존성**: Ablation study에서 나타나듯, 학습 시 초기 타겟 박스가 매우 부정확할 경우 성능이 급격히 저하된다. 이는 자기지도 학습의 고질적인 문제인 '잘못된 신호(noisy signal)'에 대한 취약성을 보여준다.
- **마스크 정확도**: 분할 전파 결과에서 마스크 예측이 완전히 정교하지 못한 부분이 발견되었는데, 이는 지도 학습 데이터 없이 사이클 일관성만으로는 픽셀 단위의 세밀한 경계선을 학습하는 데 한계가 있음을 시사한다.
- **가정 사항**: 본 모델은 전방-후방 추적의 일관성이 곧 정답이라는 가정을 전제로 한다. 하지만 객체가 완전히 사라졌다가 다시 나타나거나, 매우 유사한 다른 객체로 전이된 경우(ID switch), 사이클 일관성이 오히려 잘못된 학습 방향을 제시할 위험이 있다.

## 📌 TL;DR

본 논문은 전방 추적 후 다시 역순으로 추적하여 원래 위치로 돌아오는지 확인하는 **Cycle-consistency** 메커니즘을 도입하여, 사람이 만든 라벨 없이도 학습 가능한 **자기지도 Siamese 추적 네트워크**를 제안하였다. 이 방법은 Siamese RPN 및 마스크 회귀 네트워크를 통합하여 객체 추적(VOT)과 분할 전파(DAVIS) 두 가지 작업 모두에서 기존 자기지도 학습 모델들을 압도하는 성능을 보였으며, 특히 실시간 처리가 가능하다는 점에서 실용성이 매우 높다. 향후 라벨링 비용 절감이 절실한 실시간 비디오 분석 시스템에 적용될 가능성이 크다.
