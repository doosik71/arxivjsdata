# Inter-slice image augmentation based on frame interpolation for boosting medical image segmentation accuracy

Zhaotao Wu, Jia Wei, Wenguang Yuan, Jiabing Wang, Tolga Tasdizen (2020)

## 🧩 Problem to Solve

의료 영상 분석에서 딥러닝 모델의 성능을 높이기 위해서는 대량의 학습 데이터가 필요하지만, 실제 의료 현장에서는 희귀 사례, 한정된 의료 자원, 고비용의 라벨링 작업 등으로 인해 고품질의 데이터를 충분히 확보하는 것이 매우 어렵다.

기존의 데이터 증강(Data Augmentation) 방식은 회전(Rotation), 뒤집기(Flipping), 스케일링(Scaling)과 같은 단순한 파라미터 기반의 변환을 통해 가상 샘플을 추가하는 방식이다. 그러나 이러한 방식은 단순히 샘플의 수를 늘릴 뿐, 데이터가 가진 본질적인 정보량을 증가시키지는 못한다는 한계가 있다.

특히, 의료 볼륨 데이터는 평면 방향(In-plane)의 해상도는 높지만, 슬라이스 간 거리(Inter-slice distance)가 멀어 평면 관통 방향(Through-plane)의 해상도는 낮다는 특성이 있다. 본 논문은 이러한 특성에 주목하여, 인접한 두 슬라이스 사이에 중간 슬라이스를 생성함으로써 학습 데이터의 수와 정보량을 동시에 증강하여 의료 영상 분할(Segmentation)의 정확도를 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 의료 볼륨 데이터를 공간적으로 연속된 이미지 시퀀스로 간주하고, 비디오 프레임 보간(Frame Interpolation) 기법을 적용하여 인접한 두 슬라이스 사이의 중간 이미지와 그에 대응하는 분할 라벨을 생성하는 것이다.

주요 기여 사항은 다음과 같다:

1. **프레임 보간 기반 증강**: 두 개의 연속된 슬라이스와 라벨로부터 임의의 수의 중간 슬라이스와 라벨을 생성하는 데이터 증강 프레임워크를 제안하였다.
2. **이중 판별자(Dual Discriminators) 구조**: 전체 이미지의 진위 여부를 판단하는 Global Discriminator와 분할에 유용한 특정 영역의 진위 여부를 판단하는 Local Discriminator를 도입하여 합성 이미지의 품질을 높였다.
3. **적응형 어텐션 네트워크(Adaptive Attention Network)**: Local Discriminator 내에 어텐션 메커니즘을 통합하여, 모델이 타겟 객체와 그 주변의 유용한 특징에 자동으로 집중하게 함으로써 합성 이미지의 사실성을 극대화하였다.

## 📎 Related Works

### 1. 의료 영상 분할 (Medical Image Segmentation)

전통적으로는 영역 기반, 에지 검출 기반, 그래프 기반 방법들이 사용되었으나, 최근에는 U-Net과 같은 Deep Learning 기반 방법들이 우수한 성능을 보이고 있다. 특히 U-Net은 수축 경로(Contracting path)와 확장 경로(Expansive path)를 통한 Skip-connection을 통해 위치 정보와 세밀한 텍스트 정보, 그리고 의미론적 정보를 동시에 학습할 수 있어 의료 영상 분야에서 널리 사용된다.

### 2. 데이터 증강 (Data Augmentation)

회전, 스케일링, 이동, 감마 보정 및 탄성 변형(Elastic transformation) 등이 주로 사용된다. 이러한 방법들은 구현이 쉽고 일반화 성능을 높이는 데 효과적이지만, 앞서 언급했듯 새로운 공간적 정보를 창출하는 데는 한계가 있다.

### 3. 비디오 보간 (Video Interpolation)

두 프레임 사이의 중간 프레임을 생성하는 기술로, 주로 Optical Flow를 예측하여 워핑(Warping)하는 방식이나, 최근에는 Optical Flow 없이 직접 커널을 추정하여 보간하는 방식들이 제안되었다. 본 논문은 이러한 비디오의 시간적 연속성을 의료 영상의 공간적 연속성으로 치환하여 적용하였다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 방법은 크게 세 단계로 구성된다: (1) 객체 분류기(Object Classifier) 학습을 통한 어텐션 네트워크 확보 $\rightarrow$ (2) 중간 슬라이스 합성 네트워크(Intermediate Slice Synthesis Network) 학습 $\rightarrow$ (3) 합성된 이미지와 라벨을 이용한 U-Net 분할 모델 학습.

### 1. 객체 분류기 및 어텐션 네트워크

합성 이미지의 유용한 부분을 집중적으로 학습시키기 위해, 먼저 입력 이미지에 타겟 객체가 존재하는지 판별하는 분류기를 학습시킨다.

- **구조**: 특징 추출 브랜치와 액티베이션 맵(Activation map) 생성 브랜치로 구성된 어텐션 네트워크와 Fully Connected Layer 기반의 분류기로 이루어져 있다.
- **목표**: 분류를 위해 이미지의 어느 부분이 중요한지를 학습하게 하여, 이후 Local Discriminator에서 사용할 어텐션 맵을 생성한다.
- **손실 함수**:
$$\ell = -\sum_{i=1}^{n} Y_i \log \hat{Y}_i$$
여기서 $Y_i$는 실제 존재 여부, $\hat{Y}_i$는 예측값이다.

### 2. 중간 슬라이스 합성 모델

두 입력 이미지 $I_0$와 $I_{T+1}$ 사이의 $T$개의 중간 슬라이스 $\{\hat{I}_t\}_{t=1}^T$를 생성한다.

- **작동 원리**: U-Net 구조를 통해 양방향 공간 변환 $\hat{F}_{t \to 0}$와 $\hat{F}_{t \to T+1}$을 계산하고, 이를 이용하여 두 이미지를 워핑한 후 선형 결합한다.
- **합성 방정식**:
$$\hat{I}_t = \left(1 - \frac{t}{T+1}\right)g(I_0, \hat{F}_{t \to 0}) + \frac{t}{T+1}g(I_{T+1}, \hat{F}_{t \to T+1})$$
여기서 $g(\cdot, \cdot)$는 Bilinear interpolation을 이용한 Backward warping 함수이다.

- **손실 함수**: 재구성 손실($\ell_{rec}$), 지각 손실($\ell_{per}$), 워핑 손실($\ell_{warp}$), 매끄러움 손실($\ell_{smooth}$)과 적대적 손실($\ell_{adv}$)의 가중 합으로 정의된다.
$$\ell = \lambda_{rec}\ell_{rec} + \lambda_{per}\ell_{per} + \lambda_{warp}\ell_{warp} + \lambda_{smooth}\ell_{smooth} + \lambda_{adv}\ell_{adv}$$
여기서 $\ell_{adv}$는 다음과 같이 정의된다:
$$\ell_{adv} = -\frac{1}{T}\sum_{t=1}^T \log LD(\hat{I}_t) - \frac{1}{T}\sum_{t=1}^T \log GD(\hat{I}_t)$$

### 3. 판별자 (Discriminators)

- **Global Discriminator (GD)**: 이미지 전체를 입력받아 합성 이미지와 실제 이미지의 진위를 판별한다.
- **Local Discriminator (LD)**: 사전 학습된 어텐션 네트워크를 통해 추출된 '유용한 영역'만을 입력받아 진위를 판별한다. 이를 통해 타겟 객체 주변의 디테일한 사실성을 높인다.

### 4. 데이터 증강 및 분할

학습된 합성 모델을 사용하여 연속된 두 슬라이스 $I_0, I_1$과 라벨 $L_0, L_1$로부터 중간 이미지 $\hat{I}_t$와 라벨 $\hat{L}_t$를 생성한다. 이때 이미지와 라벨에 동일한 공간 변환을 적용하여 라벨의 정확성을 보장한다.
$$\hat{I}_t = (1-t)g(I_0, \hat{F}_{t \to 0}) + tg(I_1, \hat{F}_{t \to 1})$$
$$\hat{L}_t = (1-t)g(L_0, \hat{F}_{t \to 0}) + tg(L_1, \hat{F}_{t \to 1})$$

## 📊 Results

### 실험 설정

- **데이터셋**: SLIVER07 (간 CT 영상), CHAOS2019 (복부 장기 CT 영상). 데이터 부족 상황을 시뮬레이션하기 위해 학습 데이터셋을 환자 수 5명, 7명, 9명으로 제한하여 실험하였다.
- **지표**: Dice Score (분할된 영역과 실제 영역의 겹침 정도를 측정).
- **비교 대상 (Baselines)**:
  - 증강 없음 (Normal)
  - 전통적 증강: Rotation, Scaling, Gamma correction, Elastic transformation (rand-aug).
  - 제안 방법 변형: `ours-normal` (판별자 없음), `ours-Dis` (어텐션 없는 판별자), `ours-Dis-Att` (어텐션 포함 판별자).

### 주요 결과

1. **정량적 성능**: 모든 케이스에서 `ours-Dis-Att`가 가장 높은 Dice Score를 기록하였다. 특히 학습 데이터가 매우 적은 상황(환자 5명)에서도 전통적인 증강 기법보다 월등한 성능 향상을 보였다.
2. **정성적 분석**: Elastic transformation으로 생성된 이미지는 에지가 불연속적인 반면, 제안 방법은 훨씬 매끄럽고 사실적인 에지를 생성하였다. 특히 어텐션 메커니즘을 사용했을 때 타겟 객체 내부의 텍스트가 더욱 명확하게 생성되었다.
3. **보간 슬라이스 수의 영향**: 두 슬라이스 사이에 3개의 중간 슬라이스를 생성했을 때 가장 좋은 성능이 나타났다. 슬라이스 수를 너무 많이 늘리면(예: 5개 이상), 실제 이미지와의 괴리(Deviation)가 커져 오히려 성능이 저하되는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 단순히 데이터를 기하학적으로 변형하는 기존 방식에서 벗어나, 의료 영상의 공간적 연속성이라는 도메인 특성을 활용하여 **새로운 정보가 포함된 데이터를 생성**했다는 점이 매우 뛰어나다. 특히 Local Discriminator와 Attention Network의 조합은 의료 영상에서 가장 중요한 '관심 영역(ROI)'의 품질을 집중적으로 개선하여 실제 분할 성능 향상으로 이끌었다.

### 한계 및 비판적 해석

- **모달리티의 제한**: 본 논문은 CT 영상에 대해서만 검증되었다. MRI와 같이 신호 강도가 다른 모달리티에서도 동일한 효과가 있을지는 추가 검증이 필요하다.
- **합성 데이터의 한계**: 실험 결과에서 보듯, 보간 슬라이스 수가 늘어날수록 성능이 떨어지는 지점이 존재한다. 이는 모델이 생성한 데이터가 완벽한 Ground Truth가 아니며, 누적된 오차가 학습에 악영향을 줄 수 있음을 시사한다.
- **가정의 타당성**: "분류에 유용한 영역이 분할에도 유용하다"는 가정을 통해 어텐션 네트워크를 설계하였는데, 이는 직관적으로 타당하나 두 작업(Classification vs Segmentation)의 특성 차이에 대한 심도 있는 분석은 부족해 보인다.

## 📌 TL;DR

본 논문은 의료 영상의 슬라이스 간 간격이 넓다는 점에 착안하여, **비디오 프레임 보간 기술을 이용해 연속된 슬라이스 사이의 중간 이미지와 라벨을 생성하는 데이터 증강 기법**을 제안한다. 특히 **Global/Local 이중 판별자와 어텐션 메커니즘**을 도입하여 합성 데이터의 사실성을 높였으며, 이를 통해 데이터가 부족한 상황에서도 U-Net 기반 의료 영상 분할 성능을 유의미하게 향상시켰다. 이 연구는 향후 3D 의료 영상 분할의 데이터 부족 문제를 해결하는 데 중요한 기초가 될 가능성이 높다.
