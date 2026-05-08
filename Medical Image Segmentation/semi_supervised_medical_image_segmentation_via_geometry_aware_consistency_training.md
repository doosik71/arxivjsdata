# Semi-supervised Medical Image Segmentation via Geometry-aware Consistency Training

Zihang Liu, Chunhui Zhao (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 지도 학습(Supervised Learning)의 성능을 제한하는 핵심 요인인 **레이블링 된 데이터의 부족(Scarcity of labeled data)** 문제를 해결하고자 한다. 의료 영상의 특성상 전문의가 직접 레이블을 생성하는 과정은 매우 비용이 많이 들고 노동 집약적이기 때문에, 소량의 레이블 데이터와 다량의 레이블 없는 데이터를 함께 사용하는 준지도 학습(Semi-supervised Learning)의 필요성이 대두되었다.

특히 저자들은 기존의 준지도 학습 방법들이 가진 두 가지 한계점을 지적한다. 첫째, 영상 내의 모든 영역을 동일한 중요도로 처리하여, 분할 난이도가 높고 중요한 정보가 집중된 **경계 영역(Boundary regions)**에 대한 고려가 부족하다는 점이다. 둘째, 의료 영상 특유의 노이즈와 낮은 해상도로 인해 발생하는 **예측 불확실성(Prediction uncertainty)** 문제가 존재한다는 점이다. 따라서 본 연구의 목표는 기하학적 정보를 활용하여 모호한 경계 영역에 집중하고, 예측 불확실성을 줄임으로써 적은 양의 레이블 데이터만으로도 높은 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **기하학적 구조 인지(Geometry-awareness)**와 **이중 뷰 네트워크(Dual-view network)**를 통한 일관성 학습이다.

1. **기하학적 인지 프레임워크**: 단순히 픽셀 단위의 분류를 수행하는 것을 넘어, 객체의 전역적 기하학적 윤곽을 학습하는 보조 작업인 Signed Distance Map(SDM) 예측을 도입하였다.
2. **지수 가중치 일관성 손실(Exponentially Weighted Consistency Loss)**: SDM을 통해 계산된 경계로부터의 거리를 이용하여, 경계 근처의 어려운 샘플(Hard samples)에 더 높은 가중치를 부여하는 전략을 제안하여 학습 효율을 높였다.
3. **이중 뷰 네트워크(Dual-view Network)**: 서로 다른 업샘플링 전략(Deconvolution 및 Tri-linear Interpolation)을 사용하는 두 개의 디코더를 구성하여 모델의 다양성을 확보하고 예측 불확실성을 낮추었다.

## 📎 Related Works

### 기존 연구 및 한계

- **지도 학습 기반 분할**: U-Net, V-Net과 같은 인코더-디코더 구조가 주류를 이루며 높은 성능을 보였으나, 대량의 레이블 데이터가 필수적이며 데이터 부족 시 과적합(Overfitting) 문제가 심각하다.
- **준지도 학습 접근법**:
  - **Self-training**: 의사 레이블(Pseudo-labels)을 생성하여 재학습하는 방식이나, 초기 모델의 오류가 전파되어 강화될 위험이 있다.
  - **Consistency-based**: 입력 데이터에 섭동(Perturbation)을 주어 예측 결과의 일관성을 강제하는 방식(예: Mean-Teacher, $\pi$-model)이다. 하지만 대부분의 방법이 영역별 중요도의 차이를 무시하고 전역적인 일관성만을 강조한다는 한계가 있다.

### 차별점

본 논문은 기존의 일관성 기반 학습에 **기하학적 제약(Geometric constraint)**을 결합하였다. 단순한 데이터 섭동이 아니라, 세만틱 정보(Segmentation map)와 기하학적 정보(SDM)라는 서로 다른 관점에서의 일관성을 강제함으로써 모델의 강건성을 높였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 네트워크는 공유 인코더 $E$와 두 개의 서로 다른 디코더 $D_1, D_2$로 구성된다. 각 디코더는 동일한 인코더 특징을 입력받아 두 가지 결과물을 출력한다.

- **Segmentation map ($f_{seg}$)**: 픽셀 단위의 분할 결과.
- **Signed Distance Map ($f_{sdm}$)**: 객체 경계로부터의 거리를 나타내는 맵.

두 디코더는 각각 Deconvolution과 Tri-linear Interpolation이라는 서로 다른 업샘플링 방식을 사용하여 모델의 다양성을 확보한다.

### 2. Signed Distance Map (SDM) 및 변환

SDM은 객체의 경계 $\partial\Omega$를 기준으로 내부 영역 $\Omega_{in}$은 음수 값, 외부 영역 $\Omega_{out}$은 양수 값을 가지며, 경계 위에서는 0을 갖는 맵이다. 수식은 다음과 같다.

$$T(x) = \begin{cases} -\inf_{y \in \partial\Omega} \|x-y\|_2, & x \in \Omega_{in} \\ 0, & x \in \partial\Omega \\ +\inf_{y \in \partial\Omega} \|x-y\|_2, & x \in \Omega_{out} \end{cases}$$

예측된 SDM($f_{sdm}$)을 다시 분할 맵 형태로 변환하기 위해 다음과 같은 시그모이드 형태의 근사 함수 $T^{-1}$를 사용한다.

$$T^{-1}(t) = \frac{1}{1 + e^{-k \cdot t}}$$

여기서 $k$는 하이퍼파라미터이며, 값이 클수록 이진 분할 맵에 더 가까운 형태가 된다.

### 3. 지수 가중치 일관성 학습 (Exponentially Weighted Consistency)

저자들은 경계 영역의 중요성을 강조하기 위해 SDM 값을 이용한 가중치 $\omega$를 설계하였다.

$$\omega_j = e^{-\rho \cdot |f_{sdm}^j(x)|}, \quad j=1,2$$

여기서 $\rho$는 가중치 분포를 조절하는 하이퍼파라미터이다. 경계에 가까울수록 $|f_{sdm}|$ 값이 작아지므로 가중치 $\omega$는 커지게 된다. 이를 적용한 최종 가중치 일관성 손실 $\mathcal{L}_{dwgc}$는 두 디코더의 서로 다른 작업(Segmentation $\leftrightarrow$ SDM) 간의 교차 일관성을 측정한다.

$$\mathcal{L}_{dwgc}(x) = \sum_{x_i \in \Omega} \left( \omega_1 \cdot \|f_{seg}^1(x_i) - T^{-1}(f_{sdm}^2(x_i))\|^2 + \omega_2 \cdot \|f_{seg}^2(x_i) - T^{-1}(f_{sdm}^1(x_i))\|^2 \right)$$

### 4. 전체 학습 절차 및 손실 함수

전체 손실 함수 $\mathcal{L}_{total}$은 지도 학습 손실 $\mathcal{L}_{sup}$과 가중치 일관성 손실 $\mathcal{L}_{dwgc}$의 합으로 정의된다.

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \mu \mathcal{L}_{dwgc}$$

- **지도 학습 손실 ($\mathcal{L}_{sup}$)**: 레이블 데이터에 대해 Dice loss와 Cross-entropy loss의 합($\mathcal{L}_{seg}$)과 SDM 예측 오차($\mathcal{L}_{sdm}$)를 함께 최적화한다.
  $$\mathcal{L}_{sup} = \mathcal{L}_{seg} + \beta \cdot \mathcal{L}_{sdm}$$
- **$\mu$ (Ramp-up function)**: 학습 초기에는 일관성 손실의 영향력을 낮추어 학습을 안정화하기 위해 가우시안 램프업 함수를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Left Atrium (100개 사례), Brain Tumor (BraTS 2019, 335개 사례).
- **설정**: 레이블 데이터를 10% 또는 20%만 사용하여 준지도 학습 환경을 구축.
- **평가 지표**: Dice Similarity Coefficient (Dice), Jaccard Index, Average Surface Distance (ASD), 95% Hausdorff Distance (95HD).

### 주요 결과

1. **정량적 성능**: Left Atrium 데이터셋에서 20% 레이블 사용 시 Dice 90.34%를 달성하여 V-Net 기반의 다른 준지도 학습 방법들(DAP, UA-MT, SASSNet 등)보다 우수한 성능을 보였다. 특히 10%의 극소량 데이터 환경에서 타 방법론 대비 성능 하락 폭이 훨씬 적어 강건함을 입증하였다.
2. **경계 정밀도**: 경계 기반 지표인 ASD와 95HD에서 경쟁 모델들을 크게 앞질렀다 (예: Left Atrium 20% 설정에서 ASD 1.70mm 기록). 이는 기하학적 제약 조건이 경계 영역의 정밀도를 높이는 데 결정적인 역할을 했음을 시사한다.
3. **일반화 능력**: 정형화된 구조를 가진 좌심방뿐만 아니라, 형태와 크기가 매우 다양한 뇌종양(Brain Tumor) 데이터셋에서도 Dice 83.54%를 기록하며 SOTA 성능을 달성하여 범용적인 적용 가능성을 확인하였다.

### Ablation Study

- **Dual-view 효과**: 단일 디코더보다 두 개의 다른 디코더를 사용할 때 예측 불확실성이 줄어들어 성능이 향상되었다.
- **SDM 도입 효과**: SDM 보조 작업이 추가되었을 때 세만틱 특징 표현력이 풍부해짐을 확인하였다.
- **가중치 전략**: 단순 일관성 손실($\mathcal{L}_{gc}$)보다 지수 가중치를 적용한 손실($\mathcal{L}_{dwgc}$)을 사용했을 때 가장 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 가장 취약한 부분인 '경계 영역'의 학습을 위해 기하학적 정보(SDM)를 도입하고, 이를 가중치 시스템으로 연결한 점이 매우 영리한 설계이다. 단순히 데이터에 노이즈를 섞는 기존의 Consistency training과 달리, **세만틱 관점(Segmentation)과 기하학적 관점(SDM)이라는 서로 다른 도메인 간의 일관성을 강제**함으로써 모델이 더 고차원적인 특징을 학습하도록 유도하였다.

또한, 이중 뷰 네트워크를 통해 모델의 다양성을 확보한 것은 앙상블 효과와 유사하게 예측의 분산을 줄이는 효과를 가져왔다. 다만, 하이퍼파라미터 $\rho$의 값에 따라 성능 변화가 존재하며, 최적의 $\rho$를 찾기 위한 실험적 과정이 필요하다는 점은 실무 적용 시 고려해야 할 사항이다.

결론적으로 본 연구는 레이블이 극도로 부족한 환경에서도 기하학적 사전 지식(Prior)을 네트워크 구조와 손실 함수에 효율적으로 녹여냄으로써, 의료 영상 분할의 실용성을 크게 높였다고 평가할 수 있다.

## 📌 TL;DR

- **핵심 기여**: SDM(Signed Distance Map)을 활용한 기하학적 인지 기반 준지도 학습 프레임워크 제안.
- **주요 장치**:
    1. 경계 영역에 집중하는 **지수 가중치 일관성 손실** 도입.
    2. 불확실성을 낮추기 위한 **이중 뷰(Dual-view) 디코더** 구조 설계.
- **성과**: 좌심방 및 뇌종양 데이터셋에서 SOTA 달성, 특히 경계 정밀도 지표(ASD, 95HD)에서 압도적 성능 및 적은 데이터(10%)에 대한 높은 강건성 입증.
- **의의**: 의료 영상의 특성(경계 모호성, 데이터 부족)을 기하학적 제약 조건으로 해결하여 준지도 학습의 새로운 방향성을 제시함.
