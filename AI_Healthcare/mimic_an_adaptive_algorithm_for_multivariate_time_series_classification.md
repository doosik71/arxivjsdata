# Mimic: An adaptive algorithm for multivariate time series classification

Yuhui Wang, Diane J. Cook (2020)

## 🧩 Problem to Solve

다변량 시계열 분류(Multivariate Time Series Classification, MTSC) 분야에서는 예측 성능과 모델의 해석 가능성(Interpretability) 사이의 상충 관계가 존재한다. 딥러닝 기반의 방법론들은 높은 예측 정확도를 제공하지만, 내부 동작 과정을 알 수 없는 '블랙박스(Black-box)' 특성으로 인해 투명성이 부족하다. 반면, Shapelet 기반의 방법론들은 시계열 데이터에서 대표적인 패턴을 찾아 시각화할 수 있어 해석 가능성이 높지만, 복잡한 다변량 데이터에서는 예측 성능이 떨어지는 경향이 있다.

본 논문의 목표는 강력한 블랙박스 분류기의 예측 성능을 유지하면서도, Shapelet 분류기처럼 시각적인 해석 능력을 제공하는 새로운 알고리즘인 Mimic을 제안하는 것이다. 즉, 기존의 고성능 분류기를 "모방(Mimic)" 하여 그 성능을 유지하면서, 동시에 사용자가 이해할 수 있는 시각적 표현을 생성하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 사전 학습된(Pre-trained) 블랙박스 분류기가 의사결정을 내릴 때 어떤 입력 영역에 영향을 많이 받는지를 분석하여, 이를 시각적인 패턴인 MimicShape로 추출하는 것이다.

주요 기여 사항은 다음과 같다.

1. **투명성을 희생하지 않는 고정확도 방법론 제안**: 블랙박스 모델의 성능을 모방하면서도 해석 가능한 모델을 생성한다.
2. **적응형(Adaptive) 구조**: 특정 모델에 종속되지 않고 다양한 유형의 시계열 분류기와 결합하여 사용할 수 있는 범용적인 구조를 가진다.
3. **다양한 데이터셋 및 분류기에 대한 효과성 입증**: 26개의 다변량 시계열 데이터셋을 통해 Mimic이 다양한 분류기의 성능을 정확하게 모방하고 시각화할 수 있음을 보였다.

## 📎 Related Works

논문에서는 MTSC를 위한 기존 접근 방식들을 다음과 같이 분류하여 설명한다.

1. **Shapelet 기반 방법론**: 시계열 데이터에서 클래스를 가장 잘 대표하는 부분 시퀀스(Subsequence)인 Shapelet을 찾는 방식이다. Ultra Fast Shapelets (UFS), Generalized Random Shapelet Forests (gRSF), Shapelet Transform Classifier (STC) 등이 있으며, 시각적 해석이 가능하지만 다변량 데이터에서 성능이 불안정하거나 정확도가 낮다는 한계가 있다.
2. **사전 기반(Dictionary-based) 방법론**: sliding window를 통해 부분 시퀀스를 추출하고 이를 단어(Word) 형태로 변환하여 분류하는 방식이다. SMTS, WEASEL-MUSE 등이 이에 해당하며, 투명성이 부족하다는 단점이 있다.
3. **거리/주파수/구간 기반 방법론**: KNN-DTW(거리 기반), TSF(구간 기반), RISE(주파수 기반) 등이 있다. KNN-DTW는 노이즈에 취약하고 시간/공간 복잡도가 높다는 한계가 있다.
4. **딥러닝 기반 방법론**: LSTM, CNN, ResNet, InceptionTime 등이 있으며, 특히 InceptionTime은 매우 높은 정확도를 보이지만 전형적인 블랙박스 모델로 내부 동작을 이해하기 어렵다.

Mimic은 이러한 기존 연구들의 한계, 즉 '성능은 좋지만 해석이 안 되거나(DL)', '해석은 되지만 성능이 낮은(Shapelet)' 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

Mimic 알고리즘은 크게 네 가지 구성 요소로 이루어진다: **Random Mask**, **Pre-trained Classifier**, **Map Constructor**, 그리고 **MimicShape Generator**. 전체 과정은 입력 데이터에 무작위 마스크를 씌워 분류기의 반응을 살피고, 이를 통해 중요도 맵(Importance Map)을 생성한 뒤 최종적으로 대표 패턴인 MimicShape를 추출하는 순서로 진행된다.

### 상세 단계 및 원리

**1. 데이터 정규화 (Normalization)**
입력 데이터 $\mathcal{X}$를 $[0, 1]$ 범위로 조정하되, 마스크 필터 연산을 위해 모든 값이 0보다 커야 한다. 따라서 다음과 같이 수정된 Max-Min 정규화를 사용한다.
$$x_{norm} = \frac{x - \min(x) + 1}{\max(x) - \min(x) + 1}$$
이 결과 모든 정규화된 값은 $(0, 1]$ 범위에 존재하게 된다.

**2. 중요도 맵(Importance Map) 생성**
중요도 맵 $I$는 분류기가 의사결정을 내릴 때 어떤 좌표(시간 및 변수)가 결정적인 영향을 주었는지 나타낸다.

- **Random Masking**: 입력 데이터 $\mathcal{X}$에 이진 필터 $M$을 원소별 곱셈($\odot$)으로 적용하여 일부 좌표만 보존하고 나머지는 차단한다.
- **Confidence Score**: 마스킹된 데이터 $\mathcal{X} \odot M$을 사전 학습된 분류기에 입력하여 클래스 레이블 $y$에 대한 확률 분포 $Pr(\mathcal{X} \odot M, y)$를 얻는다.
- **수학적 정의**: 특정 좌표 $\beta$의 중요도는 좌표 $\beta$가 보존되었을 때의 기대 점수로 정의된다.
$$I_{\mathcal{X}, f}(\beta) = E_M [ Pr(\mathcal{X} \odot M) | M(\beta)=1 ]$$
- **Monte Carlo 추정**: 모든 가능한 마스크를 계산하는 것은 불가능하므로, $N$개의 무작위 마스크 $\{M_1, M_2, \dots, M_N\}$를 샘플링하여 다음과 같이 근사한다.
$$I_{\mathcal{X}, f}(\beta) \approx \frac{1}{E[M] \cdot N} \sum_{i=1}^N Pr(\mathcal{X} \odot M_i) \cdot M_i(\beta)$$
여기서 $Pr(\mathcal{X} \odot M_i)$는 가중치 역할을 하며, 특정 좌표가 포함된 마스크들이 지속적으로 높은 신뢰도 점수를 낼 때 해당 좌표의 중요도가 높아진다.

**3. MimicShape 생성**
생성된 중요도 맵 $I$를 바탕으로 실제 시각화 가능한 패턴을 추출한다.

- **이진화(Binarization)**: 원래의 중요도 맵에서 0이 아닌 값은 모두 1로, 0인 값은 0으로 변환하여 이진 맵 $I_{mimic}$을 만든다.
- **데이터 제약(Constraining)**: 입력 데이터 $\mathcal{X}$에 이진 맵 $I_{mimic}$을 곱하여 중요하지 않은 영역을 제거한 $\mathcal{X}_I$를 생성한다.
$$\mathcal{X}_I = \mathcal{X} \cdot I_{mimic}$$
- **패턴 추출**: $\mathcal{X}_I$에서 0이 아닌 부분 시퀀스들을 분리하고, 이들 사이에서 **Dynamic Time Warping (DTW)** 알고리즘을 적용하여 가장 대표적인 패턴인 MimicShape를 찾아낸다.

## 📊 Results

### 실험 설정

- **데이터셋**: UEA MTSC 아카이브 중 길이가 동일한 26개의 다변량 시계열 데이터셋을 사용하였다.
- **비교 대상**:
  - Shapelet 기반: STC, gRFS
  - 딥러닝 기반: Multivariate LSTM-FCN, InceptionTime (IT)
  - 기타: Decision Tree
- **평가 지표**: 10-fold cross-validation을 통한 분류 정확도(Accuracy).

### 주요 결과

1. **예측 성능 유지**: MimicShape를 적용한 모델의 정확도와 기반이 된 블랙박스 분류기의 정확도 차이는 통계적으로 유의미하지 않았다($p=0.4544$). 즉, Mimic은 성능 저하 없이 블랙박스 모델을 성공적으로 모방한다.
2. **Shapelet 방법론 대비 우위**: 블랙박스 모델을 모방한 MimicShape는 26개 데이터셋 중 24개에서 기존의 Shapelet 기반 알고리즘(STC, gRFS)보다 더 높은 성능을 보였다.
3. **범용성 확인**: 딥러닝 모델뿐만 아니라 Decision Tree와 같은 다른 분류기에도 적용 가능함을 확인하였다. 다만, Decision Tree를 모방했을 때는 성능이 다소 낮게 나타났는데, 이는 Decision Tree가 확률 분포(Distribution)를 생성하지 않기 때문인 것으로 분석된다.

## 🧠 Insights & Discussion

### 강점

본 논문의 가장 큰 강점은 모델 내부의 복잡한 파라미터를 분석하는 대신, **"입력 데이터의 어떤 부분이 결과에 영향을 주는가"**라는 관점에서 접근하여 해석 가능성을 확보했다는 점이다. 이를 통해 어떤 블랙박스 모델이라도 (확률 값을 출력할 수만 있다면) 동일한 프레임워크 내에서 해석 가능한 형태로 변환할 수 있다.

### 한계 및 비판적 해석

- **확률 분포 의존성**: 본 방법론은 분류기가 $Pr(\mathcal{X}, y)$와 같은 확률 분포를 제공해야 한다는 전제가 있다. 따라서 Decision Tree와 같이 하드 레이블(Hard label)만 출력하는 모델에 대해서는 중요도 맵이 모호해져 성능이 떨어진다.
- **계산 비용**: 중요도 맵을 생성하기 위해 수많은 무작위 마스크를 생성하고 분류기를 반복적으로 실행(Monte Carlo sampling)해야 하므로, 추론 단계에서 추가적인 계산 비용이 발생한다.
- **패턴 추출의 단순성**: 중요도 맵을 단순히 이진화하여 DTW로 패턴을 찾는 방식이 실제 블랙박스 모델의 복잡한 비선형적 특징 추출 과정을 완벽하게 대변하는지에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 고성능 블랙박스 시계열 분류기의 예측력을 유지하면서 해석 가능성을 부여하는 **Mimic** 알고리즘을 제안한다. 무작위 마스킹과 중요도 맵(Importance Map)을 통해 모델이 주목하는 입력 영역을 찾아내고, 이를 시각적인 **MimicShape**로 추출함으로써 "성능"과 "투명성"이라는 두 마리 토끼를 모두 잡고자 했다. 특히 딥러닝 모델(InceptionTime 등)을 모방했을 때 기존 Shapelet 방법론보다 높은 성능을 보이면서도 해석 가능성을 제공한다는 점에서, 향후 의료나 금융과 같이 신뢰성이 중요한 분야의 시계열 분석에 크게 기여할 가능성이 높다.
