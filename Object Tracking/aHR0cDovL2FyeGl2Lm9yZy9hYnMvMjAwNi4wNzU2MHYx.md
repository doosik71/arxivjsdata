# Accurate Anchor Free Tracking

Shengyun Peng, Yunxuan Yu, Kun Wang, Lei He (2020)

## 🧩 Problem to Solve

본 논문은 비주얼 객체 추적(Visual Object Tracking) 분야에서 기존 Siamese 기반 추적기들이 가진 효율성과 유연성의 한계를 해결하고자 한다.

가장 핵심적인 문제는 기존의 성공적인 Siamese 추적기들이 대부분 **Anchor-based** 방식에 의존한다는 점이다. Anchor-based 방식은 잠재적인 객체 위치를 정의하기 위해 수많은 Anchor(미리 정의된 다양한 크기와 비율의 바운딩 박스)를 생성하고, 각각의 Anchor에 대해 분류(Classification)와 회귀(Regression)를 수행한다. 이 과정은 다음과 같은 문제를 야기한다:

1. **계산 비효율성**: 모든 가능한 Anchor를 열거하고 처리해야 하므로 연산량이 매우 많다.
2. **유연성 부족**: 미리 정의된 Anchor의 형태에 의존하기 때문에, 객체의 가로세로 비율이 비정상적이거나 극단적인 경우 정확한 바운딩 박스를 제안하는 능력이 제한된다.

따라서 본 논문의 목표는 Anchor 없이 객체의 중심점, 오프셋, 크기를 직접 회귀하여 추적 속도를 획기적으로 높이고 정확도를 개선한 **Anchor Free Siamese Network (AFSN)**를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여는 다음과 같이 요약할 수 있다:

1. **최초의 Anchor-Free Siamese 추적기 제안**: 대규모 데이터셋으로 학습된 엔드-투-엔드(end-to-end) 방식의 AFSN을 제안하였다. 객체를 단순히 중심점, tracking offset, object size라는 세 가지 요소로 정의하여 복잡도를 크게 낮추었다.
2. **네트워크 백본(Backbone) 최적화**: 네트워크의 stride, receptive field, group convolution, kernel size 등이 추적 성능에 미치는 영향을 정량적으로 분석하여 최적의 백본 구조를 도출하였다.
3. **성능 및 속도 향상**: 기존의 Anchor-based 추적기들에 비해 훨씬 빠른 추론 속도(최대 425배)를 달성함과 동시에, 대부분의 벤치마크 데이터셋에서 더 높은 정확도를 입증하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Siamese Network 기반 추적기**: SiamFC, SiamRPN, SiamRPN++ 등이 대표적이다. 이들은 템플릿과 검색 영역 간의 유사도를 비교하여 객체를 찾는다. 특히 SiamRPN 계열은 Region Proposal Network(RPN)를 도입해 정확도를 높였으나, 앞서 언급한 Anchor 기반 방식의 비효율성 문제를 그대로 가지고 있다.
2. **Anchor-Free Detection**: CornerNet, CenterNet 등 객체 탐지 분야에서는 이미 Anchor 없이 키포인트(keypoint)나 중심점(center point)을 예측하는 방식이 성공적으로 적용되어 효율성과 성능을 입증하였다.

### 기존 방식과의 차별점

기존의 Siamese 추적기들이 "수많은 Anchor 중 어떤 것이 정답인가"를 분류하는 방식이었다면, AFSN은 "객체의 중심이 어디이며, 그 크기가 얼마인가"를 직접 회귀하는 방식을 채택한다. 이를 통해 후처리를 최소화하고 단 한 번의 추론으로 바운딩 박스를 결정함으로써 효율성을 극대화하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

AFSN은 특징 추출을 위한 Siamese 네트워크와 이를 기반으로 한 세 가지 출력 브랜치(Classification, Offset, Scale)로 구성된다. 템플릿 이미지($z$)와 검색 이미지($x$)가 동일한 가중치를 공유하는 네트워크를 통과한 후, Correlation 연산을 통해 최종 결과 맵을 생성한다.

### 2. 주요 구성 요소 및 상세 설명

#### A. Bounding Box Center (중심점 예측)

객체의 중심을 찾기 위해 classification branch를 통해 score map $Y$를 생성한다. 중심점의 정답 라벨은 단순한 점이 아니라 2차원 정규 분포(Gaussian distribution)를 따른다. 이는 중심에서 멀어질수록 값이 낮아지게 하여 학습의 안정성을 높이기 위함이다.

- **손실 함수**: Focal Loss를 사용하여 foreground와 background 사이의 불균형을 해소한다.
$$L_{cls} = -\frac{1}{N} \sum \begin{cases} (1-\hat{Y}_{xyk})^\alpha \log(\hat{Y}_{xyk}) & \text{if } Y_{xyk} = 1 \\ (1-Y_{xyk})^\beta (\hat{Y}_{xyk})^\alpha \log(1-\hat{Y}_{xyk}) & \text{if } Y_{xyk} = 0 \end{cases}$$

#### B. Tracking Offset (추적 오프셋)

네트워크의 stride(본 논문에서는 8)로 인해 발생하는 해상도 손실을 보완하기 위해 tracking offset을 예측한다. 이는 score map의 각 지점에서 실제 객체 중심과의 미세한 거리 차이를 보정하는 역할을 한다.

- **손실 함수**: $L_1$ loss를 사용하여 예측된 오프셋과 실제 오프셋 간의 차이를 최소화한다.

#### C. Scale Estimation (크기 예측)

객체의 너비($w$)와 높이($h$)를 직접 예측한다. 이때 예측값이 항상 양수가 되도록 지수 함수($e^\alpha, e^\beta$) 형태를 사용한다.

- **손실 함수**: 바운딩 박스 중심에서의 $L_1$ loss를 사용한다.
$$L_{scl} = \frac{1}{N} \sum_{k} [|\hat{\alpha}_{P_k} - \alpha_k| + |\hat{\beta}_{P_k} - \beta_k|]$$

### 3. 전체 학습 및 추론 절차

- **전체 손실 함수**: 세 가지 손실을 가중치 $\lambda$로 조절하여 합산한다.
$$\text{loss} = L_{cls} + \lambda_{off} L_{off} + \lambda_{scl} L_{scl}$$
- **추론 과정**: Score map에서 가장 높은 응답 값을 가진 지점을 중심점으로 선택하고, 여기에 예측된 offset을 더하고 예측된 size($w, h$)를 적용하여 최종 바운딩 박스를 산출한다.
$$\text{BBox} = (\hat{x}_k + \delta \hat{x}_k - \hat{w}_k/2, \hat{y}_k + \delta \hat{y}_k - \hat{h}_k/2, \hat{w}_k, \hat{h}_k)$$

### 4. 네트워크 백본 최적화

저자들은 단순한 깊은 네트워크가 추적 성능을 항상 높이지 않는다는 점을 발견하였다.

- **Stride와 Receptive Field**: Receptive field가 너무 크면 세부 특징(색상, 모양)을 잃어 위치 정밀도가 떨어지고, 너무 작으면 강건성(robustness)이 떨어진다. 분석 결과, 템플릿 이미지 크기의 70%~80% 정도의 receptive field가 가장 적절함을 확인하였다.
- **Group Convolution**: 채널을 분리하여 처리하는 group convolution이 연산량을 줄이면서도 추적의 강건성을 높이는 효과가 있음을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB2015, VOT2015, VOT2016, VOT2018, TrackingNet.
- **학습 데이터**: GOT-10k 데이터셋을 사용하여 처음부터 학습(train from scratch).
- **지표**: Precision, Success rate, EAO(Expected Average Overlap), FPS(Frames Per Second).

### 정량적 결과

1. **속도**: Titan Xp GPU 기준 **136 FPS**를 달성하였다. 이는 기존의 최고 성능 Anchor-based 추적기들보다 3배에서 최대 425배까지 빠른 속도이다.
2. **정확도**:
    - **OTB2015**: SiamRPN 대비 Precision 0.93% 증가, Success rate 5.97% 증가.
    - **VOT 시리즈**: VOT2015, 2016에서 EAO 기준 1위를 기록하였다.
    - **VOT2018**: EAO 0.398을 기록하여 대부분의 추적기를 앞섰다. (단, SiamRPN++가 EAO 면에서 4% 더 높았으나, AFSN이 3.9배 더 빠르다.)
    - **TrackingNet**: Precision(0.607)과 Success(0.655) 지표 모두에서 1위를 차지하였다.

### 절제 실험 (Ablation Study)

SiamFC와 SiamRPN의 라벨을 Anchor-free 방식으로 변경하여 실험한 결과, 모델 구조를 바꾸지 않고 라벨링 방식만 변경해도 OTB2015에서 Precision 7.65%, Success 5.15%가 상승하였다. 이는 Anchor-free 설계 자체가 추적 성능 향상에 직접적인 기여를 함을 시사한다.

## 🧠 Insights & Discussion

### 강점

- **효율성의 극대화**: Anchor를 생성하고 분류하는 무거운 과정을 제거함으로써 실시간성을 획기적으로 높이면서도 성능 저하가 없음을 보였다.
- **실제적 백본 분석**: 단순히 최신 모델(ResNet 등)을 가져다 쓰는 것이 아니라, 추적 태스크의 특성(템플릿 고정, 국소적 검색)에 맞는 stride와 receptive field의 트레이드-오프를 분석하여 최적의 구조를 제시하였다.

### 한계 및 비판적 해석

- **SiamRPN++와의 성능 차이**: VOT2018에서 SiamRPN++보다 EAO가 약간 낮게 나타났다. 이는 매우 깊은 네트워크가 주는 특징 추출 능력이 정밀도 면에서 이점이 있음을 보여주지만, 저자들의 주장대로 속도 차이를 고려하면 AFSN의 효율성이 압도적이다.
- **온라인 업데이트 부재**: 본 논문의 모델은 오프라인 학습된 특징을 사용하며 온라인 업데이트를 수행하지 않는다. 따라서 타겟의 급격한 외형 변화가 발생하는 시나리오에서의 한계는 명확히 명시되지 않았으나 잠재적인 약점이 될 수 있다.

## 📌 TL;DR

본 논문은 기존 Siamese 추적기의 고질적인 문제였던 **Anchor 기반 방식의 비효율성과 유연성 부족을 해결하기 위해 최초의 Anchor-Free Siamese Network (AFSN)를 제안**하였다. 객체를 중심점, 오프셋, 크기로 단순화하여 직접 회귀함으로써 **추론 속도를 획기적으로(최대 425배) 향상**시켰으며, 동시에 주요 벤치마크에서 **SOTA급의 정확도**를 달성하였다. 특히 추적 태스크에 최적화된 백본 네트워크 구조 분석을 제공함으로써, 향후 효율적인 딥러닝 기반 추적기 설계에 중요한 가이드라인을 제시한 연구이다.
