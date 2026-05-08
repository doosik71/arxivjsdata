# Uncertainty Estimation in Instance Segmentation with Star-convex Shapes

Qasim M. K. Siddiqui, Sebastian Starke, and Peter Steinbach (2023)

## 🧩 Problem to Solve

딥러닝 기반의 인스턴스 분할(Instance Segmentation) 모델들은 뛰어난 성능을 보여주고 있으나, 때때로 잘못된 예측을 내놓으면서도 매우 높은 확신도(Confidence)를 보이는 경향이 있다. 이러한 과잉 확신(Overconfidence) 문제는 특히 학습 데이터에 포함되지 않은 새로운 특성을 가진 데이터가 들어오는 open-set 조건에서 두드러지며, 이는 안전성과 신뢰성이 중요한 실무 응용 분야에서 치명적인 결과를 초래할 수 있다.

기존의 불확실성 추정(Uncertainty Estimation) 연구들은 주로 분류(Classification)나 회귀(Regression) 작업에 집중되어 있었으며, 인스턴스 분할 분야, 특히 별 모양의 볼록 다각형(Star-convex shapes)을 다루는 모델에 대한 공간적 확신도(Spatial certainty) 측정 연구는 부족한 실정이다. 따라서 본 논문은 StarDist 모델을 활용하여 인스턴스의 위치 및 형태와 관련된 공간적 확신도를 추정하고, 이를 통해 모델의 예측 신뢰성을 정량적으로 평가하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 인스턴스 분할 모델인 StarDist에 Monte-Carlo (MC) Dropout과 Deep Ensemble 기법을 적용하여 모델의 Epistemic Uncertainty를 추정하는 프레임워크를 제안한 것이다.

가장 중심적인 아이디어는 예측된 인스턴스들의 집합을 클러스터링하여 각 인스턴스의 '공간적 확신도(Spatial Certainty)'와 '분할 확신도(Fractional Certainty)'를 각각 계산하고, 이 둘을 결합한 '하이브리드 확신도(Hybrid Certainty)'를 통해 모델의 보정(Calibration) 성능을 높이는 것이다. 특히, 기존의 마스크 기반 클러스터링 방식의 계산 복잡도 문제를 해결하기 위해 StarDist의 출력 구조를 직접 활용하는 효율적인 'Radial Approach' 클러스터링 방법을 새롭게 제안하였다.

## 📎 Related Works

딥러닝에서의 불확실성은 데이터 자체의 노이즈로 인한 Aleatoric Uncertainty와 모델의 파라미터 불확실성으로 인한 Epistemic Uncertainty로 나뉜다. 본 논문은 데이터 부족으로 인해 발생하는 Epistemic Uncertainty를 추정하는 데 집중한다.

전통적인 Bayesian Neural Networks는 계산 비용이 매우 높기 때문에, 이를 근사하기 위한 방법으로 테스트 단계에서 Dropout을 적용하는 Monte-Carlo Dropout이 제안되었다. 또한, 하이퍼파라미터 튜닝에 덜 민감하고 더 잘 보정된 결과를 제공하는 비-베이지안 방식인 Deep Ensemble 기법이 대안으로 제시되었다.

인스턴스 분할 분야에서는 Mask-RCNN과 같은 모델에 Dropout 샘플링을 적용하여 불확실성을 추정한 선행 연구(Morrison et al.)가 존재한다. 본 논문은 이러한 접근 방식을 확장하여, 밀집된 인스턴스 검출에 강점이 있는 StarDist 모델에 적용함으로써 기존 연구의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 StarDist 모델을 기반으로 하며, 추론(Inference) 단계에서 $F$번의 순전파(Forward pass)를 수행하여 여러 개의 예측 샘플을 얻는다. 이 샘플들을 클러스터링하여 동일한 객체에 해당하는 예측들을 하나의 그룹으로 묶고, 이 그룹 내의 변동성을 통해 확신도를 측정한다.

### 클러스터링 접근 방식

논문은 두 가지 클러스터링 방법을 제안하고 비교한다.

1. **Pixel Approach**:
    - Non-Maximum Suppression(NMS) 이후의 최종 마스크를 사용한다.
    - 각 예측 인스턴스 간의 Intersection-over-Union (IoU)을 계산하여, $\theta_{IoU} = 0.5$ 이상인 경우 동일한 클러스터로 묶는 Basic Sequential Algorithmic Scheme (BSAS)을 사용한다.
    - 이 방식은 데이터 증가에 따라 계산 복잡도가 이차적으로 증가($O(n^2)$)하는 단점이 있다.

2. **Radial Approach**:
    - StarDist의 Dense Output(객체 확률 $D$와 방사형 거리 $R$)을 직접 활용한다.
    - 먼저 $F$번의 샘플에 대한 평균 Dense Output $\mu_G = \{\mu_D, \mu_R\}$을 구하고, 여기에 NMS를 적용하여 인스턴스의 중심점 $C = \{(x_m, y_m)\}$들을 결정한다.
    - 각 중심점에서 각 샘플의 객체 확률이 임계값 $\theta_d = 0.5$를 넘는 경우 해당 클러스터에 할당한다. 이 방식은 데이터 양에 따라 선형적으로 스케일링되어 훨씬 효율적이다.

### 확신도 정량화 (Certainty Quantification)

클러스터 $O_m$에 대해 다음과 같은 세 가지 지표를 계산한다.

- **Spatial Certainty ($c_{spl}$)**: 모델이 인스턴스의 위치와 형태를 얼마나 일관되게 예측했는지를 측정한다. 클러스터 내의 중간값 예측(Median prediction) $P_m$과 각 샘플 $P_j$ 간의 IoU 평균으로 계산한다.
  $$c_{spl}(O_m) = \frac{1}{|O_m|} \sum_{j=1}^{|O_m|} \text{IoU}(P_j, P_m)$$

- **Fractional Certainty ($c_{frac}$)**: $F$번의 시도 중 해당 인스턴스가 얼마나 자주 검출되었는지를 측정한다.
  $$c_{frac}(O_m) = \frac{|O_m|}{N}$$

- **Hybrid Certainty ($c_{hyb}$)**: 위 두 지표를 결합하여 더 정교한 확신도를 산출한다.
  $$c_{hyb}(O_m) = c_{spl}(O_m) \cdot c_{frac}(O_m)$$

## 📊 Results

### 실험 설정

- **데이터셋**: Bubble(기포 이미지), DSB2018(세포핵 이미지), GlaS(샘 조직 이미지)의 세 가지 데이터셋을 사용하였다.
- **평가 지표**: 확신도 점수가 실제 정확도와 얼마나 일치하는지를 평가하기 위해 Calibration Diagram, Pearson Correlation Coefficient (R), Expected Calibration Error (ECE), Maximum Calibration Error (MCE)를 사용하였다.

### 주요 결과

- **확신도 지표 비교**: 실험 결과, $c_{spl}$이나 $c_{frac}$ 단독 사용보다 $c_{hyb}$를 사용했을 때 Calibration Diagram에서 정체성 함수(Identity function)에 가장 가깝게 나타나, 가장 잘 보정된 확신도 추정이 가능함을 확인하였다.
- **샘플링 기법 비교**: Deep Ensemble 기법이 Monte-Carlo Dropout보다 더 적은 수의 샘플(모델)로도 빠르게 수렴하며, 하이퍼파라미터 튜닝 없이도 더 우수한 보정 성능을 보였다.
- **순전파 횟수 ($F$)의 영향**: $F$가 증가함에 따라 오차가 감소하며 수렴하는 경향을 보였다. MC Dropout의 경우 약 20~30회, Deep Ensemble의 경우 모델 10개 정도에서 수렴이 이루어졌다.
- **최적 조합**: 결론적으로 **Deep Ensemble + Radial Approach ($F=10$)** 조합이 계산 효율성과 보정 성능 면에서 가장 뛰어난 전략임이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 인스턴스 분할에서 불확실성 추정을 위해 단순한 픽셀 기반 접근을 넘어 모델의 특성(방사형 거리 예측)을 활용한 Radial Approach를 제안함으로써 계산 효율성을 획기적으로 개선하였다.

특히, GlaS 데이터셋에서 관찰된 높은 보정 오차는 StarDist 모델이 해당 데이터셋의 형태적 특성과 맞지 않을 때, 모델이 잘못된 예측을 내리면서도 높은 확신도를 가질 수 있음을 보여준다. 이는 단순히 확신도 점수를 믿는 것이 아니라, 그 점수가 얼마나 잘 보정(Calibrated)되었는지를 평가하는 것이 모델의 신뢰성 확보에 얼마나 중요한지를 시사한다.

한계점으로는 MC Dropout의 경우 Dropout rate와 레이어 위치에 따라 보정 성능이 크게 달라지는데, 이를 최적화하기 위한 탐색 비용이 크다는 점이 언급되었다. 이를 해결하기 위해 향후 연구로 Concrete Dropout과 같은 적응형 Dropout 기법의 도입 가능성을 제시하였다.

## 📌 TL;DR

본 연구는 StarDist 모델을 사용하여 별 모양의 볼록 인스턴스 분할 시 발생하는 불확실성을 추정하는 방법을 제안하였다. MC Dropout과 Deep Ensemble을 적용하고, 효율적인 Radial clustering 방식을 통해 공간적/분할 확신도를 계산하는 프레임워크를 구축하였다. 결과적으로 Deep Ensemble과 Radial Approach를 결합한 하이브리드 확신도 점수가 가장 신뢰할 수 있는(잘 보정된) 지표임을 밝혀냈으며, 이는 향후 능동 학습(Active Learning)이나 고위험 의료 영상 분석 등 모델의 신뢰성이 필수적인 분야에 적용될 가능성이 높다.
