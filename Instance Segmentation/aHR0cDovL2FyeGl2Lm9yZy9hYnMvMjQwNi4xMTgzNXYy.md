# OoDIS: Anomaly Instance Segmentation and Detection Benchmark

Alexey Nekrasov, Rui Zhou, Miriam Ackermann, Alexander Hermans, Bastian Leibe, Matthias Rottmann (2024)

## 🧩 Problem to Solve

자율 주행 자동차와 로봇의 안전한 주행을 위해서는 주변 환경에 대한 정밀한 이해가 필수적이다. 그러나 딥러닝 모델의 학습 데이터는 실제 배포 환경에서 나타날 수 있는 모든 다양성을 포함할 수 없으며, 이로 인해 야생 동물이나 비정형 장애물과 같은 Out-of-Distribution (OOD) 객체가 등장했을 때 모델이 이를 학습 데이터 내의 기존 클래스로 오분류하는 문제가 발생한다.

기존의 anomaly segmentation 연구들은 주로 픽셀 단위의 semantic segmentation에 집중해 왔다. 하지만 실제 복잡한 도로 상황에서 여러 개의 이상 객체(예: 무리를 지어 이동하는 양 떼)가 등장할 경우, 개별 객체를 구분하여 인식하는 Instance Segmentation 능력이 없다면 객체의 동역학을 파악하거나 경로 계획(Planning)을 세우는 후속 작업에 한계가 있다.

따라서 본 논문의 목표는 anomaly instance segmentation 및 anomalous object detection 작업을 체계적으로 평가할 수 있는 통합 벤치마크인 **OoDIS**를 제안하고, 기존 방법론들의 성능을 정량적으로 분석하여 해당 분야의 한계를 명확히 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 새로운 모델 제안이 아닌, OOD 객체 인식 분야의 발전을 위한 **표준화된 벤치마크 구축**에 있다.

1. **OoDIS 벤치마크 제안**: 기존의 semantic anomaly segmentation 데이터셋인 FishyScapes L&F, RoadAnomaly21, RoadObstacle21을 확장하여, 개별 인스턴스 마스크와 Bounding Box가 포함된 instance-level 어노테이션을 제공한다.
2. **통합 평가 프로토콜 수립**: instance segmentation 및 object detection 작업에 대해 Cityscapes 및 COCO의 평가 방식을 차용한 Average Precision (AP) 및 Average Recall (AR) 지표를 도입하여 객관적인 비교 기반을 마련하였다.
3. **기존 방법론의 한계 분석**: U3HS, Mask2Anomaly, UGainS 등 최신 anomaly instance segmentation 모델들을 동일한 조건에서 평가하여, 특히 작은 객체 탐지와 정밀한 마스크 생성 능력의 부족함을 입증하였다.

## 📎 Related Works

### 기존 연구 및 한계
- **OOD Classification**: 주로 이미지 분류 수준에서 OOD를 탐지하는 연구가 주를 이루었으나, 픽셀 단위의 정밀한 위치 정보가 필요한 자율 주행 환경에는 적용이 어렵다.
- **Open-set Instance Segmentation**: 학습 과정에서 일부 unknown 객체가 포함되어 있다고 가정하는 경우가 많으나, 실제 anomaly segmentation은 완전히 본 적 없는(completely unseen) 객체가 등장하는 상황을 다루어야 하므로 가정에 차이가 있다.
- **Semantic Anomaly Segmentation**: FishyScapes L&F나 SegmentMeIfYouCan과 같은 데이터셋이 제안되었으나, 이들은 픽셀 단위의 마스크만 제공할 뿐 개별 인스턴스를 구분하는 정보가 없어, 여러 객체가 겹쳐 있거나 무리를 이룬 경우 이를 구분할 수 없다.

### OoDIS의 차별점
OoDIS는 단순히 semantic-level의 이상 탐지를 넘어, 개별 객체 단위의 Bounding Box와 Mask를 제공함으로써- 자율 주행의 안전에 직결되는 '개별 이상 객체의 정밀한 위치 및 형태 파악' 능력을 평가할 수 있게 한다.

## 🛠️ Methodology

### 벤치마크 데이터 구성
OoDIS는 세 가지 기존 데이터셋을 재어노테이션하여 통합하였다.
- **RoadAnomaly21**: 다양한 크기와 위치의 객체가 포함되며, 특히 다수의 인스턴스가 한 장면에 등장하는 경우가 많다.
- **RoadObstacle21**: 상대적으로 작은 크기의 객체들이 포함되어 있다.
- **FishyScapes L&F**: 주로 도로 위의 적치물 등이 포함되어 있으며, 앞의 두 데이터셋의 중간적 특성을 가진다.

데이터는 **Inlier**(Cityscapes 클래스), **Outlier**(이상 객체), **Ignore**(모호한 영역)의 세 가지 클래스로 정의된다.

### 평가 지표 (Metrics)
객체 크기에 따른 편향을 줄이기 위해 $\text{IoU}$ (Intersection over Union) 기반의 $\text{AP}$와 $\text{AR}$을 사용한다.

1. **Average Precision (AP)**: 
   특정 $\text{IoU}$ 임계값 $t\%$에서 정밀도-재현율 곡선의 아래 면적을 $\text{AP}_t$라고 할 때, 최종 $\text{AP}$는 다음과 같이 계산한다.
   $$\text{AP} = \frac{1}{T} \sum_{t \in T} \text{AP}_t \quad (T=\{50, 55, \dots, 95\})$$
   또한, 가장 일반적인 지표인 $\text{AP}_{50}$을 함께 보고한다.

2. **Average Recall (AR)**:
   상위 $k$개의 예측 결과에 대해 $\text{IoU}$ 임계값 세트 $T$에서의 재현율을 평균 내어 계산한다.
   $$\text{AR}_k = \frac{1}{|T|} \sum_{t \in T} \text{REC}_t(k)$$
   본 보고서에서는 $\text{AR}_1, \text{AR}_{10}, \text{AR}_{100}$을 측정하여 모델이 얼마나 많은 이상 객체를 놓치지 않고 찾아내는지 평가한다.

3. **Predictions Per Frame (PPF)**: 
   프레임당 평균 예측 수를 측정하여, 모델이 단순히 많은 예측을 내어 우연히 정답을 맞추려 하는지(over-production) 감시한다.

### 평가 대상 방법론
- **U3HS**: 학습 데이터 분포 밖에서도 일반화 가능한 class-agnostic instance embedding을 학습하고, 불확실성 영역 내에서 이를 클러스터링하여 인스턴스를 구분한다.
- **Mask2Anomaly**: Mask2Former 구조를 수정하여 background 영역의 anomaly score를 생성하고, 연결 성분 분석(Connected Components)과 In-distribution 마스크와의 교집합 분석을 통해 False Positive를 제거한다.
- **UGainS**: RbA 방법론으로 불확실성 영역을 먼저 찾고, 해당 영역에서 샘플링한 점(point)들을 Prompt로 사용하여 SAM(Segment Anything Model)을 통해 정밀한 마스크를 생성한다.

## 📊 Results

### 정량적 결과 분석
실험 결과, 모든 방법론이 anomaly instance segmentation 및 detection 작업에서 낮은 성능을 보였다.

- **Instance Segmentation**: $\text{AP}_{50}$ 기준 모든 데이터셋에서 $47\%$ 미만의 성능을 기록하였다. 특히 **UGainS**가 가장 높은 성능을 보였으나, 절대적인 수치는 여전히 낮다. $\text{RoadAnomaly21}$ 데이터셋이 가장 어려웠는데, 이는 양 떼와 같이 수십 개의 인스턴스가 밀집된 장면이 많기 때문이다.
- **Object Detection**: instance segmentation보다 더 낮은 $\text{AP}$ 수치를 보였다. 이는 Bounding Box의 IoU가 픽셀 마스크의 IoU보다 작은 오차에도 훨씬 민감하게 반응하기 때문이다. 특히 $\text{RoadObstacle21}$에서 성능이 가장 낮았는데, 이는 객체 크기가 매우 작아 정밀한 localization이 어렵기 때문으로 분석된다.

### 객체 크기별 성능 ($\text{AP}, \text{AP}_{50}$)
- **Small (< 1,000 pixels)**: 모든 모델이 매우 낮은 성능을 보였다. 특히 $\text{UGainS}$와 $\text{Mask2Anomaly}$ 사이의 성능 격차가 가장 크게 나타난 구간이다.
- **Medium (1,000 - 10,000 pixels)**: $\text{UGainS}$와 $\text{Mask2Anomaly}$가 유사한 성능을 보였다.
- **Large (> 10,000 pixels)**: $\text{Mask2Anomaly}$가 $\text{UGainS}$보다 다소 우세한 경향을 보였다.

### 정성적 결과 분석
- **U3HS**: 이상 객체의 위치는 어느 정도 파악하지만, 정밀한 마스크(Mask)를 생성하는 능력이 매우 부족하다.
- **Mask2Anomaly**: 마스크의 품질은 좋으나, 일부 객체를 놓치거나 여러 인스턴스를 하나의 세그먼트로 병합하는 오류가 발생한다.
- **UGainS**: 많은 이상 객체를 정확하게 분리하고 마스크를 생성하지만, False Positive(오탐지)가 매우 많이 발생한다는 단점이 있다.

## 🧠 Insights & Discussion

### 강점 및 가능성
본 연구는 단순히 새로운 모델을 제안하는 대신, 학계에 부족했던 **인스턴스 레벨의 OOD 평가 기준**을 제시하였다는 점에서 큰 가치가 있다. 특히 객체 크기별 분석을 통해, 현재의 OOD 탐지 기술이 '원거리의 작은 객체'를 인식하는 데 치명적인 약점이 있음을 정량적으로 증명하였다.

### 한계 및 비판적 해석
1. **성능의 전반적 저하**: 평가된 모든 최신 모델들이 $\text{AP}_{50}$ $50\%$를 넘지 못했다는 점은, 현재의 anomaly instance segmentation 접근 방식들이 실질적인 자율 주행 환경에 적용되기에는 아직 초기 단계임을 시사한다.
2. **Open-vocabulary 모델의 한계**: 논문에서 언급되었듯이 Grounding DINO와 같은 최신 open-vocabulary 탐지기조차 여전히 개선의 여지가 많으며, 특히 안전이 직결된 환경에서의 정밀한 localization은 여전히 미해결 과제이다.
3. **데이터셋의 특성**: $\text{RoadAnomaly21}$의 밀집된 객체 상황이나 $\text{RoadObstacle21}$의 소형 객체 상황은 모델에게 서로 다른 종류의 챌린지를 제공하며, 이를 동시에 해결할 수 있는 일반화된 방법론이 필요하다.

## 📌 TL;DR

본 논문은 자율 주행의 안전성을 위해 필수적인 **이상 객체 인스턴스 분할 및 탐지(Anomaly Instance Segmentation & Detection)**를 위한 통합 벤치마크 **OoDIS**를 제안한다. 기존의 semantic-level 데이터셋을 확장하여 개별 객체 단위의 마스크와 Bounding Box를 제공하며, 이를 통해 최신 모델들이 특히 **작은 객체 탐지와 정밀한 마스크 생성**에 취약함을 밝혀냈다. 이 연구는 향후 OOD 객체 인식 연구가 나아가야 할 방향(소형 객체 인식 개선 및 정밀 localization)을 제시하며, 자율 주행 시스템의 신뢰성을 높이는 데 중요한 기초 자료가 될 것이다.