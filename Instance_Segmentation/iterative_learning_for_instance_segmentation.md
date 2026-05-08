# ITERATIVE LEARNING FOR INSTANCE SEGMENTATION

Tuomas Sormunen, Arttu Läämsä, Miguel Bordallo López (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Instance Segmentation 모델을 학습시키기 위해 요구되는 방대한 양의 라벨링된 데이터 확보 문제이다. Instance Segmentation은 이미지 내의 개별 객체를 탐지하고 픽셀 단위로 분할하는 작업으로, 고성능의 딥러닝 모델을 구축하기 위해서는 정교한 Ground Truth 마스크가 필요하다. 그러나 이러한 어노테이션(Annotation) 작업은 매우 많은 시간과 비용이 소요되며, 특히 산업 현장의 visual inspection과 같은 특수 도메인에서는 전문 인력이 필요하여 병목 현상이 발생한다.

따라서 본 연구의 목표는 최소한의 인간 개입만으로도 대량의 학습 데이터를 생성하고 모델을 고도화할 수 있는 semi-supervised 기반의 few-shot self-learning 반복 학습 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 매우 적은 수의 초기 데이터만으로 학습을 시작하여, 모델이 스스로 생성한 예측 결과 중 신뢰도가 높은 데이터를 다시 학습 데이터로 사용하는 '반복적 자가 학습(Iterative Self-learning)' 구조를 제안한 것이다.

전체 프로세스는 소량의 데이터로 초기 모델을 만드는 Bootstrapping 단계에서 시작하여, 모델이 추론한 결과 중 특정 Confidence Threshold를 넘는 인스턴스만을 선택해 새로운 Ground Truth로 정의하고 이를 통해 다시 모델을 학습시키는 루프를 형성한다. 이를 통해 사람이 일일이 마스크를 그리지 않고도 데이터셋을 스스로 확장하며 성능을 높이는 메커니즘을 구현하였다.

## 📎 Related Works

논문에서는 Instance Segmentation의 대표적인 모델로 Mask R-CNN과 그 기반이 되는 Fast R-CNN을 언급한다. Mask R-CNN은 Region-based CNN을 통해 객체의 Bounding Box를 검출하고 그 내부에서 Segmentation Mask를 생성하는 효율적인 구조를 가지고 있으며, Detectron2와 같은 프레임워크를 통해 구현 및 재학습이 용이하다.

기존의 semi-supervised 또는 iterative learning 접근 방식들이 주로 Bounding Box 생성에 국한되어 있었으며, 각 반복 단계마다 인간이 직접 박스와 라벨을 수정해야 하는 Human-in-the-loop 방식에 의존했다는 점을 한계로 지적한다. 본 논문은 이러한 인간의 개입을 최소화하고, Bounding Box를 넘어 완전한 Instance Mask를 생성하는 반복 학습 시스템을 제안함으로써 기존 연구와 차별화를 둔다.

## 🛠️ Methodology

### 시스템 구조 및 파이프라인

제안된 시스템은 데이터셋을 세 가지 파티션으로 나누어 관리한다.

1. **Bootstrapping set**: 사용자가 매우 적은 수의 인스턴스에 대해서만 어노테이션을 수행한 초기 이미지 집합이다.
2. **Training set**: 라벨이 없는 대량의 이미지 집합으로, 이후 반복 학습 과정에서 모델이 생성한 pseudo-label이 추가된다.
3. **Testing set**: 모델의 성능 평가를 위해 완전히 어노테이션된 독립적인 이미지 집합이다.

전체 파이프라인은 다음의 세 단계로 진행된다.

- **Initiation phase**: 대규모 멀티 클래스 데이터셋으로 사전 학습된 모델을 Bootstrapping set을 이용해 Fine-tuning하여 초기 모델을 생성한다.
- **Iterative learning phase**: 생성된 모델을 Training set에 적용하여 추론을 수행한다. 이때 설정된 Confidence Threshold 이상의 결과만을 필터링하여 새로운 Ground Truth로 간주하고, 이를 포함해 모델을 다시 학습시킨다. 이 과정은 정해진 반복 횟수에 도달할 때까지 지속된다.
- **Evaluation phase**: 각 반복 단계 이후 외부 Test set을 통해 모델의 성능을 평가한다.

### 상세 구현 및 학습 절차

- **모델 아키텍처**: Detectron2의 R50 FPN 3x Mask R-CNN 베이스라인을 사용하였으며, COCO 2017 데이터셋으로 사전 학습된 가중치를 사용하였다.
- **학습 방식**: 각 반복 단계에서 모델의 가중치는 이전 단계에서 이어받아 업데이트된다.
- **핵심 하이퍼파라미터**:
  - **Confidence Threshold**: 추론 결과 중 어떤 인스턴스를 다음 단계의 학습 데이터로 포함할지 결정하는 기준이다.
  - **Epochs per iteration**: 각 반복 주기 내에서 수행되는 학습 횟수이다.
- **평가 지표**: $\text{AP}_{75}$ (Average Precision at 75% IoU)와 $\text{AR}_{75}$ (Average Recall at 75% IoU)를 사용하여 정량적 성능을 측정한다.

## 📊 Results

### 실험 설정

- **Coffee Dataset**: 현미경으로 촬영된 커피 입자 이미지로, 모양이 불규칙하고 서로 겹쳐 있어 난이도가 높다. Bootstrapping set은 단 2장의 이미지(일부 어노테이션 포함)로 구성되었다.
- **Fruit Dataset**: 대추야자, 무화과, 헤이즐넛의 3개 클래스로 구성되며, 금박지에 싸인 유사한 형태의 방해 객체들이 포함되어 있다.

### 주요 결과 및 분석

1. **Bootstrapping 데이터 양의 영향**: Coffee 데이터셋 실험 결과, 단 1개의 어노테이션만으로 학습을 시작해도 반복 학습을 통해 다른 조건과 대등한 성능에 도달할 수 있음을 확인하였다.
2. **Epoch 수의 영향**: 각 iteration당 Epoch 수가 너무 적으면 새로운 인스턴스를 탐지하지 못하고, 너무 많으면 이미 본 데이터에 과적합(Overfitting)되어 일반화 성능이 떨어진다.
3. **Confidence Threshold의 영향**:
    - Coffee 데이터셋(단일 클래스)에서는 낮은 Threshold ($0.25$)가 높은 $\text{AP}_{75}$와 $\text{AR}_{75}$를 보였다. 다만, 너무 낮은 Threshold는 Non-maximum Suppression(NMS) 이후에도 객체 간 중첩이 발생하는 경향이 있었다.
    - Fruit 데이터셋(멀티 클래스)에서는 Threshold가 낮을수록 수렴 속도는 빠르나, 목표 클래스가 아닌 금박지 객체 등을 오분류하여 학습 데이터에 포함시키는 문제가 발생하였다. 따라서 멀티 클래스 환경에서는 더 보수적인(높은) Threshold가 필요함을 시사한다.

## 🧠 Insights & Discussion

본 연구는 매우 적은 초기 데이터만으로도 반복적인 자가 학습을 통해 고품질의 Instance Segmentation 모델을 구축할 수 있음을 입증하였다. 특히 산업용 검사 시스템처럼 데이터 라벨링 비용이 높은 환경에서 매우 유용한 접근 방식이다.

그러나 본 연구에서는 몇 가지 한계점과 논의 사항이 제기된다.
첫째, 최적의 Confidence Threshold와 Epoch 수를 자동으로 결정하는 방법이 부재하며, 이는 데이터셋의 특성과 클래스 수에 따라 수동으로 튜닝해야 하는 Grid Search 과정이 필요하다.
둘째, 모델이 목표 클래스와 유사한 비목표 객체를 탐지하고 이를 다시 학습 데이터로 사용하는 '오류 전파(Error Propagation)' 문제가 발생할 수 있다. 저자는 이를 해결하기 위해 각 반복 단계 이후 유사한 객체들을 그룹화하는 Clustering 모듈을 도입하는 방안을 향후 연구 과제로 제시한다.

## 📌 TL;DR

본 논문은 최소한의 초기 라벨링 데이터(Bootstrapping set)만을 사용하여, 모델이 스스로 생성한 고신뢰도 예측치를 다시 학습 데이터로 활용하는 **반복적 자가 학습(Iterative Self-learning) 프레임워크**를 제안하였다. Mask R-CNN을 기반으로 구현된 이 시스템은 단일 및 다중 클래스 데이터셋에서 라벨링 비용을 획기적으로 줄이면서도 높은 성능의 Instance Segmentation 모델을 생성할 수 있음을 보여주었다. 이 연구는 특히 데이터 확보가 어려운 산업용 시각 검사 분야의 딥러닝 모델 구축에 중요한 기여를 할 가능성이 크다.
