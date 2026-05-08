# Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off

Levente Halmosi, Bálint Mohos, and Márk Jelasity (2024)

## 🧩 Problem to Solve

본 논문은 시맨틱 세그멘테이션(Semantic Segmentation) 모델의 적대적 강건성(Adversarial Robustness)을 평가하는 기존 방법론의 불충분함을 지적한다. 이미지 분류(Image Classification) 분야에서는 강건성을 측정하기 위한 신뢰할 수 있는 평가 체계가 확립되어 있으나, 세그멘테이션 분야에서는 여전히 모델의 취약성을 과소평가하는 경향이 있다.

특히 저자들은 기존의 평가 지표인 픽셀 정확도(Pixel Accuracy)와 클래스별 평균 IoU(CmIoU)가 작은 객체의 오분류를 제대로 반영하지 못한다는 점에 주목한다. 이로 인해 실제로는 매우 취약함에도 불구하고, 지표상으로는 강건한 것처럼 보이는 '가짜 강건성' 문제가 발생한다. 따라서 본 연구의 목표는 더 강력하고 다양한 공격 세트를 구성하고, 객체 크기에 따른 편향을 드러낼 수 있는 새로운 평가 지표를 도입하여 기존의 강건한 세그멘테이션 모델들을 재평가하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 세그멘테이션 모델의 강건성을 엄격하게 측정하기 위한 종합적인 평가 프레임워크를 제안한 것이다.

1. **강력한 공격 배터리(Battery of Attacks) 구축**: 기존 문헌의 최강 공격들과 저자들이 새롭게 제안한 PAdam 기반 공격들을 결합하여, 개별 입력 이미지마다 가장 효과적인 공격을 선택하는 앙상블 방식의 평가 체계를 구축하였다.
2. **새로운 공격 방법 제안**: Adam 옵티마이저를 투영 경사 하강법(PGD)에 접목한 PAdam-CE와 PAdam-Cos 공격을 제안하여 기존 공격들이 놓치는 취약점을 찾아냈다.
3. **이미지별 평균 IoU(NmIoU) 지표 도입**: 클래스 단위가 아닌 이미지 단위로 IoU를 평균 내는 $\text{NmIoU}$를 제안하여, 대형 객체는 보호하면서 소형 객체는 희생시키는 모델의 '크기 편향(Size-bias)' 문제를 가시화하였다.
4. **SOTA 모델의 강건성 재검증**: DDC-AT, SegPGD-AT 및 Croce et al.의 모델들이 실제로는 보고된 것보다 훨씬 더 취약하며, 특히 소형 객체에 대해 매우 무력함을 입증하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **적대적 공격(Attacks)**: gradient 기반 공격, Houdini, ALMAProx, DDC, SegPGD, SEA 등 다양한 공격이 제안되었다. 하지만 대부분 단일 공격의 성능에 의존하여 모델의 상한선 강건성을 측정하는 데 한계가 있었다.
- **방어 기법(Defenses)**: 적대적 입력 탐지(Detection)나 다중 작업 학습(Multi-tasking) 등이 제안되었으나, 이는 근본적인 하드 방어(Hard defense)가 아니며 적대적 훈련(Adversarial Training, AT)이 가장 유망한 해결책으로 간주된다.
- **적대적 훈련(Adversarial Training)**: DDC-AT, SegPGD-AT 등이 제안되었으며, 이들은 훈련 과정에서 적대적 샘플을 섞어 사용하여 강건성을 높이려 했다.

### 차별점

본 연구는 새로운 방어 알고리즘을 제안하는 대신, **"평가 방법이 잘못되었다면 방어 성능을 신뢰할 수 없다"**는 관점에서 접근한다. 기존 연구들이 사용한 약한 공격 세트와 크기 편향이 있는 지표를 배제하고, 가장 가혹한 조건에서의 성능을 측정함으로써 실제 배포 시 발생할 수 있는 보안 취약점을 정확히 진단한다.

## 🛠️ Methodology

### 전체 파이프라인

본 연구의 평가 파이프라인은 $\ell_\infty$ 노름 제약 조건 $\epsilon = 8/255$ 하에서 10가지의 다양한 공격을 수행하고, 그중 가장 낮은 성능을 유도하는 공격의 결과를 최종 강건성 지표로 채택하는 방식을 취한다.

### 주요 구성 요소 및 공격 방법

1. **PAdam-CE & PAdam-Cos (제안 방법)**:
    - 기존 PGD가 단순 gradient descent를 사용하는 것과 달리, Adam 옵티마이저를 사용하여 적응적 스텝 사이즈를 제어하는 Projected Adam(PAdam)을 사용한다.
    - **PAdam-CE**: 표준 교차 엔트로피 손실 함수 $\mathcal{L}_{CE}$를 최대화하여 섭동(perturbation)을 생성한다.
    - **PAdam-Cos**: 정답 레이블과 모델 출력 로짓(logit) 간의 코사인 유사도($\text{CosSim}$)를 최소화하는 방향으로 섭동을 생성한다.

2. **SEA Attack Set**: Balanced CE, Masked CE, Jensen-Shannon Divergence, Masked Spherical Loss 기반의 4가지 공격을 포함한다.

3. **Clipped Minimum Perturbation Attacks**: ALMAProx, DAG, PDPGD와 같이 최소 섭동을 찾는 공격들을 포함하며, 결과값을 $\Delta_\epsilon$ 범위 내로 클리핑하여 사용한다.

### 성능 측정 지표 (mIoU Aggregation)

본 논문은 IoU를 집계하는 두 가지 방식을 비교하여 강건성을 분석한다.

- **클래스별 집계 mIoU ($\text{CmIoU}$)**: 모든 이미지의 TP, FP, FN을 먼저 합산한 후 클래스별 IoU를 계산하고 평균을 낸다. (대형 객체가 지표를 지배함)
$$\text{CmIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{\sum_{n=1}^{N} \text{TP}_{cn}}{\sum_{n=1}^{N} \text{FP}_{cn} + \text{TP}_{cn} + \text{FN}_{cn}}$$

- **이미지별 집계 mIoU ($\text{NmIoU}$)**: 각 이미지 내에서 클래스별 IoU를 먼저 계산하고, 이를 이미지별로 평균 낸 후 다시 전체 이미지에 대해 평균을 낸다. (소형 객체의 영향력이 공평하게 반영됨)
$$\text{NmIoU} = \frac{1}{NC} \sum_{n=1}^{N} \sum_{c=1}^{C} \frac{\text{TP}_{cn}}{\text{FP}_{cn} + \text{TP}_{cn} + \text{FN}_{cn}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC 2012, Cityscapes.
- **모델**: PSPNet, DeepLabv3, UPerNet (ConvNeXt backbone).
- **비교군**: 일반 모델(Normal), DDC-AT, SegPGD-AT, SEA-AT 및 훈련 시 적대적 샘플 비율(50% vs 100%)을 달리한 모델들.

### 주요 결과

1. **50% 적대적 훈련의 무력함**: 훈련 데이터의 50%만 적대적 샘플로 구성한 모델(DDC-AT, SegPGD-AT 기본 설정)은 본 연구의 통합 공격 세트 앞에서 $\text{mIoU}$가 거의 0에 수렴하며, 사실상 강건성이 전혀 없음을 보였다.
2. **100% 적대적 훈련의 한계**: 100% 적대적 샘플을 사용한 모델들은 픽셀 정확도 면에서는 어느 정도 강건성을 보였으나, $\text{NmIoU}$ 관점에서는 여전히 취약했다. 또한, 이는 일반 모델 대비 깨끗한 이미지(clean samples)에서의 성능 저하라는 tradeoff를 수반한다.
3. **SEA-AT 모델의 크기 편향(Size-bias)**: SEA-AT 모델은 $\text{CmIoU}$는 높게 유지되지만 $\text{NmIoU}$는 급격히 떨어진다. 이는 모델이 큰 객체는 잘 보호하지만, 작은 객체는 쉽게 삭제하거나 오분류하는 경향이 있음을 의미한다.
4. **공격의 다양성 필요성**: 특정 하나의 공격이 모든 모델을 압도하지 않는다. 모델마다 취약한 공격이 다르며, 예를 들어 일반 모델에는 PAdam-Cos가, 강건한 모델에는 SEA 공격들이 더 효과적이었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 단순히 "새로운 공격"을 만든 것이 아니라, **"어떻게 평가해야 하는가"**에 대한 방법론적 가이드를 제시했다는 점에서 가치가 크다. 특히 $\text{NmIoU}$라는 지표를 통해 딥러닝 모델이 강건성을 학습할 때 발생하는 '쉬운 길 찾기(shortcut learning)'—즉, 큰 객체만 보호하고 작은 객체를 포기하는 행위—를 정량적으로 드러냈다.

### 한계 및 비판적 해석

- **비용 문제**: 저자들은 다양한 하이퍼파라미터 조합이나 백본 네트워크의 영향을 모두 실험하지 못했음을 명시했다. 이는 적대적 훈련의 연산 비용이 매우 높기 때문이며, 향후 더 효율적인 훈련 방법이 제시된다면 더 넓은 범위의 분석이 가능할 것이다.
- **배경 클래스의 영향**: PASCAL VOC 데이터셋에서 배경(background)이 전체 픽셀의 74%를 차지하여 지표를 왜곡할 가능성이 있다. 저자들은 배경을 제외한 foreground 전용 지표를 함께 제시하여 이 문제를 완화하려 했다.

## 📌 TL;DR

본 논문은 시맨틱 세그멘테이션의 강건성 평가가 기존에는 너무 관대했음을 입증하고, **PAdam 기반의 새로운 공격**과 **이미지 단위의 $\text{NmIoU}$ 지표**를 통해 SOTA 모델들의 실제 취약성을 폭로하였다. 특히, 기존의 강건한 모델들이 소형 객체를 희생시키며 지표를 높이는 **크기 편향(Size-bias)**이 있음을 밝혀냈으며, 진정한 강건성을 달성하기 위해서는 다양하고 강력한 공격 앙상블을 통한 엄격한 검증이 필수적임을 시사한다.
