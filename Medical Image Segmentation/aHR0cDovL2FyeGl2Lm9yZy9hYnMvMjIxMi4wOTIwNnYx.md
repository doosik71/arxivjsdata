# Segmentation Ability Map: Interpret deep features for medical image segmentation

Sheng He, Yanfang Feng, P. Ellen Grant, Yangming Ou (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)을 위해 널리 사용되는 심층 합성곱 신경망(Deep CNN)의 내부 동작 원리를 이해하고 해석하는 문제를 다룬다. 일반적으로 이러한 모델들은 최종 출력층의 결과만을 활용하며, 중간 은닉층에서 추출되는 심층 특징(Deep Features)들이 구체적으로 어떤 역할을 하는지, 그리고 입력 이미지에서 최종 출력에 이르기까지 분할 능력이 어떻게 전이되는지는 명확히 밝혀지지 않았다.

특히, 의료 분야에서는 모델의 결정 근거를 이해하는 설명 가능한 AI(Explainable AI) 시스템이 필수적이다. 따라서 본 연구의 목표는 심층 특징의 분할 능력을 정량적으로 측정하고 시각화할 수 있는 도구를 개발하여, 데이터 과학자에게는 모델 개선의 통찰을 제공하고, 최종 사용자(의료진)에게는 모델의 결과를 신뢰할 수 있는 근거를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Prototype Segmentation (ProtoSeg)** 방법론을 통해 임의의 심층 특징 맵(Feature Map)으로부터 이진 분할 맵을 생성하고, 이를 기반으로 해당 특징의 분할 능력을 정량화하는 것이다.

중심적인 설계 직관은 "동일한 클래스(대상 또는 배경)에 속하는 픽셀들의 특징 벡터 간 거리는 가깝고, 서로 다른 클래스 간의 거리는 멀어야 한다"는 가정에 기반한다. 이를 위해 모델의 출력값을 가이드로 사용하여 각 클래스의 대표값인 프로토타입(Prototype)을 계산하고, 각 픽셀을 가장 가까운 프로토타입에 할당함으로써 특징 맵 자체의 분할 능력을 평가하는 **Segmentation Ability Map (SAM)**과 이를 수치화한 **SA score**를 제안한다.

## 📎 Related Works

기존의 딥러닝 모델 해석 방법론으로는 Attention 메커니즘, Network Dissection, Class Activation Map (CAM) 등이 있다.

- **Attention Mechanism**: 특정 영역을 강조하여 시각화할 수 있으나, 각 특징의 중요도를 정량적으로 측정하거나 픽셀 단위의 결정 과정을 이해하는 데 한계가 있다.
- **CAM 및 Grad-CAM**: 이미지 분류(Image-to-Class) 문제에서는 효과적이지만, 의료 영상 분할과 같은 이미지-투-이미지(Image-to-Image) 문제에 직접 적용하기 어렵다. 분류 문제는 이미지 전체를 하나의 클래스로 판단하지만, 분할 문제는 각 픽셀마다 독립적인 레이블을 결정해야 하기 때문이다.

본 논문은 이러한 기존 방식들이 '절대적인 활성화 값'에 집중하는 반면, 분할 문제에서는 대상과 배경을 얼마나 잘 분리하는가 하는 **'분리 능력(Separation Ability)'**이 더 중요하다는 점을 강조하며 차별점을 둔다.

## 🛠️ Methodology

### 1. Prototype Segmentation (ProtoSeg)

ProtoSeg는 추가적인 파라미터가 필요 없는(Parameter-free) 플러그 앤 플레이 모듈로, 특정 심층 특징 $\mathbf{f}$가 주어졌을 때 다음과 같은 절차로 분할 맵을 생성한다.

**가. 프로토타입 계산**
신경망의 초기 출력 맵 $B \in [0, 1]$ (0: 배경, 1: 대상)를 마스크로 사용하여, 대상 영역($c_t$)과 배경 영역($c_b$)의 중심점인 프로토타입을 계산한다.
$$c_t = \frac{\sum (B_i \cdot f_i)}{\sum B_i}, \quad c_b = \frac{\sum ((1 - B_i) \cdot f_i)}{\sum (1 - B_i)}$$
여기서 $f_i$는 픽셀 $i$에서의 특징 벡터이다.

**나. 픽셀 분류 및 SAM 생성**
각 픽셀 $f_i$와 두 프로토타입 간의 거리(Euclidean distance)를 기반으로 확률을 계산한다. $p(f_i, c_t) > p(f_i, c_b)$인 경우 해당 픽셀을 대상으로 분류하며, 이를 통해 최종적으로 이진 분할 맵인 $S_f$ (SAM)를 얻는다.

### 2. Segmentation Ability (SA) Score

생성된 SAM $S_f$가 실제 정답(Ground-truth, $G$)과 얼마나 유사한지를 측정하기 위해 Dice score를 사용하며, 이를 **SA score**라고 정의한다.
$$\text{SA Score}(S_f, G) = \frac{2|S_f \cap G|}{|S_f| + |G|}$$
SA score가 높을수록 해당 심층 특징이 대상과 배경을 잘 구분하는 강력한 분별력을 가졌음을 의미한다.

### 3. Offline 및 Online ProtoSeg

- **Offline ProtoSeg**: 이미 학습된 네트워크의 은닉층 특징들을 추출하여 사후적으로 분할 능력을 해석하는 방식이다.
- **Online ProtoSeg**: ProtoSeg 과정이 미분 가능함을 이용하여, SA score를 학습 손실 함수(Loss function)에 추가하는 방식이다.
$$\mathcal{L} = \mathcal{L}_g + \sum \frac{\text{SA Score}}{N}$$
이를 통해 모델 학습 과정에서 중간 특징들의 분할 능력을 직접적으로 향상시킬 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: BraTS(뇌종양), ISIC(피부 병변), COVID-19(CT), Prostate(전립선), Pancreas(췌장) 등 5개 데이터셋 사용.
- **모델**: 표준 U-Net 구조를 사용하였으며, 18개의 컨볼루션 레이어로 구성됨.
- **평가 지표**: Dice score.

### 2. 주요 결과

- **레이어별 분할 능력 전이**: SA score를 분석한 결과, 초기 레이어(Early layers) $\rightarrow$ 심층 추상 레이어(Deep abstract layers) $\rightarrow$ 후기 레이어(Late layers) 순으로 분할 능력이 전이됨을 확인하였다. 특히 심층 추상 레이어에서는 해상도가 낮아 SA score가 일시적으로 낮게 나타나지만, 대상 영역의 대략적인 위치(Semantic context)를 파악하는 역할을 수행한다.
- **Online ProtoSeg의 효과**: SA score를 손실 함수에 포함하여 학습시킨 결과, 최종 출력의 정확도를 떨어뜨리지 않으면서도 중간 특징들의 SA score를 유의미하게 높여 모델의 해석 가능성을 증대시켰다.
- **마지막 레이어의 유닛 분석**: 마지막 레이어의 64개 유닛 중 실제 높은 SA score를 가진 '활성 유닛(Active units)'은 약 20%~50%에 불과했다. 이는 U-Net의 마지막 레이어가 과도하게 설계(Oversized)되었음을 시사하며, 네트워크 프루닝(Pruning)의 가능성을 보여준다.
- **출력 품질 추정 (Confidence Score)**: 정답이 없는 테스트 이미지에 대해 마지막 두 레이어 유닛들의 평균 SA score $\mu(x)$를 계산하였을 때, 이 값이 실제 Dice 정확도와 강한 상관관계를 보였다. 이를 통해 신뢰도가 낮은(SA score가 낮은) 결과물을 자동으로 선별하여 전문가의 재검토를 요청하는 'Human-in-the-loop' 워크플로우가 가능함을 입증하였다.

## 🧠 Insights & Discussion

### 1. U-Net의 동작 원리 재해석

본 연구는 U-Net이 일종의 **'디노이징 모델(Denoise model)'**로 동작한다는 통찰을 제공한다. 인코더 경로(Down-sampling)에서는 입력 이미지의 노이즈를 제거하고 전역적 문맥(Global context)을 통해 대상을 대략적으로 위치시키며, 디코더 경로(Up-sampling)에서는 초기 레이어의 세부 특징과 결합하여 정교한 형태를 복원한다.

### 2. 데이터셋 특성 파악

입력 이미지의 강도/색상 값만으로 직접 ProtoSeg를 수행하여 계산한 SA score($S_x$)와 모델 출력($B$)을 비교함으로써 데이터셋의 난이도를 평가하였다. 예를 들어 BraTS와 ISIC는 입력 값만으로도 어느 정도 분리가 가능한 '쉬운' 데이터셋인 반면, COVID-19 데이터셋은 모델을 통한 성능 향상 폭이 적은 '어려운' 데이터셋임을 정량적으로 확인하였다.

### 3. 한계점 및 논의

- **지표의 다양성**: 현재는 Dice score만을 사용했으나, 향후 Jaccard index나 Sensitivity 등 다양한 지표로 확장할 필요가 있다.
- **차원 확장**: 2D 네트워크 위주로 실험이 진행되었으며, 3D 의료 영상 네트워크로의 확장이 필요하다.
- **가중치와의 관계**: ProtoSeg는 특징의 '분리 능력'에 집중하므로, 높은 SA score를 가진 유닛이 반드시 큰 학습 가중치를 가진다는 보장은 없다.

## 📌 TL;DR

본 논문은 심층 신경망의 은닉 특징들을 정량적으로 해석하기 위한 파라미터 없는 모듈인 **ProtoSeg**와 **SA score**를 제안한다. 이를 통해 U-Net의 레이어별 분할 능력 전이 과정을 시각화하고, 모델의 마지막 레이어에 상당한 중복성(Redundancy)이 존재함을 밝혀냈다. 특히, 정답이 없는 데이터에 대해서도 특징 맵의 일관성을 통해 예측 결과의 신뢰도를 추정할 수 있어, 의료 현장에서 고위험 예측치를 선별하는 신뢰성 있는 AI 시스템 구축에 기여할 수 있다.
