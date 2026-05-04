# Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels

Stine Hansen, Srishti Gautam, Robert Jenssen, Michael Kampffmeyer (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 레이블링된 데이터의 부족 문제를 해결하기 위해 Few-Shot Learning(FSL)을 적용하는 것을 목표로 한다. 특히, 기존의 Prototypical Few-Shot Segmentation(FSS) 모델들이 가진 한계를 극복하고자 한다.

기존의 FSS 모델들은 대개 각 클래스(전경 및 배경)에 대한 Prototype(대표 벡터)을 생성하여 쿼리 이미지의 픽셀과 매칭하는 방식을 사용한다. 그러나 의료 영상에서 배경(Background) 클래스는 매우 크고 공간적으로 매우 이질적(Heterogeneous)이기 때문에, 소수의 서포트 슬라이스에서 추출한 몇 개의 Prototype만으로는 배경의 특성을 충분히 묘사하기 어렵다. 이는 결국 국부 정보의 손실로 이어져 분할 성능을 저하시키는 원인이 된다.

따라서 본 연구의 목표는 배경을 명시적으로 모델링하지 않고도 효과적으로 전경(Foreground)을 분할할 수 있는 새로운 Few-Shot 의료 영상 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 세 가지로 요약할 수 있다.

1. **Anomaly Detection-Inspired Approach**: 배경을 모델링하는 대신, 상대적으로 균질한 전경 클래스만을 단일 Prototype으로 모델링하고, 이와 다른 모든 영역을 이상치(Anomaly)로 간주하여 분할하는 방식을 제안하였다.
2. **3D Supervoxel-based Self-Supervision**: 의료 영상의 3차원 구조적 특성을 활용하기 위해, 기존의 2D Superpixel 기반 자기지도학습(Self-Supervision)을 3D Supervoxel로 확장하여 네트워크를 사전 학습시키는 방법론을 제시하였다.
3. **New Evaluation Protocol (EP2)**: 쿼리 이미지 내 대상 클래스의 위치 정보(Weak-label) 없이도 전체 볼륨을 분할할 수 있는 보다 실제적인 평가 프로토콜을 도입하여 제안 방법의 강건함을 입증하였다.

## 📎 Related Works

### Few-Shot Meta-learning 및 Semantic Segmentation

전통적인 Few-Shot Learning은 Meta-learning을 통해 새로운 태스크에 빠르게 적응하는 것을 목표로 한다. 특히 Metric-learning 기반의 Prototypical Network는 임베딩 공간에서 클래스별 Prototype을 생성하고 가장 가까운 Prototype을 찾는 방식을 취한다. 이를 분할 작업으로 확장한 PANet, PPNet 등이 있으며, 최근에는 의료 영상 분야에서도 이러한 접근 방식이 시도되었다.

### 기존 접근 방식의 한계

Ouyang et al. (2020) 등의 연구는 Superpixel 기반의 자기지도학습을 통해 레이블 없는 데이터로 FSS 모델을 학습시키는 방법을 제안하였다. 또한 ALPNet과 같이 국부적인 Prototype을 추가하여 배경의 이질성 문제를 해결하려 하였다. 하지만 저자들은 소수의 슬라이스만으로는 배경의 다양성을 모두 포착하는 것이 불가능하며, 이는 임시방편적인 해결책(ad-hoc solution)에 불과하다고 주장한다.

## 🛠️ Methodology

### 1. Anomaly Detection-Inspired FSS (ADNet)

제안된 ADNet은 배경 Prototype을 생성하지 않고 오직 전경 Prototype만을 사용한다.

**전경 Prototype 추출**
공유 인코더 $f_\theta$를 통해 서포트 이미지 $x$와 쿼리 이미지 $x^*$를 임베딩하여 각각 $F_s$와 $F_q$를 얻는다. 전경 클래스 $c$에 대해 Masked Average Pooling(MAP)을 적용하여 단일 전경 Prototype $p \in \mathbb{R}^d$를 계산한다.
$$p = \frac{\sum_{x,y} F_s(x,y) \odot y_{fg}(x,y)}{\sum_{x,y} y_{fg}(x,y)}$$
여기서 $\odot$은 Hadamard product이며, $y_{fg}$는 전경 마스크이다.

**Anomaly Score 및 분할**
쿼리 이미지의 각 픽셀 $F_q(x,y)$와 전경 Prototype $p$ 사이의 코사인 유사도를 기반으로 Anomaly Score $S(x,y)$를 계산한다.
$$S(x,y) = -\alpha \frac{F_q(x,y) \cdot p}{\|F_q(x,y)\| \|p\|}$$
유사도가 높을수록(전경일 확률이 높을수록) $S$ 값은 낮아지며, 유사도가 낮을수록(배경일 확률이 높을수록) $S$ 값은 높아진다. 최종 분할은 학습 가능한 임계값(Learned Threshold) $T$를 이용해 수행하며, 미분 가능하도록 shifted Sigmoid를 적용한다.
$$\hat{y}_{qfg}(x,y) = 1 - \sigma(S(x,y) - T)$$

**손실 함수 (Loss Function)**
전체 손실 함수 $\mathcal{L}$은 다음 세 가지의 합으로 구성된다.
$$\mathcal{L} = \mathcal{L}_S + \mathcal{L}_T + \mathcal{L}_{PAR}$$

- $\mathcal{L}_S$: 예측 마스크와 정답 마스크 간의 Binary Cross-Entropy 손실이다.
- $\mathcal{L}_T = T/\alpha$: 학습된 임계값 $T$를 최소화하여 전경의 임베딩이 더 콤팩트하게 형성되도록 유도한다.
- $\mathcal{L}_{PAR}$: 서포트와 쿼리의 역할을 바꾸어 다시 분할을 수행하는 Prototype Alignment 정규화 손실이다.

### 2. Supervoxel-Based Self-Supervision

레이블이 없는 데이터로 모델을 학습시키기 위해 3D Supervoxel을 활용한 자기지도학습 파이프라인을 제안한다.

1. **Supervoxel 생성**: 3D 영상 볼륨에서 유사한 특성을 가진 복셀들을 그룹화하여 Supervoxel을 오프라인으로 생성한다. 이때 $z$축의 해상도 차이를 고려하여 거리를 재가중치화한다.
2. **에피소드 구성**: 하나의 무작위 Supervoxel을 전경 클래스로 설정하여 3D pseudo-mask를 생성한다.
3. **샘플링**: 해당 Supervoxel이 포함된 볼륨에서 두 개의 2D 슬라이스를 무작위로 추출하여 각각 서포트 이미지와 쿼리 이미지로 사용한다.
4. **변형 적용**: 데이터 강건성을 위해 이미지에 무작위 변환을 적용하여 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MS-CMRSeg(심장 MRI), CHAOS(복부 장기 MRI)
- **지표**: Mean Dice Score
- **평가 프로토콜**:
  - **EP1**: 쿼리 이미지 내 전경 위치 정보(Weak-label)를 제공하는 기존 방식.
  - **EP2**: 서포트 슬라이스만으로 쿼리 볼륨 전체를 분할하는 제안 방식 (더 실제적인 시나리오).

### 주요 결과

- **EP1 결과**: 제안된 vSSL-ADNet은 기존 SOTA 모델인 pSSL-ALPNet과 유사한 성능을 보이면서도, 훨씬 적은 수의 Prototype을 사용한다.
- **EP2 결과**: 쿼리 이미지의 배경 영역이 훨씬 넓어지는 환경에서, 배경을 명시적으로 모델링하는 ALPNet 등의 성능은 급격히 하락한다. 반면, ADNet은 배경을 모델링하지 않으므로 매우 강건한 성능을 유지하며, 특히 복부 데이터셋에서 ALPNet 대비 20%p 이상의 높은 Dice Score 향상을 보였다.
- **3D 확장**: 3D ResNeXt-101을 백본으로 사용한 결과, 특히 복부 데이터셋의 왼쪽 신장(Left Kidney)과 비장(Spleen)처럼 경계가 모호한 클래스 간의 분리 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 논문의 핵심 통찰은 **"전경은 상대적으로 균질하지만, 배경은 매우 이질적이다"**라는 점이다. 배경을 억지로 모델링하려 하기보다 전경만을 정밀하게 모델링하고 나머지를 제외하는 Anomaly Detection 관점의 접근이 의료 영상의 특성에 더 적합함을 입증하였다. 또한, 2D Superpixel을 3D Supervoxel로 확장함으로써 볼륨 데이터의 공간적 연속성을 학습에 활용한 점이 유효하였다.

### 한계 및 비판적 해석

- **전경 균질성 가정**: 제안 방법은 전경이 균질하다는 가정에 의존한다. 만약 전경 클래스가 내부적으로 매우 다양한 특성을 가진 여러 영역으로 구성되어 있다면, 단일 Prototype만으로는 불충분할 수 있다.
- **Supervoxel의 노이즈**: Supervoxel 생성 시 경계가 모호한 장기(예: 왼쪽 신장과 비장)들이 하나의 Supervoxel로 묶일 경우, 네트워크가 두 클래스를 동일한 클러스터로 임베딩하게 되어 추론 시 혼동이 발생할 수 있다. 이는 Supervoxel pseudo-label의 노이즈 문제로, 향후 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

이 논문은 배경의 복잡성으로 인해 발생하는 Few-Shot 의료 영상 분할의 성능 저하 문제를 해결하기 위해, **배경 모델링을 완전히 제거하고 전경 Prototype만을 이용해 이상치를 탐지하는 방식의 ADNet**을 제안하였다. 더불어 **3D Supervoxel 기반의 자기지도학습**을 통해 레이블 없이도 강력한 특징 추출기를 학습시켰으며, 실제 환경과 유사한 평가 프로토콜(EP2)에서 기존 SOTA 모델들을 압도하는 성능과 강건함을 보였다. 이 연구는 향후 3D CNN을 이용한 직접적인 볼륨 분할 연구로 확장될 가능성이 높다.
