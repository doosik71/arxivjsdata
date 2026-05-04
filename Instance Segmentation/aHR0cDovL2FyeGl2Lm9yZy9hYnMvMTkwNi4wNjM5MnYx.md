# Instance Segmentation with Point Supervision

Issam H. Laradji, Negar Rostamzadeh, Pedro O. Pinheiro, David Vazquez, Mark Schmidt (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 인스턴스 분할(Instance Segmentation) 모델 학습에 필요한 막대한 양의 픽셀 단위 라벨링(Per-pixel labels) 비용을 줄이는 것이다. 인스턴스 분할은 각 객체의 픽셀을 분류함과 동시에 개별 객체 인스턴스를 구분해야 하는 작업으로, 자율 주행, 의료 영상 분석 등 다양한 분야에서 중요하게 사용된다.

하지만 PASCAL VOC나 CityScapes와 같은 데이터셋에서 픽셀 단위의 정교한 마스크를 생성하는 것은 인간 작업자에게 매우 많은 시간과 노력을 요구한다. 예를 들어, CityScapes의 경우 이미지 한 장당 최대 1.5시간의 라벨링 시간이 소요될 수 있다. 따라서 본 연구의 목표는 단지 객체당 하나의 점(Point-level annotation)만으로 구성된 약한 지도 학습(Weakly-supervised learning) 환경에서도 효과적인 인스턴스 분할 마스크를 생성하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 포인트 수준의 지도 학습만으로 인스턴스 분할을 수행하는 **WISE (Weakly-supervised Instance SEgmentation)** 네트워크를 제안한 것이다.

중심적인 설계 아이디어는 네트워크를 두 개의 브랜치, 즉 **Localization Network (L-Net)**와 **Embedding Network (E-Net)**로 나누어 구성하는 것이다. L-Net은 객체의 대략적인 위치를 찾고, E-Net은 동일한 객체에 속한 픽셀들이 임베딩 공간에서 서로 가깝게 위치하도록 학습한다. 또한, 정답 마스크가 없는 한계를 극복하기 위해 클래스 불가지론적(Class-agnostic) 객체 제안 방법(Object Proposal Method)인 SharpMask를 통해 생성된 **Pseudo-masks**를 학습에 활용하여 임베딩 공간을 최적화한다.

## 📎 Related Works

기존의 인스턴스 분할 방법론인 Mask R-CNN이나 MaskLab 등은 대부분 픽셀 단위의 밀집 라벨(Dense labels)에 의존하므로 라벨링 비용이 매우 높다는 한계가 있다. 임베딩 기반의 인스턴스 분할 방법들은 픽셀 간의 유사도를 측정해 그룹화하는 방식을 취하지만, 이 역시 정교한 라벨이 필요하다.

약한 지도 학습(Weakly supervised) 분야에서는 이미지 수준(Image-level)이나 바운딩 박스(Bounding box)를 이용한 연구들이 진행되었으나, 포인트 수준의 지도 학습을 인스턴스 분할에 적용한 사례는 드물다. 특히 포인트 지도 학습은 세만틱 분할(Semantic segmentation)에서는 효과적임이 증명되었으나, 인스턴스 분할에서는 각 인스턴스를 구분해야 한다는 추가적인 난관이 존재한다. 본 연구는 LCFCN의 위치 추정 능력과 메트릭 학습(Metric learning) 기반의 임베딩 기법을 결합하여 이 문제를 해결하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

WISE 네트워크는 ResNet-50을 공통 백본(Backbone)으로 공유하며, 두 개의 독립적인 경로인 L-Net과 E-Net으로 구성된다. 전체 파이프라인은 **위치 추정 $\rightarrow$ 임베딩 생성 $\rightarrow$ 픽셀 그룹화 $\rightarrow$ 마스크 정교화** 순으로 진행된다.

### 1. Localization Branch (L-Net)

L-Net의 목적은 이미지 내 각 객체의 위치와 카테고리를 식별하는 것이다. LC-FCN 구조를 기반으로 하며, 각 객체당 하나의 작은 블롭(Blob)을 생성하도록 학습된다. 사용되는 손실 함수 $\mathcal{L}_L$은 다음과 같다.

$$\mathcal{L}_L = \mathcal{L}_I + \mathcal{L}_P + \mathcal{L}_S + \mathcal{L}_F$$

여기서 $\mathcal{L}_I$는 이미지 수준 손실, $\mathcal{L}_P$는 포인트 수준 손실, $\mathcal{L}_S$는 분할(Split) 손실, $\mathcal{L}_F$는 거짓 양성(False positive) 손실이다. 이를 통해 네트워크가 각 인스턴스당 단 하나의 작은 블롭만 예측하도록 강제하며, 최종적으로는 해당 블롭 내에서 활성화 값이 가장 높은 픽셀을 객체의 대표 위치로 사용한다.

### 2. Embedding Branch (E-Net)

E-Net은 FCN8 구조를 기반으로 하며, 각 픽셀을 $d$차원의 임베딩 벡터로 매핑한다. 동일한 객체에 속한 픽셀들은 임베딩 공간에서 가깝게, 서로 다른 객체는 멀게 배치하는 것이 목표이다. 두 픽셀 $i, j$의 유사도 $S(i, j)$는 다음과 같은 squared exponential kernel 함수로 정의된다.

$$S(i,j) = \exp\left(-\frac{\|E_i - E_j\|^2}{2d}\right)$$

학습을 위해 정답 마스크 대신 SharpMask로 생성된 Pseudo-masks를 사용하여 픽셀 쌍(Pixel pairs) $P$를 구성하고, 아래의 손실 함수 $\mathcal{L}_E$를 최소화한다.

$$\mathcal{L}_E = -\sum_{(i,j) \in P} \left[ \mathbb{1}_{\{y_i=y_j\}} \log S(E_i, E_j) + \mathbb{1}_{\{y_i \neq y_j\}} \log(1 - S(E_i, E_j)) \right]$$

최종적인 전체 손실 함수 $\mathcal{L}_W$는 다음과 같이 가중합으로 정의된다.

$$\mathcal{L}_W = \lambda \cdot \mathcal{L}_L + (1 - \lambda) \cdot \mathcal{L}_E$$

### 3. 추론 및 예측 절차 (Test Time)

1. **위치 예측:** L-Net이 각 객체의 대표 픽셀 좌표를 출력한다.
2. **임베딩 생성:** E-Net이 이미지의 모든 픽셀에 대한 임베딩 벡터를 생성한다.
3. **유사도 계산:** L-Net이 찾은 대표 픽셀 및 배경 픽셀들과 다른 모든 픽셀 간의 유사도를 계산하여 가장 유사한 객체 그룹에 할당한다.
4. **마스크 정교화:** 생성된 초기 마스크를 SharpMask의 제안 영역들과 비교하여, Jaccard similarity가 가장 높은 Pseudo-mask로 교체함으로써 경계를 정교하게 다듬는다.

## 📊 Results

### 실험 설정

- **데이터셋:** PASCAL VOC 2012, COCO 2014, KITTI, CityScapes.
- **평가 지표:** Average Precision (AP) $\text{AP}_{25}, \text{AP}_{50}, \text{AP}_{75}$.
- **비교 대상:** Fully-supervised (Mask R-CNN), Weakly-supervised (PRM), 그리고 다양한 ablation baseline들.

### 주요 결과

1. **라벨링 예산 제한 실험 (Fixed Annotation Budget):**
   - 동일한 시간 예산(약 8.13시간) 내에서 이미지 수준 라벨, 포인트 수준 라벨, 픽셀 수준 라벨을 수집하여 비교했을 때, WISE(포인트 수준)가 다른 모든 약한 지도 학습 방법 및 소량의 픽셀 라벨을 사용한 Mask R-CNN보다 월등히 높은 성능을 보였다. 이는 포인트 라벨링이 비용 대비 효율이 가장 높은 방법임을 시사한다.

2. **정량적 성능:**
   - **PASCAL VOC:** $\text{AP}_{50}$ 기준으로 완전 지도 학습 방법보다는 낮지만, 이미지 수준 지도 학습 방법들(PRM 등)보다 경쟁력 있는 결과를 보여주었다.
   - **KITTI:** 자율 주행 벤치마크에서 완전 지도 학습 방법들과 상당히 근접한 성능($\text{MWCov}$ 74.2)을 달성하였다.
   - **CityScapes:** 완전 지도 학습과의 격차는 크지만, 바운딩 박스를 테스트 시점에 사용하는 방법들과 비교했을 때 E-Net 기반의 접근 방식이 경쟁력 있음을 확인하였다.

3. **Ablation Study:**
   - L-Net의 원본 블롭(Blob)만 사용하는 것보다 E-Net을 통한 임베딩 그룹화와 Pseudo-mask 정교화 과정을 거쳤을 때 성능이 비약적으로 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 논문은 인스턴스 분할 분야에서 매우 저렴한 포인트 수준의 지도 학습만으로도 실용적인 성능을 낼 수 있음을 보여주었다. 특히 고정된 라벨링 예산 하에서의 실험은 실제 데이터 구축 비용을 고려했을 때 매우 의미 있는 결과이다. L-Net과 E-Net의 상호 보완적인 구조(위치 추정 $\rightarrow$ 영역 확장)가 포인트 라벨의 정보 부족을 효과적으로 메웠다고 평가할 수 있다.

### 한계 및 가정

- **Pseudo-mask 의존성:** 모델이 SharpMask라는 사전 학습된 객체 제안 방법론에 크게 의존하고 있다. 만약 제안 방법론이 잘못된 영역을 추출한다면 E-Net의 학습 방향이 왜곡될 위험이 있다.
- **성능 격차:** 완전 지도 학습(Fully-supervised) 모델과의 성능 차이는 여전히 존재하며, 특히 복잡한 배경이나 객체가 밀집된 환경에서의 정밀도는 개선의 여지가 많다.
- **제안 방법론의 고정:** 사용된 Proposal method를 타겟 데이터셋에 맞게 파인튜닝하지 않고 그대로 사용했다는 점은 가능성을 보여주지만, 동시에 최적화되지 않은 도구를 사용했다는 한계도 된다.

## 📌 TL;DR

본 연구는 객체당 단 하나의 포인트 라벨만을 사용하여 인스턴스 분할을 수행하는 **WISE** 네트워크를 제안하였다. 객체 위치를 찾는 **L-Net**과 픽셀 간 유사도를 학습하는 **E-Net**을 결합하고, 사전 학습된 객체 제안(Object Proposal)의 Pseudo-mask를 통해 학습 효율을 높였다. 실험 결과, 제한된 라벨링 예산 내에서 가장 효율적인 성능을 보였으며, 포인트 수준 지도 학습 기반 인스턴스 분할의 강력한 베이스라인을 제시하였다. 향후 제안 방법론 없이(Proposal-free) 직접 마스크를 생성하는 방향으로의 연구 확장 가능성을 열어두었다.
