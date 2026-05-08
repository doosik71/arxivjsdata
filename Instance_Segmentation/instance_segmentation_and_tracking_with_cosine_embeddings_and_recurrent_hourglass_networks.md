# Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks

Christian Payer, Darko Stern, Thomas Neff, Horst Bischof, and Martin Urschler (2018)

## 🧩 Problem to Solve

본 논문은 동일한 클래스에 속하는 개별 객체들을 구분하여 분할하는 Instance Segmentation과, 이를 시간에 따라 추적하는 Instance Tracking 문제를 해결하고자 한다.

일반적인 Semantic Segmentation이 픽셀 단위로 클래스 라벨을 할당하는 것과 달리, Instance Segmentation은 같은 클래스 내에서도 서로 다른 객체에 고유한 ID를 부여해야 하므로 훨씬 복잡한 작업이다. 특히 세포 추적(Cell Tracking)과 같은 생의학 영상 분석에서는 수백 개의 작은 객체들이 동시에 존재하며, 시간이 흐름에 따라 이들의 위치가 변하거나 세포 분열(Mitosis)과 같은 이벤트가 발생한다. 기존의 방식들은 한 번에 하나의 객체만 분할하거나, 템포럴 정보(Temporal information)를 충분히 활용하지 못해 비디오 시퀀스에서 객체의 일관성을 유지하는 데 어려움이 있었다. 따라서 본 연구의 목표는 비디오의 시간적 정보를 통합하여 인스턴스를 정확하게 분할하고 추적할 수 있는 재귀적 딥러닝 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같은 세 가지 설계 아이디어로 요약된다.

첫째, **Recurrent Stacked Hourglass Network**를 제안하였다. 기존의 Stacked Hourglass 아키텍처 내부에 Convolutional Gated Recurrent Units (ConvGRU)를 통합함으로써, 비디오 프레임 간의 시간적 정보를 전파하고 기억하여 인스턴스 추적의 일관성을 높였다.

둘째, **Cosine Embedding Loss**라는 새로운 손실 함수를 도입하였다. 픽셀들을 고차원 임베딩 공간으로 매핑하고, 동일 인스턴스 내의 픽셀들은 유사한 방향을 가지게 하며, 서로 다른 인스턴스 간에는 서로 직교(Orthogonal)하도록 유도한다.

셋째, **사색 정리(Four Color Map Theorem)**의 직관을 활용하여 임베딩 학습을 최적화하였다. 모든 인스턴스가 서로 다른 임베딩을 가질 필요 없이, 인접한 인스턴스들만 서로 다른 임베딩을 가지면 된다는 가정을 통해 학습의 복잡도를 낮추고 최적화를 용이하게 하였다.

## 📎 Related Works

기존의 Instance Segmentation 방식은 주로 객체를 먼저 검출(Detection)한 후 개별적으로 마스크를 생성하는 Mask R-CNN과 같은 방식이나, 재귀적 네트워크를 이용해 이미 분할된 객체를 기억하는 방식이 사용되었다. 또한 최근에는 모든 픽셀에 대해 임베딩을 예측하고 이를 클러스터링하는 방식이 제안되었다.

그러나 기존의 임베딩 기반 방식들은 주로 Euclidean 공간에서의 거리를 최대화하는 손실 함수를 사용하였는데, 이는 재귀적 네트워크와 결합했을 때 임베딩 값의 범위가 제한되지 않아 학습이 불안정해지는 문제가 발생하였다. 또한, 비디오 데이터에서 시간적 맥락을 임베딩 학습에 직접적으로 통합하여 인스턴스 추적을 수행하려는 시도는 본 논문 이전에는 거의 제시되지 않았다.

## 🛠️ Methodology

### 1. Recurrent Stacked Hourglass Network

전체 시스템은 Stacked Hourglass Network를 기반으로 하며, 시간적 정보를 처리하기 위해 다음과 같은 구조를 가진다.

- **ConvGRU 통합**: Hourglass 네트워크의 수축 경로(Contracting path)와 확장 경로(Expanding path) 사이에 ConvGRU 레이어를 배치하여 이전 프레임의 정보를 다음 프레임으로 전달한다.
- **구조적 특징**: 3x3 필터와 64개의 출력 채널을 가진 단일 컨볼루션 레이어를 사용하며, 두 개의 Hourglass를 직렬로 쌓아(Stacked) 예측 정밀도를 높였다. 첫 번째 Hourglass의 출력을 입력 이미지와 결합(Concatenate)하여 두 번째 Hourglass의 입력으로 사용한다.

### 2. Cosine Embedding Loss

네트워크는 각 픽셀 $p$에 대해 $d$-차원의 임베딩 벡터 $e_p \in \mathbb{R}^d$를 예측한다. 본 논문은 벡터의 크기가 아닌 방향(Direction)에 집중하기 위해 Cosine Similarity를 사용한다.

두 벡터 $e_1, e_2$의 코사인 유사도는 다음과 같이 정의된다:
$$\cos(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|}$$

손실 함수 $L$은 다음과 같이 구성된다:
$$L = \frac{1}{|I|} \sum_{i \in I} \left( 1 - \frac{1}{|S^{(i)}|} \sum_{p \in S^{(i)}} \cos(\bar{e}^{(i)}, e_p) \right) + \left( \frac{1}{|N^{(i)}|} \sum_{p \in N^{(i)}} \cos(\bar{e}^{(i)}, e_p)^2 \right)$$

여기서 $\bar{e}^{(i)}$는 인스턴스 $i$에 속한 픽셀들의 평균 임베딩 벡터이다.

- **첫 번째 항**: 동일 인스턴스 $S^{(i)}$ 내의 픽셀들이 평균 벡터 $\bar{e}^{(i)}$와 같은 방향을 갖도록 하여 유사도를 1로 유도한다.
- **두 번째 항**: 인접한 다른 인스턴스들 $N^{(i)}$의 픽셀들이 평균 벡터 $\bar{e}^{(i)}$와 직교(Orthogonal)하도록 하여 유사도를 0으로 유도한다.

### 3. Clustering and Tracking

예측된 임베딩을 실제 인스턴스로 변환하기 위해 **HDBSCAN** 클러스터링 알고리즘을 사용한다.

- **데이터 포인트 구성**: 임베딩 값뿐만 아니라 이미지 좌표 $(x, y)$를 함께 입력으로 사용하여, 멀리 떨어져 있어 동일한 임베딩을 가진 서로 다른 객체들이 하나로 묶이는 것을 방지한다.
- **추적 절차**: 연속된 프레임 쌍에 대해 클러스터링을 수행하고, 프레임 간 객체 동일성은 IoU(Intersection over Union)가 가장 높은 객체끼리 매칭하여 추적한다. 세포 분열과 같은 이벤트의 경우, 이전 프레임에서 IoU가 가장 높았던 객체를 부모(Parent)로 지정한다.

## 📊 Results

### 1. Left Ventricle Segmentation (Heart MRI)

재귀적 구조의 효용성을 검증하기 위해 심장 MRI 영상의 좌심실 분할 실험을 수행하였다.

- **비교 대상**: ConvGRU를 일반 Convolution 레이어로 대체한 Non-recurrent 네트워크.
- **결과**: Recurrent 네트워크가 IoU 관점에서 더 높은 성능을 보였으며, 이는 시간적 정보의 통합이 분할 성능 향상에 기여함을 시사한다.

### 2. Leaf Instance Segmentation (Still Images)

Cosine Embedding Loss의 범용성을 확인하기 위해 정지 이미지의 잎 분할 실험을 진행하였다.

- **결과**: SBD(Symmetric Best Dice)와 $|DiC|$(개수 차이) 지표에서 기존의 최신 방법론들과 대등하거나 우수한 성능을 보였으며, 특히 구조가 훨씬 단순함에도 불구하고 높은 성능을 달성하였다.

### 3. Cell Instance Tracking (ISBI Challenge)

메인 실험으로 6개의 세포 추적 데이터셋을 사용하여 평가하였다.

- **결과**: 추적(TRA) 지표에서 6개 데이터셋 중 2개에서 1위, 2개에서 2위를 기록하며 State-of-the-art 성능을 보였다. 특히 세포가 밀집된 DIC-HeLa 데이터셋에서는 추적과 분할 모두에서 다른 모든 방법론을 압도하였다.
- **한계**: 매우 작은 세포들이 포함된 Fluo-HeLa, Fluo-SIM+ 데이터셋에서는 네트워크 입력 크기에 맞춘 다운샘플링 과정에서 세포 정보가 손실되어 성능이 낮게 나타났다.

## 🧠 Insights & Discussion

본 연구는 임베딩 기반의 인스턴스 분할에 시간적 정보를 결합하여 추적 문제를 해결하려 했다는 점에서 독창성이 있다.

특히 **Cosine Similarity**를 도입한 점이 결정적이었다. Euclidean 거리 기반의 손실 함수는 재귀적 네트워크에서 값이 발산하는 경향이 있으나, 코사인 유사도는 임베딩을 정규화된 방향성 데이터로 다루기 때문에 학습의 안정성을 크게 높였다. 또한, 사색 정리의 아이디어를 빌려 **인접 인스턴스 간의 차별성**만을 강조함으로써, 수많은 객체가 존재하는 상황에서도 최적화 효율을 높일 수 있었다.

다만, 입력 이미지의 해상도를 고정하여 다운샘플링하는 과정에서 작은 객체에 대한 분해능이 떨어진다는 점이 명확한 한계로 드러났다. 저자들은 이를 해결하기 위해 이미지를 작은 패치로 나누어 처리하는 슬라이딩 윈도우 방식의 도입을 향후 과제로 제시하고 있다.

## 📌 TL;DR

본 논문은 **ConvGRU가 통합된 Stacked Hourglass Network**와 **Cosine Embedding Loss**를 통해 비디오 내 객체의 인스턴스 분할 및 추적을 수행하는 프레임워크를 제안한다. 특히 인접한 객체끼리만 서로 다른 임베딩을 갖게 하는 효율적인 학습 전략과 코사인 유사도를 이용한 수치적 안정성을 통해, 세포 추적 챌린지에서 최상위권의 성능을 달성하였다. 이 연구는 특히 생의학 영상과 같이 다수의 작은 객체를 정밀하게 추적해야 하는 분야에 중요한 방법론적 기반을 제공한다.
