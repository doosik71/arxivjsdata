# Unsupervised Segmentation of 3D Medical Images Based on Clustering and Deep Representation Learning

Takayasu Moriya et al. (2018)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상의 비지도 분할(Unsupervised Segmentation) 문제를 해결하고자 한다. 최근 Convolutional Neural Networks (CNNs)를 이용한 영상 분할 기술은 비약적인 발전을 이루었으나, 대부분의 방법론이 대량의 수동 어노테이션(manual annotation) 데이터를 필요로 하는 지도 학습(supervised learning)에 의존하고 있다. 의료 영상의 특성상 전문가에 의한 정밀한 어노테이션을 확보하는 것은 매우 어렵고 비용이 많이 들기 때문에, 증가하는 의료 영상 데이터 양에 대응하기 위해서는 지도 학습의 한계를 극복할 수 있는 비지도 학습 기반의 분할 방법이 필수적이다.

본 연구의 구체적인 목표는 마이크로 컴퓨터 단층촬영(micro-CT)으로 촬영된 폐암 조직 영상에서 침습성 암(invasive carcinoma), 비침습성 암(noninvasive carcinoma), 그리고 정상 조직(normal tissue)의 세 가지 병리 영역을 자동으로 분할하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비지도 딥 표현 학습(Unsupervised Deep Representation Learning)과 클러스터링을 결합하여, 별도의 정답 라벨 없이도 영상의 특징을 학습하고 영역을 분할하는 통합 프레임워크를 제안하는 것이다.

가장 중심적인 기여는 Joint Unsupervised Learning (JULE) 프레임워크를 3D 의료 영상 분할에 최초로 도입했다는 점이다. JULE는 CNN이 생성한 표현(representation)을 클러스터링하고, 다시 그 클러스터 라벨을 지도 신호(supervisory signal)로 사용하여 CNN의 파라미터를 업데이트하는 순환적 구조를 가진다. 이를 통해 데이터로부터 직접 변별력 있는 특징을 학습하고, 최종적으로 $k$-means 클러스터링을 통해 정밀한 분할 결과를 얻는 파이프라인을 구축하였다.

## 📎 Related Works

기존의 3D 의료 영상 비지도 분할 연구들은 주로 클러스터링 기반의 방법론을 사용해 왔다. 그러나 대부분의 기존 접근 방식은 수동으로 설계된 특징(hand-crafted features)을 추출한 뒤 전통적인 클러스터링 알고리즘을 적용하는 수준에 머물러 있었다. 이러한 방식은 데이터의 복잡한 고차원적 특징을 충분히 포착하지 못한다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 딥러닝 기반의 표현 학습을 도입하여 데이터로부터 최적의 특징을 자동으로 학습하며, 특히 JULE를 활용함으로써 데이터의 변동성(영상 타입, 크기, 샘플 수 등)에 강건하게 대응하고 다양한 클러스터링 알고리즘에 적합한 표현을 학습할 수 있다는 차별점을 가진다.

## 🛠️ Methodology

제안된 방법론은 크게 두 단계의 페이즈로 구성된다.

### 1. Deep Representation Learning (Phase 1)

첫 번째 단계에서는 대상 영상에서 무작위로 추출한 3D 패치들을 이용하여 JULE를 통해 깊은 특징 표현을 학습한다. JULE의 핵심은 의미 있는 클러스터 라벨이 표현 학습의 지도 신호가 되고, 다시 변별력 있는 표현이 더 정확한 클러스터링을 가능하게 한다는 상호 보완적 관계에 있다.

JULE의 목적 함수는 다음과 같이 정의된다:
$$\hat{y}, \hat{\theta} = \arg \min_{y, \theta} L(y, \theta | I)$$
여기서 $I$는 레이블이 없는 이미지 패치 집합, $y$는 클러스터 라벨, $\theta$는 CNN의 파라미터, 그리고 $L$은 손실 함수를 의미한다.

학습 과정은 다음과 같은 순환 구조를 가진다:

- **Forward Pass**: 현재 CNN 파라미터 $\theta_t$를 통해 추출된 표현 $X_t$에 대해 Agglomerative Clustering을 수행하여 새로운 라벨 $y_t$를 할당한다.
- **Backward Pass**: 할당된 $y_t$를 정답 라벨로 간주하여 3D CNN을 학습시키고 파라미터를 $\theta_{t+1}$로 업데이트한다.

본 논문은 이를 3D 의료 영상에 적용하기 위해 CNN 전체 아키텍처에 3D Convolution을 적용하였으며, 최종 클러스터 라벨을 사용한 마지막 Backward pass를 추가하여 CNN이 가장 정밀한 표현을 학습할 수 있도록 확장하였다.

### 2. Segmentation (Phase 2)

학습된 CNN을 이용하여 대상 영상 전체를 분할한다.

1. 대상 영상에서 일정 간격(stride $s$)으로 $w \times w \times w$ 크기의 패치를 모두 추출한다.
2. 학습된 CNN을 통해 각 패치를 160차원의 특징 표현으로 변환한다.
3. 변환된 표현들에 대해 $k$-means 클러스터링을 적용하여 $K=3$개의 클러스터로 나눈다.
4. 각 패치의 중심을 기준으로 $s \times s \times s$ 크기의 서브패치 영역에 해당 클러스터 라벨을 투영하여 최종 3D 분할 영상을 생성한다.

### CNN 아키텍처 및 세부 설정

- **구조**: 3개의 Convolutional layers $\rightarrow$ 1개의 Max pooling layer $\rightarrow$ 2개의 Fully-connected layers 순으로 구성된다.
- **세부 사양**:
  - 모든 Convolutional kernel 크기는 $5 \times 5 \times 5$, stride는 1이며, 각 층마다 50개의 커널을 사용한다.
  - Max pooling kernel은 $2 \times 2 \times 2$, stride는 2이다.
  - FC layer의 뉴런 수는 각각 1350개와 160개이며, 마지막 FC layer 이후에는 L2-normalization을 적용한다.
  - 활성화 함수로는 ReLU를 사용하며, 각 Convolutional layer 출력에 Batch Normalization을 적용한다.
  - 입력 패치 크기는 $27 \times 27 \times 27$ voxels이다.

## 📊 Results

### 실험 설정

- **데이터셋**: micro-CT로 촬영된 3개의 폐암 조직 영상(lung-A, lung-B, lung-C)을 사용하였다.
- **평가 지표**: 분할 정확도를 측정하기 위해 Normalized Mutual Information (NMI)을 사용하였으며, 수동으로 어노테이션된 7개의 슬라이스를 기준으로 평가하였다.
- **비교 대상**: 전통적인 $k$-means 분할 방식과 Multithreshold Otsu 방법론을 기준선(baseline)으로 설정하였다.

### 정량적 결과

실험 결과, JULE 기반의 분할 방법이 기존의 비지도 학습 방법들보다 높은 NMI 값을 기록하며 우수한 성능을 보였다. 비록 절대적인 NMI 수치가 매우 높지는 않았으나, 모든 데이터셋에서 기존 방법론을 상회하는 결과를 나타냈다.

### 정성적 결과

정성적 평가 결과, 제안 방법은 정상 조직 영역과 암 영역(침습성 및 비침습성 암)을 효과적으로 구분해 내는 것으로 확인되었다. 특히 정상 조직은 저강도(low intensity) 영역으로, 암 영역은 고강도(high intensity) 영역으로 잘 분리되었다.

## 🧠 Insights & Discussion

본 연구의 결과는 JULE가 영상의 강도(intensity) 및 강도의 변화량(variation)을 반영하는 특징을 효과적으로 학습할 수 있음을 시사한다.

- **강도 기반 분리**: 일반적으로 정상 조직은 낮은 강도를, 암 조직은 높은 강도를 가지므로, CNN이 이를 구분하는 특징을 학습하였다.
- **변화량 기반 분리**: 침습성 암과 정상 조직은 강도의 변화가 적은 반면, 비침습성 암은 강도의 변화가 큰 특성이 있다. JULE 기반 분할이 lung-A와 lung-B에서 침습성 암과 비침습성 암을 구분해 낸 것은 CNN이 이러한 강도의 변동성 특징까지 포착했음을 의미한다.

다만, NMI 수치가 아주 높지 않은 점은 비지도 학습의 본질적인 한계일 수 있으며, 향후 더 많은 데이터나 정교한 클러스터링 기법과의 결합이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 레이블이 없는 3D 의료 영상(micro-CT 폐암 조직)을 분할하기 위해, CNN 표현 학습과 클러스터링을 교차적으로 수행하는 JULE 프레임워크를 적용한 비지도 분할 방법을 제안하였다. 실험 결과, 제안 방법은 전통적인 $k$-means나 Otsu 방법보다 우수한 분할 성능을 보였으며, 특히 조직 간의 강도 차이와 변동성을 효과적으로 학습하여 병리 영역을 구분할 수 있음을 입증하였다. 이 연구는 수동 어노테이션 비용이 높은 의료 영상 분야에서 딥러닝 기반 비지도 학습의 실용적 가능성을 제시하였다는 점에서 의의가 있다.
