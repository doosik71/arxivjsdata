# A Spatial Guided Self-supervised Clustering Network for Medical Image Segmentation

Euijoon Ahn, Dagan Feng and Jinman Kim (발행 연도 미명시)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 감독 학습(Supervised Learning) 기반 딥러닝 모델이 겪고 있는 데이터 라벨링 의존성 문제를 해결하고자 한다. 의료 영상의 특성상 전문가에 의한 정교한 어노테이션(Annotation) 작업은 막대한 비용과 시간이 소요되며, 전문가 간의 주관적 차이(Inter- and intra-observer variability)로 인해 일관된 고품질의 라벨 데이터를 대량으로 확보하는 것이 매우 어렵다.

따라서 본 연구의 목표는 대량의 라벨링된 데이터 없이도 단일 이미지로부터 픽셀의 특성 표현(Feature representation)과 클러스터 할당(Clustering assignment)을 엔드투엔드(End-to-end) 방식으로 반복 학습하여 정확한 분할 결과를 도출하는 '공간 안내 자기지도 클러스터링 네트워크(Spatial Guided Self-supervised Clustering Network, SGSCN)'를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단순한 픽셀 값의 유사성을 넘어, 이미지 내의 **공간적 관계(Spatial relationships)**를 학습 과정에 직접적으로 통합하는 것이다. 이를 위해 다음과 같은 설계 아이디어를 도입하였다.

1.  **반복적 자기지도 학습**: 정답 라벨 대신 클러스터링 할당 결과를 대리 라벨(Surrogate labels)로 사용하여 CNN의 파라미터를 업데이트하고, 이를 통해 특성 표현과 클러스터링 성능을 동시에 향상시킨다.
2.  **Sparse Spatial Loss 도입**: $L_1$-norm 기반의 정규화를 통해 공간적으로 연결된 픽셀(예: 에지 부분)들을 효과적으로 그룹화하여 경계선 분할 성능을 높인다.
3.  **Context-based Consistency Loss 도입**: 특정 클러스터에 속한 픽셀들이 해당 클러스터의 중심점(Cluster centre)과 공간적으로 가깝게 위치하도록 강제함으로써, 노이즈를 억제하고 영역의 형태를 명확히 한다.

## 📎 Related Works

기존의 의료 영상 분석에서는 도메인 적응(Domain Adaptation)이나 오토인코더(Auto-encoder)를 이용한 비지도 특성 학습(Unsupervised feature learning) 등이 시도되었다. 최근에는 데이터 스스로 감독 신호를 생성하는 자기지도 학습(Self-supervised learning)이 주목받고 있으며, 특히 이미지 클러스터링을 이용해 가짜 라벨을 생성하는 방식이 제안되었다.

대표적으로 DeepCluster는 특성 표현과 클러스터 할당을 반복적으로 학습하며, IIC(Invariant Information Clustering)는 공간적으로 변환된 이미지 패치 간의 상호 정보량(Mutual information)을 최대화하여 클러스터링을 수행한다. 하지만 이러한 기존 방식들은 다음과 같은 한계가 있다.
- 클러스터의 크기($k$값)를 수동으로 설정해야 한다.
- 경계가 모호하거나(Fuzzy boundaries), 복잡한 형태, 아티팩트 및 노이즈가 존재하는 의료 영상의 특성을 충분히 반영하지 못해 분할 성능이 저하될 수 있다.

## 🛠️ Methodology

### 전체 시스템 구조
SGSCN은 입력 이미지 $x_i$를 합성곱 분할 네트워크 $F(\theta)$에 통과시켜 분할 맵 $O_i$를 생성한다. 이후 $O_i$를 평균 0, 분산 1로 정규화하여 $\tilde{O}_i$를 얻으며, 각 픽셀에서 가장 응답 값이 큰 채널을 선택($\text{argmax}$)하여 최종 클러스터 라벨 $C_i$를 결정한다. 이 과정은 특성 학습과 클러스터링이 안정화될 때까지 반복된다.

### 손실 함수 및 학습 목표
네트워크는 다음 세 가지 손실 함수의 합을 최소화하는 방향으로 학습된다.

**1. Cross-entropy Loss ($\mathcal{L}_{ce}$)**
자기지도 클러스터링을 위해 정규화된 분할 맵 $\tilde{O}_i$와 예측된 클러스터 라벨 $C_i$ 사이의 교차 엔트로피를 계산한다.
$$\mathcal{L}_{ce}(\tilde{O}_i, C_i) = \sum_{k} \sum_{p} -\delta(p - C_i) \ln \tilde{O}_{i, p}$$
여기서 $p$는 클러스터 인덱스이며, $\delta(\cdot)$는 지시 함수(Indicator function)이다.

**2. Sparse Spatial Loss ($\mathcal{L}_{ss}$)**
공간적 연결성을 강화하기 위해 $\tilde{O}_i$의 수평 및 수직 차이에 $L_1$-norm을 적용한다. 이는 미세한 에지나 연결된 영역에 대해 페널티를 적게 주어 공간적 관계를 더 잘 파악하게 한다.
$$\mathcal{L}_{ss}(\tilde{O}_i) = \sum_{k} \sum_{l} \left( |\tilde{O}_{i, k-1, l} - \tilde{O}_{i, k, l}| + |\tilde{O}_{i, k, l-1} - \tilde{O}_{i, k, l}| \right)$$

**3. Context-based Consistency Loss ($\mathcal{L}_{cc}$)**
각 클러스터의 중심점 $\mu_{i, a}$를 공간 확률 분포 함수로 계산하고, 해당 클러스터에 속한 픽셀들이 중심점에서 멀리 떨어지지 않도록 제한한다.
먼저 클러스터 중심 $\mu_{i, a}$는 다음과 같이 정의된다.
$$\mu_{i, a} = \frac{\sum_{k, l} (k, l) \cdot \tilde{O}_{i, a}(k, l)}{\sum \tilde{O}_{i, a}(k, l)}$$
이를 이용하여 중심점과 픽셀 간의 유클리드 거리를 기반으로 손실 함수를 계산한다.
$$\mathcal{L}_{cc}(\tilde{O}_i) = \sum_{k, l} \frac{\| (k, l) - (\mu_{i, x} - \mu_{i, y}) \|^2 \cdot \tilde{O}_{i, a}(k, l)}{\sum \tilde{O}_{i, a}(k, l)}$$

### 학습 절차 및 아키텍처
- **아키텍처**: 3개의 합성곱 층(Convolutional layers)으로 구성된 얕은 CNN을 사용하며, 각 층은 ReLU 활성화 함수와 Batch Normalization을 포함한다. 커널 크기는 $3 \times 3$, 스트라이드는 1, 패딩은 1이며, 필터 수는 최대 클러스터 수인 100으로 설정하였다.
- **절차**: 최대 클러스터 수 $k$를 100으로 설정하고 학습을 시작하면, 유사한 픽셀들이 동일한 클러스터로 묶이면서 실제 유효한 클러스터의 수는 점차 줄어든다.

## 📊 Results

### 실험 설정
- **데이터셋**: 피부 병변 분할을 위한 PH2 데이터셋(200장)과 간 종양 분할을 위한 SYSU-US 데이터셋(100장)을 사용하였다.
- **비교 대상**: 전통적인 $k$-means 클러스터링, 최신 자기지도 학습 방법인 DeepCluster 및 IIC와 비교하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC), Hammoude distance (HM), XOR를 사용하였다. (DSC는 높을수록, HM과 XOR는 낮을수록 우수함)

### 정량적 결과
- **피부 병변 분할 (PH2)**: SGSCN은 DSC 83.4%, XOR 28.2%를 기록하여 DeepCluster(DSC 79.6%)와 IIC(DSC 81.2%)보다 우수한 성능을 보였다.
- **간 종양 분할 (SYSU-US)**: 초음파 영상의 노이즈로 인해 전반적인 성능은 낮아졌으나, SGSCN이 DSC 63.2%, HM 46.2%, XOR 52.3%로 가장 높은 정확도를 기록하였다.

### 정성적 분석
실험 결과, Sparse Spatial Loss는 공간적으로 연결된 영역의 분할을 돕고, Context-based Consistency Loss는 클러스터 중심에 집중하게 함으로써 모호한 경계와 노이즈가 많은 영역에서의 분할 성능을 유의미하게 향상시켰음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 공간적 제약 조건을 손실 함수에 통합함으로써, 라벨 없는 의료 영상에서도 고성능의 분할이 가능하다는 것을 입증하였다. 특히 DeepCluster와 같은 기존 방식이 공간적 관계를 고려하지 않아 경계가 모호한 영역에서 취약했던 점을 성공적으로 개선하였다.

**강점 및 한계**:
- **강점**: 수동적인 $k$값 설정 없이도 반복 학습을 통해 최적의 특성 표현을 찾아내며, 공간 가이드 손실을 통해 의료 영상 특유의 노이즈와 모호한 경계 문제를 완화하였다.
- **한계**: 병변이나 종양의 크기가 매우 작거나, 시각적으로 구분이 뚜렷하지 않고 경계가 불완전한 경우에는 여전히 분할 성능이 떨어진다는 한계가 있다.
- **향후 방향**: 저자들은 Affine 변환이나 Thin Plate Spline Grid와 같은 기하학적 제약 조건을 추가함으로써 특성 표현을 더욱 개선할 수 있을 것으로 전망하고 있다.

## 📌 TL;DR

이 논문은 라벨 데이터가 부족한 의료 영상 분할 문제를 해결하기 위해, 공간적 연결성과 클러스터 중심성을 강제하는 두 가지 특수 손실 함수($\mathcal{L}_{ss}, \mathcal{L}_{cc}$)를 도입한 **SGSCN**을 제안한다. 이 네트워크는 단일 이미지에서 특성 학습과 클러스터링을 반복적으로 수행하며, 실험 결과 기존의 비지도/자기지도 학습 방법(k-means, DeepCluster, IIC)보다 피부 병변 및 간 종양 분할에서 더 높은 정확도를 달성하였다. 이는 향후 대규모 라벨링 없이도 다양한 의료 영상 분석에 적용될 가능성이 높음을 시사한다.