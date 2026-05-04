# Self-Prediction for Joint Instance and Semantic Segmentation of Point Clouds

Jinxian Liu, Minghui Yu, Bingbing Ni, and Ye Chen (2020)

## 🧩 Problem to Solve

본 논문은 3D point cloud의 **인스턴스 분할(Instance Segmentation)**과 **시맨틱 분할(Semantic Segmentation)**을 동시에 수행하는 효율적인 학습 체계를 제안한다.

기존의 3D 분할 연구들은 주로 새로운 컨볼루션 연산자(convolutional operators)를 설계하여 포인트 간의 기하학적/시맨틱 관계를 캡처하는 데 집중해 왔다. 그러나 이러한 방식은 포인트 간의 관계 탐색에 대한 명확한 제약이나 가이드라인이 부족하여, 네트워크가 가진 잠재력을 완전히 활용하지 못한다는 한계가 있다. 

또한, 인스턴스 분할과 시맨틱 분할은 서로 밀접하게 연관되어 있음에도 불구하고, 많은 기존 연구들이 이 두 작업을 독립적으로 처리하거나 시맨틱 분할의 후처리 단계로 인스턴스 분할을 다루는 직렬 방식(serial fashion)을 채택한다. 이러한 구조는 시맨틱 분할의 성능이 인스턴스 분할의 결과에 직접적인 영향을 미치게 하여 최적의 해를 찾는 데 방해가 된다. 따라서 본 논문의 목표는 포인트 간 관계 탐색을 강제하는 새로운 학습 체계인 **Self-Prediction**을 도입하고, 이를 통해 인스턴스와 시맨틱 분할이 서로 보완하며 성능을 높이는 통합 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Self-Prediction**이라는 보조 학습 과제를 통해 백본 네트워크가 더 변별력 있는 특징(discriminative features)을 학습하도록 강제하는 것이다.

구체적으로, 하나의 포인트 클라우드 샘플을 두 개의 부분 집합으로 나누고, 한쪽 집합의 라벨을 이용해 다른 쪽 집합의 라벨을 예측하게 만든다. 이 과정에서 포인트 간의 유사도를 기반으로 한 **Label Propagation Algorithm(라벨 전파 알고리즘)**을 사용한다. 만약 네트워크가 포인트 간의 관계와 기하학적 구조를 충분히 학습했다면, 일부 포인트의 라벨만으로도 나머지 포인트의 라벨을 정확하게 예측할 수 있을 것이라는 직관에 기반한다. 

또한, 인스턴스 특징과 시맨틱 특징을 결합하여 통합된 임베딩을 생성하고, 이를 통해 두 작업을 동시에 Self-Prediction 과제에 활용함으로써 인스턴스 및 시맨틱 분할 성능을 상호 강화하는 통합 프레임워크를 제안한다.

## 📎 Related Works

### 3D Point Cloud Segmentation
- **인스턴스 분할**: SGPN, ASIS, JSIS3D 등이 제안되었다. 특히 ASIS는 시맨틱 정보를 활용해 인스턴스 임베딩을 학습하여 두 작업을 연관시키려 했으나, 추론 시 추가적인 연산 부담이 발생한다. 3D-BoNet과 같은 제안 기반(proposal-based) 방법들은 효율적이지만 입력 데이터의 다양성에 대한 적응력이 떨어진다.
- **시맨틱 분할**: PointNet, PointNet++, DGCNN 등이 포인트 클라우드의 불규칙한 구조를 처리하기 위해 제안되었다. DGCNN의 EdgeConv와 같이 포인트 간의 관계를 명시적으로 모델링하는 방식들이 성능 향상을 이끌었으나, 여전히 관계 탐색을 가이드할 수 있는 제약 조건의 부재가 한계로 지적된다.

### Label Propagation Algorithms
라벨 전파 알고리즘은 소수의 라벨링된 데이터로부터 라벨이 없는 데이터로 정보를 확산시키는 비지도 학습 기반의 기법이다. 본 논문은 이 알고리즘을 단순히 라벨을 전파하는 용도가 아니라, 네트워크가 포인트 간의 관계를 더 잘 학습하도록 만드는 **지도 학습의 제약 조건(supervised constraint)**으로 재정의하여 사용한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
제안된 프레임워크는 하나의 **Backbone Network**와 세 개의 헤드(**Instance-head, Semantic-head, Self-Prediction head**)로 구성된다.
- **Backbone**: PointNet, PointNet++ 등 기존의 다양한 아키텍처를 사용할 수 있으며, 입력 포인트 클라우드로부터 특징 행렬 $F \in \mathbb{R}^{N \times H}$를 추출한다.
- **Instance-head**: $F$를 입력받아 인스턴스 임베딩 $F_{ins} \in \mathbb{R}^{N \times H_{ins}}$를 생성한다.
- **Semantic-head**: $F$를 입력받아 시맨틱 임베딩 $F_{sem} \in \mathbb{R}^{N \times H_{sem}}$을 생성하고 클래스를 분류한다.
- **Self-Prediction head**: $F_{ins}$와 $F_{sem}$을 결합하여 공동 임베딩 $F_{joint}$를 만들고, 이를 이용해 Self-Prediction 과제를 수행한다.

### 2. Self-Prediction 메커니즘
Self-Prediction은 포인트 간의 관계를 탐색하도록 강제하는 보조 작업이다. 과정은 다음과 같다.

**1) 그래프 구축**
백본 네트워크를 통해 추출된 특징 $\phi(x_i)$를 사용하여 모든 포인트 간의 완전 그래프(complete graph) $W$를 구축한다. 가우시안 유사도 함수를 사용하여 가중치를 정의한다.
$$W_{ij} = \exp\left(-\frac{d(\phi(x_i), \phi(x_j))^2}{2\sigma^2}\right)$$
여기서 $d$는 유클리드 거리이며, $\sigma$는 스케일 파라미터(실험에서는 1로 설정)이다. 이후 라플라시안 행렬(Laplacian matrix) $L$을 계산하여 그래프를 정규화한다.
$$L = D^{-1/2}WD^{-1/2}$$
($D$는 $W$의 행 합을 대각 성분으로 하는 대각 행렬이다.)

**2) 라벨 전파 (Label Propagation)**
포인트 클라우드를 두 그룹 $X^S$와 $X^U$로 동일하게 나눈다. $X^S$의 라벨을 알고 있을 때 $X^U$의 라벨을 예측하는 방식과 그 반대 방향의 예측을 동시에 수행하는 양방향 전파를 실시한다. 전파 과정의 반복식은 다음과 같다.
$$S^{(t+1)} = \alpha LS^{(t)} + (1-\alpha)S^0$$
여기서 $\alpha$는 전파 비율을 조절하는 파라미터이다. 실제 구현에서는 수렴할 때까지 반복하는 대신, 다음과 같은 닫힌 형태(closed form)의 수식을 사용하여 직접 결과를 계산한다.
$$S^* = (I - \alpha L)^{-1} S^0$$
최종적으로 예측된 결과 $Y^*$를 실제 정답(Ground Truth) 라벨 $Y$와 비교하여 학습시킨다.

### 3. 통합 학습 프레임워크 및 손실 함수
본 모델은 인스턴스와 시맨틱 정보를 결합하여 공동의 라벨 행렬 $Y_{joint}$를 구성하고 Self-Prediction을 수행한다.

**1) 인스턴스 손실 ($L_{ins}$)**
인스턴스 임베딩의 응집도와 분별력을 높이기 위해 Variance loss, Distance loss, Regularization loss의 합으로 정의한다.
$$L_{ins} = L_{var} + L_{dist} + 0.001 \cdot L_{reg}$$

**2) 시맨틱 손실 ($L_{sem}$)**
전형적인 Cross-Entropy 손실 함수를 사용하여 각 포인트의 클래스를 예측한다.
$$L_{sem} = -\frac{1}{N} \sum_{i=1}^{N} [Y^{sem}]_i \log p_i$$

**3) Self-Prediction 손실 ($L_{sp}$)**
Self-Prediction 헤드를 통해 예측된 인스턴스 및 시맨틱 라벨과 정답 간의 Cross-Entropy 손실을 계산한다.
$$L_{sp} = -\frac{1}{N} \sum_{i=1}^{N} ([Y^{ins}]_i^* \log q_i + [Y^{sem}]_i^* \log r_i)$$

**4) 최종 목적 함수**
세 가지 손실을 가중 합산하여 전체 네트워크를 동시에 최적화한다.
$$L = L_{ins} + L_{sem} + \beta L_{sp}$$
여기서 $\beta$는 Self-Prediction 손실의 기여도를 조절하는 파라미터(실험에서는 0.8로 설정)이다.

## 📊 Results

### 실험 설정
- **데이터셋**: S3DIS (실내 장면), ShapeNet (물체 파트)
- **평가 지표**: 
    - 시맨틱 분할: mIoU, mAcc, oAcc
    - 인스턴스 분할: mPrec, mRec, mCov, mWCov (IoU 임계값 0.5)
- **백본 네트워크**: PointNet, PointNet++, DGCNN

### 주요 결과
- **S3DIS 데이터셋**:
    - PointNet++를 백본으로 사용했을 때, 인스턴스 분할에서 SOTA(State-of-the-art) 성능을 달성하였다. (6-fold CV 기준 mPrec 67.5%, mRec 54.6%)
    - 시맨틱 분할에서도 기존 SOTA 모델들과 경쟁 가능한 수준의 성능을 보였다.
    - baseline(Self-Prediction 제외) 대비 성능 향상이 뚜렷하며, 특히 인스턴스 분할의 mPrec와 mRec가 크게 상승하였다.
- **ShapeNet 데이터셋**:
    - DGCNN 백본을 적용했을 때 pIoU 86.2%, mpIoU 83.1%를 기록하며 baseline 및 ASIS보다 우수한 성능을 보였다.

### 분석 결과
- **Self-Prediction의 효과**: `Ins-SP`(인스턴스만 적용)나 `Sem-SP`(시맨틱만 적용)보다 `InsSem-SP`(두 작업 통합 적용)가 가장 높은 성능을 기록하여, 두 작업이 서로를 강화한다는 점을 입증하였다.
- **추론 효율성**: Self-Prediction 헤드는 학습 단계에서만 사용되며, 추론 시에는 제거되므로 추가적인 연산 비용이나 파라미터 증가가 전혀 없다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문의 가장 큰 강점은 네트워크 아키텍처 자체를 변경하지 않고도 **학습 전략(learning scheme)**만으로 성능을 비약적으로 향상시켰다는 점이다. 특히 Self-Prediction이라는 보조 과제가 백본 네트워크가 포인트 간의 고차원적 관계를 더 잘 학습하도록 가이드하는 강력한 정규화(regularization) 도구로 작용함을 보여주었다. 또한, 특정 백본에 종속되지 않고 PointNet, PointNet++, DGCNN 등 다양한 구조에 범용적으로 적용 가능하다는 점이 매우 효율적이다.

### 한계 및 논의사항
- **계산 복잡도**: 학습 단계에서 $N \times N$ 크기의 완전 그래프를 구축하고 $(I - \alpha L)^{-1}$ 행렬 연산을 수행하므로, 포인트 수가 매우 많은 경우 메모리 및 연산 비용이 급격히 증가할 가능성이 있다. 논문에서는 이를 해결하기 위해 포인트 클라우드를 여러 그룹으로 나누어 처리하는 방식을 제안했으나, 대규모 씬에 대한 확장성 검증이 더 필요해 보인다.
- **하이퍼파라미터**: $\alpha, \beta, \sigma$ 등의 파라미터에 대해 어느 정도 강건함(robustness)을 보였으나, 데이터셋의 특성에 따라 최적값이 달라질 수 있다는 점이 남아있다.

## 📌 TL;DR

본 논문은 3D 포인트 클라우드의 인스턴스 및 시맨틱 분할 성능을 높이기 위해, 일부 포인트의 라벨로 나머지 포인트의 라벨을 예측하는 **Self-Prediction** 학습 체계를 제안한다. 이 방식은 라벨 전파 알고리즘을 통해 백본 네트워크가 포인트 간의 관계적 특징을 더 잘 추출하도록 강제하며, 추론 시에는 추가 연산 없이 성능 향상만 누릴 수 있다는 장점이 있다. S3DIS와 ShapeNet 데이터셋에서 SOTA급 성능을 입증하였으며, 향후 다양한 포인트 클라우드 분석 작업의 학습 가이드라인으로 활용될 가능성이 높다.