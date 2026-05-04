# Scale Disparity of Instances in Interactive Point Cloud Segmentation

Chenrui Han, Xuan Yu, Yuxuan Xie, Yili Liu, Sitong Mao, Shunbo Zhou, Rong Xiong, Yue Wang (2024)

## 🧩 Problem to Solve

본 논문은 대화형 포인트 클라우드 세그멘테이션(Interactive Point Cloud Segmentation)에서 발생하는 **인스턴스 간의 규모 차이(Scale Disparity)** 문제를 해결하고자 한다. 

일반적인 인스턴스 세그멘테이션에서는 주로 형태가 분명한 'Thing'(사물) 카테고리에 집중하지만, 실제 사용자가 대화형 인터페이스를 통해 세그멘테이션을 수행할 때는 'Stuff'(배경/환경, 예: 도로, 벽) 카테고리를 하나의 인스턴스로 분리해내길 원하는 경우가 많다. 이 경우, 컵과 같이 매우 작은 객체부터 건물이나 도로와 같이 매우 큰 객체까지 그 크기의 범위가 극단적으로 넓어진다. 

기존의 CNN 기반 방식은 수용 영역(Receptive Field)의 한계로 인해 대규모 인스턴스 세그멘테이션에 취약하며, 기존 트랜스포머 기반 방식은 사용자 클릭만을 쿼리로 사용하기 때문에 포인트 클라우드 전체를 커버하는 능력이 부족하여 규모 차이 문제를 해결하지 못했다. 따라서 본 연구의 목표는 사물(Thing)과 환경(Stuff) 모두를 정확하게 세그멘테이션할 수 있는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **쿼리의 양을 늘리고 주의 집중(Attention)의 범위를 확장**하여 인스턴스의 크기에 상관없이 일관된 성능을 유지하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1.  **Query Augmentation Module**: 사용자 클릭 외에 전역 쿼리 샘플링 전략(Global Query Sampling)을 통해 추가적인 쿼리를 생성함으로써, 인스턴스의 크기가 커지더라도 충분한 수의 쿼리가 해당 영역을 커버할 수 있도록 한다.
2.  **Global Attention**: 쿼리 증강으로 인해 발생할 수 있는 오탐(False Positive)을 방지하기 위해, 로컬 어텐션 대신 전역 어텐션을 사용하여 쿼리들이 전역적인 정보를 교환하고 마스크를 최적화하도록 한다.
3.  **ClickFormer**: 위 요소들을 통합하여 실내외 환경 및 오픈 월드 설정에서 적은 수의 클릭만으로도 높은 정확도를 보이는 대화형 세그멘테이션 모델을 제안한다.

## 📎 Related Works

**1. 포인트 클라우드 세그멘테이션**
기존의 완전 지도 학습 방식은 방대한 양의 어노테이션 데이터가 필요하다는 한계가 있으며, 이를 해결하기 위한 약지도 또는 준지도 학습 방법들이 제안되었으나, 이는 주로 성능 향상에 초점을 맞추었을 뿐 새로운 장면이나 미학습 카테고리로의 일반화 문제는 충분히 다루지 않았다.

**2. 대화형 이미지 세그멘테이션**
2D 영역에서는 클릭 기반의 인터랙티브 세그멘테이션(예: SAM)이 활발히 연구되었으나, 데이터 분포의 본질적인 차이로 인해 이를 3D 영역으로 직접 전이하는 것은 한계가 있다.

**3. 대화형 포인트 클라우드 세그멘테이션**
기존 연구들은 주로 RGB 이미지의 클릭을 이용하거나(LiDAR 데이터 적용 불가), 클릭을 포인트 클라우드의 추가 속성으로 인코딩하는 방식(CRSNet), 또는 3D 볼륨으로 인코딩하는 방식(InterObject3D)을 사용했다. 하지만 이러한 방법들은 모두 수용 영역의 한계로 인해 'Stuff' 카테고리와 같은 대규모 인스턴스를 세그멘테이션하는 데 어려움을 겪는다.

## 🛠️ Methodology

### 전체 시스템 구조
ClickFormer는 크게 세 가지 구성 요소로 이루어진다.
1.  **Feature Encoder**: 입력 포인트 클라우드를 복셀 특징(Voxel Features)으로 인코딩한다. 실외 장면에는 GD-MAE를, 실내 장면에는 MinkowskiUNet을 사용한다.
2.  **Query Augmentation Module**: 사용자 클릭과 전역 샘플링된 증강 쿼리를 인코딩한다.
3.  **Mask Decoder**: Query-Voxel Transformer와 Mask Segmentation Module로 구성되어 최종 바이너리 마스크를 생성한다.

### 상세 구성 요소 및 알고리즘

#### 1. Query Augmentation Module
사용자의 클릭 $S = \{s_1, s_2, \dots, s_t\}$를 각각 독립적인 쿼리로 인코딩한다. 여기서 각 클릭 $s_t$는 좌표 $\text{pos}_t$와 양성/음성 여부를 나타내는 $\text{sgn}_t$를 가진다. 여기에 더해 **Farthest Point Sampling(FPS)**을 사용하여 포인트 클라우드 전체에서 균일하게 분포된 **증강 쿼리(Augmentation Queries)**를 추가로 샘플링한다.

각 쿼리 $q_k$는 다음과 같이 정의된다.
$$q_k = (c_k, pe_k) = 
\begin{cases} 
(v_k + v_{pos}, pe_k) & \text{for positive clicks} \\
(v_k + v_{neg}, pe_k) & \text{for negative clicks} \\
(v_k, pe_k) & \text{for augmentation points}
\end{cases}$$
여기서 $pe_k$는 Fourier positional embedding을 통한 위치 정보이며, $v_{pos}, v_{neg}$는 클릭의 성격을 구분하는 학습 가능한 벡터이다. $v_k$는 어텐션 메커니즘을 통해 마스크 임베딩을 추출하는 학습 가능 벡터이다.

#### 2. Mask Decoder (Two-way Query-voxel Transformer)
쿼리와 복셀 임베딩이 서로 정보를 업데이트하는 양방향 어텐션 구조를 가진다. 한 레이어는 다음 4단계를 거친다.
- **Q2Q (Query-to-Query)**: 쿼리 간 self-attention을 통해 증강 쿼리가 클릭 쿼리로부터 힌트를 얻는다.
- **Q2V (Query-to-Voxel)**: 쿼리가 복셀 임베딩을 통해 로컬 특징을 추출하여 자신의 내용을 업데이트한다.
- **MLP**: 각 쿼리를 업데이트한다.
- **V2Q (Voxel-to-Query)**: 쿼리의 프롬프트 정보를 복셀 임베딩에 반영하여 복셀 특징을 '클릭 인식(click-aware)' 상태로 만든다.

특히, 모든 어텐션 층에서 **Global Attention**을 사용하여 쿼리가 로컬 영역에만 매몰되어 발생하는 오탐을 억제한다.

#### 3. Mask Segmentation Module
최종적으로 포인트별 세그멘테이션을 수행한다. 포인트 $i$의 특징 $x_i$를 다음과 같이 추출한다.
$$x_i = f_{extractor}([x_{voxel}, pe_{rel}])$$
이후 $k$번째 쿼리와의 유사도를 통해 해당 포인트가 마스크에 속할 확률을 계산한다.
$$\text{mask}_{i,k} = \sigma(\langle x_i, c_k \rangle)$$
또한, 쿼리 콘텐츠 $c_k$를 통해 해당 쿼리가 전경(Foreground)일 확률 $\text{prob}_k$를 예측한다.
$$\text{prob}_k = f_{predictor}(c_k)$$
최종 점수 $\text{score}_i$는 모든 쿼리의 예측값을 합산하여 결정된다.
$$\text{score}_i = \sum_k \text{mask}_{i,k} \cdot \text{prob}_k$$

### 학습 절차 및 손실 함수
학습에는 예측 마스크 $M$과 정답 마스크 $M_{gt}$ 사이의 **Binary Cross-Entropy (BCE) Loss**를 사용한다.
$$L = w_{class} [\lambda_{fg} L_{BCE}(M_{fg}, M_{gt}) + \lambda_{bg} L_{BCE}(M_{bg}, M_{gt})]$$
- $w_{class}$: 클래스 간 불균형을 맞추기 위한 가중치
- $\lambda_{fg}, \lambda_{bg}$: 전경과 배경의 양적 차이를 조절하는 가중치

사용자 클릭은 학습 단계에서 실제 데이터를 수집하기 어려우므로, 정답 마스크와 배경에서 무작위로 샘플링하는 시뮬레이션 방식을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 실외(nuScenes, SemanticKITTI, KITTI360), 실내(ScanNet, S3DIS)
- **비교 대상**: CRSNet, InterObject3D
- **측정 지표**: $\text{IoU@k}$ 및 카테고리별 평균인 $\text{mIoU@k}$ (k는 클릭 횟수)

### 주요 결과
1.  **실외 데이터셋 (nuScenes $\rightarrow$ nuScenes)**:
    - ClickFormer는 모든 클릭 횟수에서 베이스라인을 압도한다.
    - 특히 **Stuff 카테고리**에서 성능 향상이 극명하다. 단 1번의 클릭만으로 $\text{mIoU} \ 48.58\%$를 달성했으며, 이는 베이스라인들이 10번의 클릭을 사용했을 때의 성능보다 높다.
2.  **일반화 성능 (SemanticKITTI $\rightarrow$ KITTI360)**:
    - 학습하지 않은 새로운 도메인의 장면에서도 Thing($59.34\%$)과 Stuff($58.43\%$) 모두에서 높은 mIoU(10 clicks 기준)를 기록하며 강력한 일반화 능력을 보였다.
3.  **실내 데이터셋 (ScanNet $\rightarrow$ ScanNet)**:
    - 실내에서도 Stuff 카테고리에 대해 1회 클릭 시 $52.60\%$, 10회 클릭 시 $70.00\%$의 mIoU를 기록하며 기존 방법론 대비 월등한 성능을 보였다.

### Ablation Study (nuScenes 기준)
- **Augmentation Queries 제거**: Thing과 Stuff 모두에서 $\text{mIoU}$가 약 $15\%$ 감소하여, 규모 차이 해결에 필수적임을 입증했다.
- **V2Q Cross-attention 제거**: Thing 카테고리의 성능이 약 $25\%$ 급감했다. 이는 Thing의 특징 분산이 크기 때문에 쿼리 정보를 복셀에 반영하는 과정이 매우 중요함을 시사한다.
- **Global Attention $\rightarrow$ Local Attention**: Thing 카테고리에서 성능 저하가 뚜렷하게 나타났다. 이는 전역 정보 교환이 오탐을 줄이는 데 핵심적임을 보여준다.

## 🧠 Insights & Discussion

본 논문은 대화형 세그멘테이션에서 간과되었던 **'인스턴스 규모의 다양성'**이라는 문제를 명확히 정의하고 이를 해결하기 위한 구조적 대안을 제시했다. 

**강점**은 단순히 모델의 용량을 키운 것이 아니라, FPS 기반의 전역 쿼리 증강을 통해 인스턴스 크기에 관계없이 일정한 수용 영역을 확보했다는 점이다. 또한, 쿼리 수를 일반적인 트랜스포머 모델보다 훨씬 적게 유지함으로써 계산 비용을 낮추고, 그 여유분으로 Global Attention을 적용해 정확도를 높인 전략이 효율적이었다.

**한계 및 논의점**은 다음과 같다. 
- 증강 쿼리의 수를 2배로 늘렸을 때 성능이 소폭 향상되었으나 계산 비용이 증가하는 트레이드-오프가 존재한다. 최적의 쿼리 수를 결정하는 일반적인 기준에 대한 논의가 더 필요하다.
- 현재의 클릭 시뮬레이션 방식이 실제 인간의 정교한 인터랙션 패턴을 완전히 대체할 수 있는지에 대한 검증이 추가된다면 더 설득력 있는 연구가 될 것이다.

## 📌 TL;DR

본 논문은 포인트 클라우드 대화형 세그멘테이션에서 사물(Thing)과 환경(Stuff) 간의 극심한 크기 차이로 인해 발생하는 성능 저하 문제를 해결하기 위해 **ClickFormer**를 제안한다. **전역 쿼리 증강(Query Augmentation)**과 **전역 어텐션(Global Attention)**을 도입하여 인스턴스 크기에 상관없이 정밀한 세그멘테이션이 가능하게 했으며, 실내외 다양한 데이터셋에서 기존 SOTA 모델들을 크게 상회하는 성능과 우수한 일반화 능력을 입증하였다. 이 연구는 향후 3D 장면의 효율적인 데이터 어노테이션 도구 개발에 기여할 가능성이 매우 높다.