# Deep Spectral Improvement for Unsupervised Image Instance Segmentation

Farnoosh Arefi, Amir M. Mansourian, Shohreh Kasaei (2024)

## 🧩 Problem to Solve

본 논문은 지도 학습(Supervised Learning) 기반의 인스턴스 분할(Instance Segmentation)이 요구하는 방대한 양의 픽셀 단위 주석(Dense Annotation) 비용 문제를 해결하기 위해 비지도 학습(Unsupervised Learning) 관점의 접근 방식을 다룬다. 특히, 최근 주목받는 Deep Spectral Methods(DSM)가 객체 탐지(Localization)나 시맨틱 분할(Semantic Segmentation)에서는 성과를 거두었으나, 개별 객체를 구분해야 하는 인스턴스 분할 작업에서는 상대적으로 연구가 부족했다는 점에 주목한다.

저자들은 자가 지도 학습(Self-supervised Learning) 백본에서 추출된 특징 맵(Feature Map)의 모든 채널이 인스턴스 분할에 유용한 것은 아니며, 일부 채널은 노이즈를 포함하여 오히려 성능을 저하시킨다는 문제를 제기한다. 또한, 기존 DSM에서 널리 사용되는 내적(Dot Product) 기반의 유사도 측정 방식이 특징 값의 극단적인 변화에 민감하고 특징의 분포를 무시하기 때문에, 인스턴스 분할 시 부정확한 세그먼트를 생성하는 한계가 있음을 지적한다. 따라서 본 논문의 목표는 노이즈 채널을 효과적으로 제거하고, 인스턴스 간의 구분을 명확히 할 수 있는 새로운 유사도 지표를 제안하여 비지도 인스턴스 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특징 맵의 채널 선택 과정을 최적화하고, 값 중심의 유사도 측정에서 분포 중심의 유사도 측정으로 전환하는 것이다.

첫째, 채널의 엔트로피(Entropy)를 이용해 노이즈 채널을 제거하는 Noise Channel Reduction(NCR)과 표준편차(Standard Deviation)를 이용해 인스턴스 구분 능력이 낮은 채널을 제거하는 Deviation-based Channel Reduction(DCR) 모듈을 제안한다. 이를 통해 인스턴스 분할에 가장 적합한 '유망한 채널(Promising Channel)'만을 선택적으로 활용한다.

둘째, 특징 값의 단순 수치보다 특징의 분포를 반영할 수 있는 Bray-Curtis 유사도와 값의 차이를 페널티로 부여하는 Chebyshev 유사도를 결합한 Bray-curtis over Chebyshev (BoC) 지표를 제안한다. 이는 기존의 내적 방식이 가진 수치 민감도 문제를 해결하고, 동일 인스턴스 내의 픽셀들이 유사한 분포를 가지도록 유도하여 보다 강건한 유사도 행렬(Affinity Matrix)을 구축하게 한다.

## 📎 Related Works

본 연구는 자가 지도 학습과 스펙트럴 방법론이라는 두 가지 주요 흐름을 기반으로 한다.

자습 지도 학습 분야에서는 Contrastive Learning이나 Masked Autoencoder(MAE)와 같은 기법들이 발전해 왔으며, 특히 DINO(Self-distillation with NO labels)는 비지도 분할 작업에 매우 유용한 특징 맵을 생성하는 것으로 알려져 있다. 본 논문은 DINO를 백본으로 사용하여 별도의 레이블 없이도 객체의 의미적 구조를 추출한다.

스펙트럴 방법론에서는 그래프 이론을 이미지 분할에 적용하여, 픽셀 간의 유사도를 가중치로 하는 그래프의 라플라시안(Laplacian) 행렬의 고유벡터(Eigenvector)를 통해 이미지를 분할하는 방식이 제안되었다. 특히 DSM[17]은 자가 지도 학습 특징과 스펙트럴 클러스터링을 결합하여 강력한 베이스라인을 제시하였다. 그러나 기존 연구들은 주로 전경-배경(Fg-Bg) 분리나 시맨틱 분할에 치중되어 있었으며, 인스턴스 분할을 위해 다수의 객체를 개별적으로 추출하는 정교한 채널 선택 및 유사도 측정 메커니즘은 부족한 상태였다.

## 🛠️ Methodology

### 전체 파이프라인

본 논문의 인스턴스 분할 프로세스는 다음과 같은 단계로 구성된다:

1. DINO 백본을 통해 특징 맵 $F$를 추출한다.
2. NCR 모듈을 통해 노이즈 채널을 제거하여 안정화된 특징 맵 $F'$를 얻고, 이를 이용해 전경-배경 분할 마스크를 생성한다.
3. $F'$에 DCR 모듈을 적용하여 인스턴스 구분 능력이 높은 채널만 남긴 $F''$를 생성한다.
4. $F''$와 전경 마스크를 곱하여 배경을 제거한 후, BoC 지표를 통해 유사도 행렬(Affinity Matrix)을 구축한다.
5. 라플라시안 행렬의 고유벡터를 이용해 픽셀들을 클러스터링함으로써 최종 인스턴스 마스크를 추출한다.

### Noise Channel Reduction (NCR)

NCR은 채널의 무작위성(Randomness)을 측정하는 엔트로피를 기준으로 채널을 필터링한다. 먼저 각 채널 $c$에 대해 히스토그램을 기반으로 확률 분포 함수(PDF)를 계산한다:
$$\text{PDF}(c) = \frac{\text{Hist}(c)}{H \times W}$$
이후, 다음과 같이 엔트로피를 계산하여 값이 낮은(안정적인) 상위 $M$개의 채널만 유지한다:
$$\text{Entropy}(c) = -\sum_{b=1}^{B} \text{PDF}_b(c) \cdot \log_2(\text{PDF}_b(c))$$
여기서 $B$는 빈(bin)의 개수이며 본 논문에서는 30으로 설정하였다.

### Deviation-based Channel Reduction (DCR)

DCR은 인스턴스 간의 변별력을 높이기 위해 표준편차(STD)가 높은 채널을 선택한다. NCR을 거친 $F'$에서 각 채널의 표준편차를 계산한다:
$$\text{STD}(c) = \sqrt{\frac{1}{H \times W} \sum_{x=1}^{H \times W} (x - \bar{x})^2}$$
표준편차가 낮은 채널은 인스턴스 간의 값 차이가 적어 구분 능력이 떨어지므로, STD가 높은 상위 $N$개의 채널만을 선택하여 $F''$를 구성한다.

### Bray-curtis over Chebyshev (BoC) Metric

기존의 내적 방식은 특정 픽셀의 값이 비정상적으로 높거나 낮을 때 유사도 결과가 왜곡되는 문제가 있다. 이를 해결하기 위해 분포 기반의 Bray-Curtis 유사도와 거리 기반의 Chebyshev 유사도를 결합한다.

먼저, 두 특징 벡터 $U, T$ 사이의 Bray-Curtis 거리와 유사도는 다음과 같다:
$$\text{BC}_{\text{diss}} = \frac{\sum |u_i - t_i|}{\sum |u_i + t_i|}, \quad \text{BC}_{\text{sim}} = \frac{1}{1 + \text{BC}_{\text{diss}}}$$
다음으로, 두 벡터 간의 최대 차이를 이용하는 Chebyshev 유사도는 다음과 같다:
$$\text{CH}_{\text{diss}} = \max_i (|u_i - t_i|), \quad \text{CH}_{\text{sim}} = \frac{1}{1 + \text{CH}_{\text{diss}}}$$
최종적으로 제안하는 BoC 지표는 두 유사도의 비율로 정의된다:
$$\text{BoC} = \frac{\text{BC}_{\text{sim}}}{\text{CH}_{\text{sim}}}$$
이는 특징의 분포적 유사성을 우선시하되, 두 벡터 간의 최대 거리(Chebyshev distance)가 멀어질 경우 페널티를 부여하여 정교한 유사도를 측정하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: YouTube-VIS 2019, OVIS(인스턴스 분할), PascalVOC 2012, Davis 2016(전경-배경 분할).
- **지표**: 전경-배경 분할은 F-score를, 인스턴스 분할은 mIoU(mean Intersection over Union)를 사용한다.
- **백본**: ViT-s16 (DINO).

### 주요 결과

1. **전경-배경 분할**: NCR 모듈을 적용하여 엔트로피가 낮은 채널의 일부(약 1/3~1/5)만 유지했을 때, 모든 채널을 사용했을 때보다 F-score가 향상되었다. 이는 자가 지도 학습 모델이 생성하는 특징 맵에 실제 작업에 방해가 되는 노이즈 채널이 포함되어 있음을 입증한다.
2. **인스턴스 분할 성능**: 제안한 BoC 지표는 YouTube-VIS 2019에서 34.41%, OVIS에서 36.14%의 mIoU를 기록하며 기존의 내적(Dot Product) 및 L1, L2, Cosine 유사도보다 우수한 성능을 보였다.
3. **강건성 분석**:
   - **폐색(Occlusion)**: MBOR 지표가 높은(폐색이 심한) 상황에서도 BoC는 타 지표보다 높은 mIoU를 유지하며 강건함을 보였다.
   - **크기 비율(Size Ratio)**: 객체 간 크기 차이가 큰 경우에도 작은 객체가 큰 객체에 흡수되는 현상을 효과적으로 억제하였다.
   - **인스턴스 간 거리**: 인스턴스들이 서로 밀접하게 붙어 있는 상황에서 BoC의 성능 향상 폭이 더욱 두드러졌다.

### Ablation Study

구성 요소별 기여도를 분석한 결과, Baseline(31.75%) $\rightarrow$ NCR 추가(32.92%) $\rightarrow$ NCR+BoC 추가(33.62%) $\rightarrow$ NCR+DCR+BoC 모두 적용(34.41%) 순으로 성능이 향상되었다. 이는 각 모듈이 상호 보완적으로 작동하여 최종 성능을 극대화함을 보여준다.

## 🧠 Insights & Discussion

본 논문은 비지도 인스턴스 분할에서 특징 선택(Channel Reduction)과 유사도 측정(Similarity Metric)의 중요성을 성공적으로 증명하였다. 특히, 단순한 수치적 내적이 아니라 특징의 '분포'를 고려하는 BoC 지표가 인스턴스 분할의 본질적인 문제(동일 객체 내의 변동성 처리)를 잘 해결했다는 점이 인상적이다.

다만, 몇 가지 한계점과 논의할 점이 존재한다. 첫째, NCR과 DCR에서 사용되는 하이퍼파라미터 $M$과 $N$이 데이터셋에 따라 최적값이 다를 수 있으며, 이를 자동으로 결정하는 메커니즘은 제시되지 않았다. 둘째, Ablation Study에서 NCR과 DCR만 함께 적용했을 때 오히려 성능이 소폭 하락(32.92% $\rightarrow$ 32.70%)하는 현상이 발견되었는데, 이는 두 모듈 간의 상호작용에 대한 추가적인 분석이 필요함을 시사한다. 마지막으로, 제안된 방법론은 DINO와 같은 강력한 사전 학습 모델에 의존하고 있어, 백본 모델의 성능 변화가 최종 결과에 미치는 영향이 매우 클 것으로 예상된다.

## 📌 TL;DR

본 연구는 비지도 인스턴스 분할 성능을 높이기 위해 **노이즈 채널 제거(NCR)**, **변별력 낮은 채널 제거(DCR)**, 그리고 분포 기반의 **새로운 유사도 지표(BoC)**를 제안하였다. 이를 통해 기존의 내적 기반 DSM 방식보다 폐색이나 크기 차이가 심한 까다로운 조건에서도 더 정확하게 개별 객체를 분리해낼 수 있었으며, 최종적으로 mIoU 성능 향상을 달성하였다. 이 연구는 향후 레이블 없는 이미지에서의 정밀한 객체 분리 및 분석 연구에 중요한 기초가 될 가능성이 높다.
