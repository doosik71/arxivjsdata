# Embedding-based Instance Segmentation in Microscopy

Manan Lalit, Pavel Tomancak, and Florian Jug (2021)

## 🧩 Problem to Solve

본 논문은 2D 및 3D 현미경 이미지 데이터에서 객체의 자동 탐지 및 인스턴스 분할(Instance Segmentation) 문제를 해결하고자 한다. 생물 의학 분야에서 현미경 데이터의 객체 분할은 매우 중요하지만, 기존의 딥러닝 기반 접근 방식들은 다음과 같은 한계를 지닌다.

첫째, Mask R-CNN과 같은 Top-down 방식은 축 정렬 바운딩 박스(axis-aligned bounding boxes)를 사용하는데, 현미경 이미지 내의 객체들은 자연 이미지와 달리 형태가 매우 복잡하고 방향이 무작위적이어서 성능이 저하된다. 둘째, StarDist와 같은 Bottom-up 방식은 객체가 별 모양으로 볼록(star-convex)하다는 가정을 전제로 하며, 이 가정을 벗어나는 복잡한 형태의 객체 분할 시 오류가 발생한다. 셋째, 대부분의 기존 방법론이 2D 데이터에 치중되어 있으며, 3D 볼륨 데이터(volumetric data)를 직접 처리하는 모델은 드물고, 존재하더라도 막대한 GPU 메모리를 요구하여 접근성이 떨어진다.

따라서 본 연구의 목표는 복잡한 형태의 객체를 효과적으로 분할할 수 있으며, 2D와 3D 데이터 모두에 적용 가능하고, 저사양 GPU에서도 학습 가능한 효율적인 인스턴스 분할 방법론인 EMBEDSEG를 제안하는 것이다.

## ✨ Key Contributions

EMBEDSEG의 핵심 아이디어는 각 픽셀이 자신이 속한 객체를 대표하는 고유한 공간 임베딩(Spatial Embedding) 위치를 예측하도록 하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Medoid 기반 임베딩**: 기존 연구(Neven et al.)가 중심점(Centroid)을 사용한 것과 달리, 객체 내부의 실제 픽셀 중 다른 모든 픽셀과의 평균 거리가 최소인 지점인 Medoid를 임베딩 타겟으로 설정하였다. 이는 임베딩 지점이 객체 외부로 벗어나 발생하는 시드 점수(seediness score) 저하 및 밀집 지역에서의 중복 문제를 해결한다.
2. **3D 확장 및 효율성**: 2D뿐만 아니라 3D 볼륨 데이터에 직접 적용 가능한 Branched ERF-Net 구조를 제안하였으며, 매우 적은 GPU 메모리 점유율을 유지하여 노트북 수준의 하드웨어에서도 학습이 가능하도록 설계하였다.
3. **Test-Time Augmentation (TTA)**: 추론 단계에서 축 정렬 회전 및 반전을 이용한 TTA를 적용하여 분할 정확도를 향상시켰다.
4. **새로운 3D 데이터셋 공개**: 생물 의학적으로 유의미한 4가지 새로운 3D 현미경 데이터셋과 이에 대한 정답 라벨(Ground Truth)을 공개하여 후속 연구의 기반을 마련하였다.

## 📎 Related Works

논문은 인스턴스 분할 방법을 Top-down과 Bottom-up으로 구분하여 설명한다.
- **Top-down 방식**: Mask R-CNN 등이 대표적이며, 바운딩 박스로 객체를 먼저 탐지한 후 마스크를 생성한다. 하지만 현미경 이미지의 무작위적인 객체 방향성과 복잡한 형태 때문에 성능 한계가 명확하다.
- **Bottom-up 방식**: 각 픽셀이 클래스나 형태를 예측하고 나중에 이를 통합한다. StarDist는 별 모양 볼록성(star-convexity) 가정을 통해 성능을 높였으나, 복잡한 형태의 객체에는 취약하다.
- **임베딩 기반 방식**: 각 픽셀이 태그나 임베딩을 예측하여 같은 객체에 속한 픽셀들이 유사한 값을 갖도록 유도한다. Neven et al.의 연구는 공간 임베딩과 클러스터링 대역폭(bandwidth)을 공동 최적화하는 방식을 제안하였으며, EMBEDSEG는 이를 계승하고 발전시킨 모델이다.

## 🛠️ Methodology

### 1. 시스템 구조
EMBEDSEG는 **Branched ERF-Net** 아키텍처를 사용한다. 이 네트워크는 입력 이미지에 대해 다음과 같은 세 가지 값을 각 픽셀 $\vec{x}_i$마다 예측한다.
- **오프셋 벡터 $\vec{o}_i$**: 픽셀 $\vec{x}_i$를 객체의 대표 지점인 임베딩 위치 $\vec{e}_i = \vec{x}_i + \vec{o}_i$로 매핑한다.
- **불확실성 벡터 $\vec{\sigma}_i$**: 예측된 임베딩 $\vec{e}_i$와 실제 타겟 $\vec{e}_k$ 사이의 오차를 추정하는 클러스터링 대역폭(bandwidth)이다.
- **시드성 점수(Seediness score) $s_i$**: 해당 픽셀이 객체의 중심(대표 지점)일 가능성을 나타낸다.

### 2. 학습 목표 및 손실 함수
전체 손실 함수 $L$은 세 가지 손실의 가중 합으로 정의된다.
$$L = w_{seed}L_{seed} + w_{IoU}L_{IoU} + w_{var}L_{var}$$

- **$L_{IoU}$**: 예측된 인스턴스 분할 결과와 정답 마스크 간의 IoU를 최대화하기 위해 미분 가능한 Lovász-Softmax 손실을 사용한다.
- **$L_{seed}$**: 예측된 시드성 점수 $s_i$가 가우시안 함수 $\phi_k$에 기반한 확률값과 일치하도록 학습시킨다.
  $$\phi_k(\vec{e}_i) = \exp \left( -\frac{1}{2} (\vec{e}_i - \vec{C}_k)^T \Sigma_k^{-1} (\vec{e}_i - \vec{C}_k) \right)$$
  여기서 $\vec{C}_k$는 객체의 대표 지점(Medoid)이며, $\Sigma_k$는 대각 공분산 행렬이다.
- **$L_{var}$**: 객체 내의 모든 픽셀이 예측하는 대역폭 $\vec{\sigma}_i$가 해당 객체의 평균 대역폭 $\vec{\sigma}_k$와 유사해지도록 하는 평활화(smoothness) 손실이다.
  $$L_{var} = \frac{1}{|S_k|} \sum_{\vec{\sigma}_i \in S_k} \|\vec{\sigma}_i - \vec{\sigma}_k\|^2$$

### 3. 추론 절차
학습된 모델을 통한 인스턴스 추출 과정은 다음과 같은 탐욕적(greedy) 방식으로 진행된다.
1. 시드성 점수 $s_i$가 임계값 $s_{fg}$보다 큰 픽셀들을 전경(foreground) 집합 $S_{fg}$로 수집한다.
2. $S_{fg}$ 내에서 시드성 점수가 가장 높은 픽셀 $\vec{x}_{seed}$를 선택한다.
3. $\vec{x}_{seed}$의 임베딩 위치 $\vec{e}_{seed}$와 대역폭 $\vec{\sigma}_{seed}$를 기준으로 임베딩 가능성(likelihood)이 $0.5$보다 큰 모든 전경 픽셀을 하나의 인스턴스 $S_k$로 묶는다.
4. 선택된 픽셀들을 $S_{fg}$에서 제거하고, 더 이상 유효한 시드 픽셀이 없을 때까지 반복한다.

## 📊 Results

### 1. 실험 설정
- **2D 데이터셋**: BBBC010, Usiigaci, DSB
- **3D 데이터셋**: Mouse-Organoid-Cells-CBG, Platynereis-Nuclei-CBG, Mouse-Skull-Nuclei-CBG, Platynereis-ISH-Nuclei-CBG (신규 공개)
- **비교 대상**: Cellpose, StarDist (및 StarDist-3D), Mask R-CNN, 3-Class U-Net, Neven et al.
- **지표**: IoU 임계값에 따른 Mean Average Precision ($\text{AP}_{dsb}$)

### 2. 주요 결과
- **정량적 성능**: EMBEDSEG는 대부분의 2D 및 3D 데이터셋에서 기존 베이스라인보다 높은 AP를 기록하였다. 특히 IoU 임계값이 높은(0.7 이상) 엄격한 기준에서도 StarDist나 Cellpose보다 우수한 성능을 보였다.
- **3D 데이터 분석**: 3D 데이터셋에서 Cellpose는 2D 슬라이스 예측을 보간하는 방식의 한계로 인해 과분할(over-segmentation) 문제가 발생했으나, EMBEDSEG는 이를 극복하였다.
- **메모리 효율성**: 학습 시 GPU 메모리 사용량이 매우 낮아(대부분 3GB 미만), 고가의 하드웨어 없이도 학습이 가능하다는 점이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 공간 임베딩 기반의 분할 방식이 현미경 이미지의 복잡한 기하학적 특성을 처리하는 데 매우 유효함을 입증하였다. 

특히 StarDist-3D와의 비교를 통해 중요한 통찰을 제시한다. StarDist-3D는 객체가 별 모양 볼록성을 띨 때는 매우 강력하지만, 그렇지 않은 경우 성능이 급감한다. 또한 StarDist-3D는 예측된 벡터들로 구성된 면(face)이 평면적(planarity)인 특성이 있어, 실제 객체의 매끄러운 곡면을 정확히 묘사하지 못하며, 이는 높은 IoU 임계값에서 AP 점수가 낮아지는 원인이 된다.

반면, EMBEDSEG는 Medoid를 사용함으로써 임베딩 지점이 객체 내부로 강제되어 안정적인 시드 추출이 가능해졌고, TTA를 통해 추론의 강건성을 확보하였다. 다만, 아주 단순한 구형/별 모양의 핵(nuclei) 데이터셋에서는 StarDist-3D가 대등하거나 더 나은 성능을 보일 수 있다는 점이 한계로 언급되었다.

## 📌 TL;DR

본 연구는 2D 및 3D 현미경 이미지의 복잡한 객체 분할을 위해 **Medoid 기반의 공간 임베딩 방법론인 EMBEDSEG**를 제안하였다. 이 모델은 기존의 별 모양 볼록성 가정을 탈피하여 복잡한 형태의 객체도 정밀하게 분할하며, 낮은 GPU 메모리 사용량으로 접근성을 높였다. 또한 4개의 새로운 3D 데이터셋을 공개함으로써 생물 의학 영상 분석 분야의 발전에 기여하였다. 향후 복잡한 구조의 세포 및 조직 분할 작업에서 매우 중요한 역할을 할 것으로 기대된다.