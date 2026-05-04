# 3D Instance Segmentation via Multi-Task Metric Learning

Jean Lahoud, Bernard Ghanem, Marc Pollefeys, Martin R. Oswald (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 밀집된 3D voxel grid 상에서의 인스턴스 라벨 분할(instance label segmentation)이다. 구체적으로는 뎁스 센서나 Multi-view Stereo 방식으로 획득되어 시맨틱 3D 재구성(semantic 3D reconstruction) 또는 장면 완성(scene completion) 처리가 완료된 볼륨 표현(volumetric representations)을 대상으로 한다.

3D 인스턴스 분할은 개별 객체의 형태 정보를 학습하여 서로 다른 인스턴스를 정확하게 분리하는 것을 목표로 하며, 특히 서로 맞닿아 있거나 불완전하게 스캔된 객체들을 분리하는 것이 핵심적인 도전 과제이다. 2D 인스턴스 분할에 비해 3D 영역은 가용 데이터셋이 부족하고, 기존 2D 방법론을 3D로 직접 확장하는 것이 어렵다는 점에서 연구의 중요성이 크다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 인스턴스 분할 문제를 **Multi-task Learning** 전략으로 해결하는 것이다. 단순히 하나의 특징을 학습하는 것이 아니라, 다음과 같은 두 가지 상호 보완적인 목표를 동시에 학습한다.

1. **Feature Embedding 학습**: 동일한 인스턴스에 속하는 voxel들은 임베딩 공간에서 서로 가깝게 위치하고, 서로 다른 인스턴스의 voxel들은 멀리 떨어지도록 하는 metric learning을 수행한다.
2. **Directional Information 추정**: 각 voxel에서 해당 인스턴스의 질량 중심(center of mass)을 향하는 방향 벡터를 밀집하게 추정한다.

이러한 다중 작업 학습은 인스턴스의 경계를 찾는 후처리 단계에서 유용하며, 특히 방향 정보 학습이 특징 임베딩 학습의 성능을 보조하여 전체적인 분할 품질을 향상시킨다.

## 📎 Related Works

논문에서는 2D 및 3D 인스턴스 분할 연구를 다음과 같이 분류하여 설명한다.

- **2D 인스턴스 분할**: Object Proposal이나 Detection 기반 방식(Mask R-CNN, YOLO 등)과 Metric Learning 기반 방식(DeBrabandere et al. 등)으로 나뉜다. Metric Learning 방식은 픽셀들을 임베딩 공간에 매핑하고 클러스터링하는 방식을 취하는데, 본 논문은 이 중 DeBrabandere et al.의 접근 방식을 3D로 확장하였다.
- **3D 인스턴스 분할**: SGPN과 같이 PointNet을 사용하는 방식이 있으나, 이는 유사도 행렬(similarity matrix)의 크기가 포인트 수의 제곱에 비례하여 확장성(scalability)이 떨어진다는 한계가 있다. 또한 GSPN, 3D-SIS, MASC 등 최신 연구들이 존재하지만, 본 논문은 시맨틱 분할 이후의 후처리 단계로서의 인스턴스 분할에 집중하며 3D 차원과 연결성 정보를 적극적으로 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

본 방법론은 시맨틱 분할(semantic segmentation) 이후의 후처리 단계로 동작한다. 입력으로 시맨틱 라벨이 포함된 3D voxel grid를 받고, 출력으로 **Feature Embedding** 공간과 **Directional Embedding** 공간의 두 가지 잠재 공간(latent space)을 생성한다.

### 네트워크 아키텍처

SSCNet 아키텍처를 기반으로 한 3D Convolutional Neural Network를 사용한다.

- **Upsampling**: Pooling 레이어로 인해 줄어든 해상도를 원래 크기로 복구하기 위해 Transpose Convolution(Deconvolution)을 사용한다.
- **Receptive Field 확장**: Dilated 3D Convolution 레이어를 사용하여 수용 영역을 넓혔으며, voxel 크기가 10cm일 때 최대 14.2m까지 커버할 수 있도록 설계하여 일반적인 실내 공간의 voxel들을 충분히 참조할 수 있게 하였다.

### 손실 함수 (Loss Functions)

#### 1. Feature Embedding Loss ($L_{FE}$)

특징 공간에서 동일 인스턴스는 뭉치고 서로 다른 인스턴스는 밀어내기 위해 세 가지 항의 합으로 정의한다.
$$L_{FE} = \gamma_{var} L_{var} + \gamma_{dist} L_{dist} + \gamma_{reg} L_{reg}$$

- **Intra-cluster variance ($L_{var}$)**: 동일 인스턴스 내의 특징 벡터 $x_i$와 해당 클러스터 중심 $\mu_c$ 사이의 거리를 좁힌다.
    $$L_{var} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i=1}^{N_c} [\|\mu_c - x_i\| - \delta_{var}]_+^2$$
- **Inter-cluster distance ($L_{dist}$)**: 서로 다른 클러스터 중심 $\mu_{c_A}$와 $\mu_{c_B}$ 사이의 거리를 최소 거리 $2\delta_{dist}$ 이상으로 유지하도록 밀어낸다.
    $$L_{dist} = \frac{1}{C(C-1)} \sum_{c_A=1}^{C} \sum_{c_B \neq c_A}^{C} [2\delta_{dist} - \|\mu_{c_A} - \mu_{c_B}\|]_+^2$$
- **Regularization ($L_{reg}$)**: 활성화 값의 범위를 제한하기 위해 모든 특징을 원점으로 끌어당긴다.
    $$L_{reg} = \frac{1}{C} \sum_{c=1}^{C} \|\mu_c\|$$

#### 2. Directional Loss ($L_{dir}$)

각 voxel $z_i$에서 객체 중심 $z_c$로 향하는 정규화된 방향 벡터 $v_{GT, i}$와 예측 벡터 $v_i$ 사이의 코사인 유사도를 최대화한다.
$$L_{dir} = -\frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i=1}^{N_c} v_i^\top v_{GT, i}, \quad v_{GT, i} = \frac{z_i - z_c}{\|z_i - z_c\|}$$

#### 3. Joint Loss

최종적으로 두 손실 함수를 가중합 하여 최소화한다.
$$L_{joint} = \alpha_{FE} L_{FE} + \alpha_{dir} L_{dir}$$
(설정값: $\alpha_{FE}=0.5, \alpha_{dir}=1$)

### 추론 및 후처리 절차

1. **Clustering**: Feature Embedding 결과에 대해 Mean-shift clustering을 적용하여 인스턴스 후보(proposals)를 생성한다.
2. **Scoring**: 생성된 후보들에 대해 방향 특징의 일관성(Direction feature consistency)과 특징 임베딩의 응집도(Feature embedding coherency)를 기준으로 점수를 매긴다.
3. **Refinement**: Connected Components를 통해 분할 제안을 생성하고, NMS(Non-Maximum Suppression)를 적용하여 겹치는 객체를 제거한다.
4. **Labeling**: 클러스터 내에서 가장 많이 등장하는 시맨틱 라벨을 해당 인스턴스의 라벨로 지정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 합성 데이터셋(Synthetic Toy Dataset)과 ScanNet v2 데이터셋을 사용하였다.
- **평가 지표**: $\text{IoU}$ 임계값에 따른 Average Precision ($\text{AP}_{25}, \text{AP}_{50}$)을 측정하였다.
- **비교 대상**: Input Segmentation, Connected Components, 그리고 Mask R-CNN proj, SGPN, GSPN, 3D-SIS, MASC, PanopticFusion 등 SOTA 방법론들과 비교하였다.

### 주요 결과

- **합성 데이터셋**: Multi-task learning 방식이 Feature Embedding만 사용한 방식보다 성능이 소폭 향상되었으며, Ground Truth 시맨틱 라벨을 사용한 Connected Components보다 훨씬 높은 성능을 보였다.
- **ScanNet 데이터셋**:
  - **Ablation Study**: 단일 작업(Single-task) 학습보다 다중 작업(Multi-task) 학습이 대부분의 클래스에서 일관되게 높은 성능을 보였다. 이는 방향 손실 함수가 특징 공간에서 더 변별력 있는 특징을 학습하도록 돕는다는 가설을 뒷받침한다.
  - **SOTA 비교**: $\text{AP}_{50}$ 지표에서 기존의 모든 비교 방법론을 제치고 가장 높은 평균 점수($0.55$)를 기록하며 1위를 달성하였다.

## 🧠 Insights & Discussion

### 강점

- **직접적인 3D 처리**: 2D 이미지 정보를 활용하여 3D로 전파하는 방식이 아니라, 3D voxel grid에서 직접 연산하므로 처리 속도가 빠르고 2D 데이터에 대한 의존도가 낮다.
- **상호 보완적 학습**: Metric learning의 전역적 특성과 방향 예측의 지역적 특성을 결합하여 인스턴스 분리 능력을 극대화하였다.

### 한계 및 비판적 해석

- **경계 인식의 부족**: 정성적 결과 분석에서 'desk'와 같은 객체가 분리되거나, 가구 라벨이 인접한 지오메트리로 번지는(bleed) 현상이 관찰되었다. 이는 본 방법론이 주로 기하학적 구조에 의존하기 때문이며, 더 정교한 객체 경계 인식 메커니즘이 필요함을 시사한다.
- **시맨틱 분할 의존성**: 본 모델은 시맨틱 분할의 후처리 단계로 설계되었으므로, 입력으로 들어오는 시맨틱 라벨의 품질이 최종 인스턴스 분할 결과에 직접적인 영향을 미친다.

## 📌 TL;DR

본 논문은 3D voxel grid 상의 인스턴스 분할을 위해 **특징 임베딩 학습(Metric Learning)**과 **객체 중심 방향 예측(Directional Prediction)**을 동시에 수행하는 **Multi-task Learning** 프레임워크를 제안한다. 이를 통해 동일 인스턴스는 가깝게, 서로 다른 인스턴스는 멀게 배치함과 동시에 기하학적 중심 정보를 활용하여 분할 정확도를 높였으며, ScanNet 벤치마크에서 $\text{AP}_{50}$ 기준 SOTA 성능을 달성하였다. 이 연구는 3D 장면 이해를 위한 효율적인 인스턴스 분리 방법론을 제시했다는 점에서 향후 로봇 공학이나 3D 재구성 분야에 중요한 기여를 할 가능성이 크다.
