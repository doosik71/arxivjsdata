# Nuclear Instance Segmentation using a Proposal-Free Spatially Aware Deep Learning Framework

Navid Alemi Koohbanani, Mostafa Jahanifar, Ali Gooya, and Nasir Rajpoot (2019)

## 🧩 Problem to Solve

본 논문은 조직학 이미지(histology images) 내의 핵(nuclei)을 개별적으로 분리하여 검출하는 Nuclear Instance Segmentation 문제를 해결하고자 한다. 조직학 이미지에서 핵의 형태와 외형은 매우 다양하며, 특히 여러 개의 핵이 서로 겹쳐 있거나 밀집되어 있는 경우(overlapping nuclei) 이를 개별 인스턴스로 분리하는 것이 매우 어렵다.

기존의 딥러닝 기반 방법론들은 이러한 밀집된 영역에서 개별 객체를 정확히 구분하는 능력이 부족하며, 특히 네트워크가 이미지 내의 상대적인 위치 정보, 즉 공간적 인식(spatial awareness) 능력이 결여되어 있다는 점이 주요 문제로 지적된다. 따라서 본 연구의 목표는 제안하는 Proposal-free 프레임워크와 공간 인식 네트워크(SpaNet)를 통해 계산 비용을 낮추면서도 겹쳐진 핵들을 효과적으로 분리하여 State-of-the-art(SOTA) 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Proposal-free 프레임워크 제안**: Region Proposal Network(RPN)와 같은 복잡한 단계 없이, 픽셀 단위의 위치 정보를 예측하고 이를 클러스터링하는 단순하고 효율적인 인스턴스 세그멘테이션 파이프라인을 구축하였다.
2. **Spatially Aware Network (SpaNet) 설계**: 이미지 좌표 정보를 입력으로 제공하고, 이를 네트워크 전체에 유지함으로써 공간적 인식 능력을 갖춘 새로운 아키텍처를 제안하였다.
3. **Multi-Scale Dense Unit (MSDU) 도입**: 다양한 커널 크기와 Dilation rate를 가진 병렬 컨볼루션 블록(MSB)을 Dense하게 연결하여, 국소적 구조부터 핵 전체 크기에 이르는 다중 스케일의 특징을 효과적으로 추출한다.
4. **핵 검출 맵(Detection Map) 기반 클러스터링**: 단순한 세그멘테이션 맵뿐만 아니라 가우시안 형태의 Centroid detection map을 함께 예측하여, 밀집된 핵 뭉치(clump) 내에서 실제 핵의 개수를 추정하고 정밀하게 분리한다.

## 📎 Related Works

기존의 핵 인스턴스 세그멘테이션 접근 방식은 크게 네 가지 방향으로 나뉜다.

1. **Region-proposal 기반**: Mask R-CNN이나 PA-Net과 같이 객체의 후보 영역을 먼저 제안하고 세그멘테이션을 수행하는 방식이다.
2. **Encoder-decoder 구조**: U-Net을 기반으로 하되, CIA-Net처럼 네트워크 구조를 변경하여 더 풍부한 특징을 추출하는 방식이다.
3. **보조 출력(Auxiliary output) 활용**: DCAN이나 BES-Net과 같이 핵의 외곽선(contour)이나 Bounding box를 추가로 예측하여 분리 성능을 높이는 방식이다.
4. **기하학적 매핑 예측**: DR-Net과 같이 거리 맵(distance map)을 예측하여 객체를 분리하는 방식이다.

본 논문은 위 방법들이 밀집된 핵들을 분리하는 데 필요한 **공간적 인식(spatial awareness)**이 부족하다는 점을 한계로 지적하며, Proposal 단계 없이 좌표 정보를 직접 학습하는 방식으로 차별화를 꾀한다.

## 🛠️ Methodology

### 1. SpaNet 아키텍처

SpaNet의 핵심은 공간 정보가 네트워크 전체에 흐르도록 설계된 구조이다.

* **Multi-Scale Dense Unit (MSDU)**: DenseNet의 구조에서 영감을 받았으며, 내부에 **Multi-Scale Block (MSB)**을 포함한다. MSB는 서로 다른 커널 크기($k$)와 Dilation rate($d$)를 가진 4개의 병렬 컨볼루션 층으로 구성되어 다중 해상도 특징 맵을 생성한다.
* **Transition Blocks**: DTB(Down Transitioning Block)는 $1 \times 1$ 컨볼루션과 $2 \times 2$ Average Pooling을 통해 해상도를 낮추고, UTB(Up Transitioning Block)는 Transposed Convolution을 통해 해상도를 높인다.
* **Spatial Awareness 유지**: DTB나 UTB 이후에는 직접적인 위치 정보가 손실될 수 있으므로, 스케일링된 네트워크 입력(좌표 맵)을 다시 Concatenation 하여 모든 층에서 공간 정보를 유지하게 한다.

### 2. Proposal-Free 인스턴스 세그멘테이션 파이프라인

전체 프로세스는 크게 세 단계로 진행된다.

#### 단계 1: 세그멘테이션 및 센트로이드 검출 (Dual-head Network)

먼저 두 개의 헤드를 가진 네트워크를 통해 **핵 마스크($M_{seg}$)**와 **센트로이드 검출 맵**을 예측한다. 센트로이드의 Ground Truth($G_n$)는 다음과 같은 가우시안 유사 함수로 생성한다.

$$G_n(x,y) = \begin{cases} \frac{1}{1+\beta\|(c_{nx}, c_{ny}) - (x,y)\|}, & \text{if } \|(c_{nx}, c_{ny}) - (x,y)\| \le r \\ 0, & \text{otherwise} \end{cases}$$

여기서 $(c_{nx}, c_{ny})$는 핵의 중심 좌표이며, $\beta=0.01, r=8$로 설정되었다. 손실 함수로는 Smooth Jaccard loss(마스크용)와 Mean Squared Error(검출 맵용)를 사용한다.

#### 단계 2: 위치 정보 예측 (Single-head SpaNet)

앞서 얻은 $M_{seg}$와 픽셀 좌표 맵($M_x, M_y$), 그리고 RGB 및 HSV 이미지를 입력으로 하여 각 픽셀이 속한 핵의 공간적 특성 벡터 $p_n$을 예측한다.

$$p_n = (c_{nx}/w, c_{ny}/h, l_{nx}/w, l_{ny}/h, r_{nx}/w, r_{ny}/h)$$

여기서 $(c_{nx}, c_{ny})$는 중심, $(l_{nx}, l_{ny})$는 좌상단, $(r_{nx}, r_{ny})$는 우하단 좌표이며, $w, h$는 Bounding box의 너비와 높이로 정규화한 값이다. 학습에는 배경 영역을 제외한 Smoothed $L_1$ loss를 사용한다.

#### 단계 3: 후처리 및 클러스터링

예측된 위치 정보 맵을 직접 클러스터링하는 대신, 다음과 같은 효율적인 절차를 거친다.

1. $M_{seg}$를 임계값 0.3으로 이진화하고 작은 객체를 제거하여 **핵 뭉치(Connected Components, CC)**를 찾는다.
2. 각 CC 영역 내에서 Centroid detection map의 local maxima 개수를 세어 해당 뭉치에 포함된 핵의 개수를 추정한다.
3. 추정된 개수를 기반으로 **Spectral Clustering**(RBF 커널 사용)을 적용하여 개별 핵 인스턴스를 최종적으로 분리한다.

## 📊 Results

### 실험 설정

* **데이터셋**: TCGA에서 추출한 7가지 조직(신장, 위, 간, 방광, 대장, 전립선, 간)의 이미지 30장(학습 16장, 테스트 14장)을 사용하였다. 테스트 셋은 학습 시 본 적 있는 조직(Seen)과 본 적 없는 조직(Unseen)으로 나누어 일반화 성능을 측정하였다.
* **학습 전략**: Stochastic Weight Averaging(SWA)과 Cycling Learning Rate를 적용하여 100 epoch 동안 학습하였다.
* **평가 지표**: AJI(Aggregated Jaccard Index)와 F1-score를 사용하여 정량 평가하였다.

### 결과 분석

제안 방법인 SpaNet은 기존 SOTA 모델들과 비교하여 우수한 성능을 보였다.

| Method | AJI (Seen / Unseen) | F1-score (Seen / Unseen) |
| :--- | :---: | :---: |
| CIA-Net | 61.29% / 63.06% | 82.44% / 84.58% |
| **Spa-Net (Ours)** | **62.39% / 63.40%** | **82.81% / 84.51%** |

특히 AJI 지표에서 기존 최고 성능 대비 약 1.10% 향상된 결과를 보였으며, 본 적 없는 조직(Unseen Organ)에서도 강건한 성능을 유지하였다.

## 🧠 Insights & Discussion

**강점 및 분석**:

* **다중 스케일 특징 추출**: MSDU를 통해 다양한 크기의 핵을 효과적으로 캡처할 수 있었으며, 이는 특히 크기가 제각각인 핵들이 섞여 있는 조직 이미지에서 유리하게 작용하였다.
* **모델 효율성**: SpaNet은 약 21M의 파라미터를 가지며, 이는 U-Net($\sim 31$M)이나 CIA-Net($\sim 40$M)보다 훨씬 적은 수치이다. 파라미터 수가 적음에도 불구하고 더 높은 성능을 낸 것은 모델의 구조적 효율성이 높고, 적은 데이터셋 환경에서 Overfitting을 방지하여 일반화 성능이 향상되었음을 시사한다.
* **분리된 네트워크 구성**: 검출 맵 예측과 위치 정보 예측을 별도의 네트워크로 구성하는 것이 단일 멀티태스크 네트워크보다 성능이 좋았음을 실험적으로 확인하였다.

**한계 및 논의**:

* 본 연구에서 사용한 데이터셋의 규모(30장)가 매우 작아, 더 대규모의 데이터셋에서도 동일한 일반화 성능이 유지될지는 추가 검증이 필요하다.
* Spectral Clustering은 효과적이지만, 픽셀 수가 매우 많은 경우 계산 복잡도가 증가할 수 있다. 이를 해결하기 위해 CC 단위로 나누어 처리하는 방식을 택했으나, 매우 큰 CC가 존재할 경우 효율성이 떨어질 가능성이 있다.

## 📌 TL;DR

본 논문은 조직학 이미지에서 겹쳐진 핵들을 정밀하게 분리하기 위해 **공간 인식 능력을 갖춘 Proposal-free 인스턴스 세그멘테이션 프레임워크(SpaNet)**를 제안한다. MSDU라는 다중 스케일 구조를 통해 공간 정보를 유지하며, [세그멘테이션 $\rightarrow$ 위치 정보 예측 $\rightarrow$ 센트로이드 기반 클러스터링]으로 이어지는 파이프라인을 통해 계산 비용을 줄이면서도 SOTA 성능을 달성하였다. 특히 적은 파라미터 수로도 높은 일반화 성능을 보여, 의료 영상 분석 분야에서 효율적인 인스턴스 분리 모델의 가능성을 제시하였다.
