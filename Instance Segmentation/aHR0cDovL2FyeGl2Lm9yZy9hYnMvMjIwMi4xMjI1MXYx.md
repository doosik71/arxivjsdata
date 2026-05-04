# ISDA: POSITION-AWARE INSTANCE SEGMENTATION WITH DEFORMABLE ATTENTION

Kaining Ying, Zhenhua Wang, Cong Bai, Pengfei Zhou (2022)

## 🧩 Problem to Solve

기존의 인스턴스 세그멘테이션(Instance Segmentation) 모델들은 전처리 단계의 제안 영역 추출(Proposal Estimation, 예: RPN)이나 후처리 단계의 비최대 억제(Non-Maximum Suppression, NMS) 과정으로 인해 완전한 엔드-투-엔드(end-to-end) 학습이 불가능하다는 한계가 있다. 특히 NMS는 미분 불가능한 연산으로, 그래디언트의 역전파를 방해하며 전체 파이프라인의 최적화를 저해한다. 

본 논문의 목표는 이러한 수동 설계된 컴포넌트와 후처리 과정을 제거하고, 학습 가능한 쿼리와 어텐션 메커니즘을 통해 NMS 없이도 정교한 마스크를 예측할 수 있는 엔드-투-엔드 인스턴스 세그멘테이션 프레임워크인 ISDA를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 인스턴스 세그멘테이션 작업을 '객체 마스크 세트'를 예측하는 문제로 재정의하는 것이다. 이를 위해 Deformable Attention 네트워크를 활용하여 객체별 특징(Object Features)과 위치 인식 커널(Position-aware Kernels)을 효율적으로 학습하고, 이를 마스크 특징 표현(Mask Feature Representation, MFR)과 결합하여 마스크를 생성한다.

주요 기여 사항은 다음과 같다.
1. Deformable Transformer 기반의 프레임워크를 도입하여 객체 쿼리(Object Queries)를 데이터로부터 효율적으로 학습함으로써 NMS가 필요 없는 엔드-투-엔드 학습을 가능하게 하였다.
2. 외형이 유사한 객체들을 효과적으로 구분하기 위해, 객체의 참조점(Reference Point)을 결합한 위치 인식 커널 생성 방법을 제시하였다.
3. MS-COCO 데이터셋에서 Mask R-CNN 및 최신 SOTA 모델 대비 우수한 성능을 입증하였다.

## 📎 Related Works

최근 컴퓨터 비전 분야에서는 Transformer 구조를 활용한 엔드-투-엔드 객체 검출 모델인 DETR이 제안되었으나, Dense Attention으로 인한 높은 계산 비용과 단일 스케일 특징 맵 사용으로 인한 작은 객체 검출 성능 저하라는 문제가 있었다. 이를 해결하기 위해 Sparse Attention과 멀티 스케일 특징을 사용하는 Deformable DETR이 등장하였으며, ISDA는 이 구조에서 영감을 얻었다.

또한, SOLOv2와 같이 데이터로부터 객체 커널을 학습하는 방식이 존재한다. 하지만 SOLOv2는 고정된 그리드 셀마다 커널을 예측하는 방식인 반면, ISDA는 학습 가능한 객체 쿼리를 사용하여 더 유연한 적응력을 가진다. 또한 SOLOv2가 전이 불변성(Translation-invariance) 문제를 해결하기 위해 상대 좌표 채널을 추가한 것과 달리, ISDA는 학습된 객체 위치(참조점)를 특징 벡터에 직접 결합함으로써 위치 인식 능력을 강화하였다.

## 🛠️ Methodology

### 전체 시스템 구조
ISDA는 크게 세 가지 블록으로 구성된다: CNN 기반의 Backbone 및 Neck, ISDA Head, 그리고 모델 학습을 감독하는 Bipartite Matching 모듈이다.

### 주요 구성 요소 및 절차

**1. Backbone and Neck**
입력 이미지 $x \in \mathbb{R}^{3 \times H \times W}$로부터 ResNet-50 backbone을 통해 4개의 서로 다른 해상도를 가진 특징 맵 $\{C_i\}_{i=2}^5$를 추출한다. Neck 모듈은 이를 처리하여 $\{P_i\}_{i=2}^6$의 멀티 스케일 특징 맵을 생성하며, 이는 이후 Deformable Transformer의 입력으로 사용된다.

**2. ISDA Head**
ISDA Head는 세부적으로 Encoder, Decoder, Mask Feature Representation(MFR), Mask Head로 나뉜다.
- **Encoder**: 멀티 스케일 특징 맵 $\{P_i\}_{i=3}^6$와 위치 인코딩(Positional Encoding)을 합산하여 입력으로 받으며, Deformable Attention을 통해 픽셀 간 관계를 모델링한다.
- **Decoder**: 학습 가능한 객체 쿼리(Object Queries)를 입력으로 받는다. Cross-attention을 통해 특징 맵에서 객체 특징을 추출하고, Self-attention을 통해 쿼리 간 상호작용을 수행한다. 최종적으로 객체 특징 벡터 $O$와 참조점 $R$의 세트를 출력한다.
- **Mask Feature Representation (MFR)**: 특징 피라미드를 통해 고해상도의 콤팩트한 마스크 특징을 학습한다. $3 \times 3$ Convolution, Group-norm, ReLU, $2 \times$ Bilinear up-sampling을 반복 수행하며, 최종적으로 $1/4$ 스케일의 특징 맵을 생성한다. 이때 $1/32$ 스케일 맵에 정규화된 픽셀 좌표를 추가하여 위치 정보를 강화한다.

**3. Mask Head (MH)**
Mask Head는 객체 특징 벡터 $O \in \mathbb{R}^{256}$, 정규화된 참조점 $R$, 그리고 $MFR \in \mathbb{R}^{256 \times H/4 \times W/4}$를 입력으로 받아 다음과 같은 단계로 마스크를 생성한다.
- 먼저, $O$를 두 개의 Feed-Forward Networks(FFNs)에 통과시켜 분류 점수 $P_c$와 raw 객체 커널 $G_{raw}$를 계산한다.
- $G_{raw}$와 참조점 $R$을 결합(concatenate)하여 위치 인식 커널 $G_{pos}$를 생성한다.
- 최종적으로 $G_{pos}$와 $MFR$을 컨볼루션(Convolution) 연산하여 객체 마스크 $M$을 생성한다.

### 학습 목표 및 손실 함수
모델 학습을 위해 예측된 마스크 세트와 Ground Truth(GT) 세트를 1:1로 매칭하는 Bipartite Matching을 수행한다. 손실 함수는 분류 손실과 마스크 IoU 손실의 합으로 정의된다. 마스크 IoU 손실은 다음과 같이 계산된다.
$$\mathcal{L}_{mask} = 1 - \text{IoU}(m_i, \hat{m}_{\sigma(i)})$$
여기서 $m_i$는 $i$번째 GT 마스크, $\hat{m}_{\sigma(i)}$는 매칭된 예측 마스크를 의미한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MS-COCO val2017 / test-dev2017
- **백본**: ImageNet으로 사전 학습된 ResNet-50
- **최적화**: AdamW 옵티마이저, 초기 학습률 $1.87 \times 10^{-5}$, 배치 사이즈 3
- **지표**: Average Precision (AP), $\text{AP}_{50}$, $\text{AP}_{75}$, $\text{AP}_S$ (소형), $\text{AP}_M$ (중형), $\text{AP}_L$ (대형)

### 정량적 결과
ISDA는 강력한 베이스라인인 Mask R-CNN 대비 AP 기준 2.6 포인트 높은 성능을 보였다. 구체적인 수치는 다음과 같다.

| Method | AP | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{AP}_S$ | $\text{AP}_M$ | $\text{AP}_L$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Mask R-CNN | 36.1 | 58.2 | 38.5 | 20.1 | 38.8 | 46.4 |
| SOLOv2 | 37.4 | 58.4 | 40.1 | 15.4 | 40.2 | 57.4 |
| **ISDA (Ours)** | **38.7** | **62.0** | **41.1** | **17.0** | **41.2** | **55.7** |

### 절제 실험 (Ablation Study)
1. **MFR 해상도**: $1/8, 1/4, 1/2$ 스케일을 비교한 결과, $1/4$ 스케일에서 가장 좋은 AP(36.5)를 기록하였다. 해상도가 높을수록 작은 객체($\text{AP}_S$) 성능은 향상되나 큰 객체($\text{AP}_L$) 성능이 저하되는 트레이드오프가 관찰되었다.
2. **위치 정보의 영향**: MFR에 좌표를 추가(MP)하고 커널에 참조점을 추가(KP)했을 때, 두 가지를 모두 적용한 경우 AP가 4.1% 향상되어 최적의 성능을 보였다. 특히 MP 없이 KP만 적용했을 때는 오히려 성능이 하락했는데, 이는 컨볼루션의 전이 불변성 때문에 MFR의 위치 정보 없이는 커널의 위치 정보만으로 유사 객체를 구분하기 어렵기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 분석
ISDA는 Deformable Transformer를 통해 멀티 스케일 특징을 효율적으로 샘플링하고 합성하므로, 특히 중형 크기의 객체 생성에서 강점을 보인다. 또한, 학습된 위치 인식 커널을 통해 Mask R-CNN에서 빈번히 발생하는 중복 마스크 생성 문제와 경계 영역의 거친 세그멘테이션 문제를 효과적으로 해결하여 더 정교한 마스크를 생성한다.

### 한계 및 향후 과제
정성적 분석 결과, ISDA는 여전히 서로 겹쳐 있는(overlapping) 객체들을 분리하여 세그멘테이션 하는 데 어려움을 겪는 경우가 있다. 이는 향후 연구에서 해결해야 할 과제로 남아 있다.

### 비판적 해석
본 논문은 NMS를 제거하여 학습 효율과 성능을 동시에 잡았다는 점에서 의의가 있다. 하지만 $\text{AP}_S$와 $\text{AP}_L$ 성능이 최신 SOTA 모델들에 비해 최고 수준은 아니라는 점은, Deformable Transformer의 샘플링 방식이 특정 크기의 객체에 편향되어 있을 가능성을 시사한다.

## 📌 TL;DR

ISDA는 Deformable Transformer를 활용하여 NMS 없이 엔드-투-엔드로 학습 가능한 인스턴스 세그멘테이션 프레임워크이다. 위치 인식 커널과 MFR의 좌표 정보를 결합하여 외형이 유사한 객체들을 효과적으로 구분하며, MS-COCO 데이터셋에서 Mask R-CNN 대비 AP 2.6% 향상을 달성하였다. 이 연구는 후처리 과정 없는 단순한 추론 파이프라인을 구축함으로써 향후 실시간 인스턴스 세그멘테이션 연구에 중요한 기반을 제공할 것으로 보인다.