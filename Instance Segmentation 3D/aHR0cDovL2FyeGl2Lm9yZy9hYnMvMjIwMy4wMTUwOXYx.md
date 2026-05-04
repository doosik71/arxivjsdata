# SoftGroup for 3D Instance Segmentation on Point Clouds

Thang Vu, Kookhoi Kim, Tung M. Luu, Xuan Thanh Nguyen, Chang D. Yoo (2022)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(Point Clouds)에서의 인스턴스 세그멘테이션(Instance Segmentation) 문제를 해결하고자 한다. 기존의 최신 방법론들은 주로 세만틱 세그멘테이션(Semantic Segmentation)을 수행한 뒤, 그 결과를 바탕으로 포인트를 그룹화하는 'Bottom-up' 방식을 채택하고 있다.

그러나 기존 방식은 세만틱 세그멘테이션 단계에서 각 포인트에 단 하나의 클래스만 할당하는 'Hard Prediction'(one-hot 예측)을 수행한다. 이러한 결정 방식은 다음과 같은 심각한 문제를 야기한다:
1. **오류 전파(Error Propagation):** 세만틱 예측 단계에서 발생한 작은 오류가 그룹화 단계로 그대로 전달되어, 예측된 인스턴스와 실제 정답(Ground Truth) 간의 겹침(Overlap) 정도가 낮아진다.
2. **거짓 양성(False Positives) 증가:** 잘못 예측된 세만틱 영역으로 인해 존재하지 않는 가짜 인스턴스가 생성되는 결과가 초래된다.

따라서 본 논문의 목표는 세만틱 예측의 불확실성을 수용할 수 있는 'Soft Grouping' 방식과 이를 보완하는 'Top-down Refinement' 단계를 도입하여 3D 인스턴스 세그멘테이션의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 세만틱 예측 시 가장 높은 확률값을 가진 클래스 하나만 선택하는 것이 아니라, 일정 임계값 이상의 점수를 가진 여러 클래스에 포인트가 동시에 속할 수 있도록 허용하는 것이다.

1. **Soft Grouping:** Hard prediction 대신 Soft semantic scores를 사용하여 그룹화를 수행함으로써, 세만틱 예측 오류로 인해 인스턴스가 쪼개지거나 누락되는 문제를 완화한다.
2. **Top-down Refinement:** Soft grouping으로 인해 발생할 수 있는 낮은 정밀도(Precision) 문제를 해결하기 위해, 생성된 인스턴스 제안(Proposal)들을 다시 한번 검토하여 긍정 샘플은 정교화하고 부정 샘플(Background)은 제거하는 단계를 추가한다.
3. **통합 프레임워크:** Bottom-up의 빠른 추론 속도와 Top-down의 정교한 정제 능력을 결합하여 성능과 효율성을 동시에 달성하였다.

## 📎 Related Works

**1. 3D 포인트 클라우드 딥러닝**
- PointNet과 같은 Point-wise 방식과 Sparse Convolution을 사용하는 Voxel-based 방식이 존재한다. 본 논문은 효율적인 연산을 위해 Submanifold Sparse Convolution 기반의 U-Net 구조를 사용한다.

**2. Proposal-based Instance Segmentation (Top-down)**
- Mask R-CNN과 유사하게 영역 제안(Region Proposal)을 먼저 생성하고 그 내부를 세그멘테이션하는 방식이다. 하지만 3D 데이터는 표면 정보만 존재하므로 고품질의 Bounding Box 제안을 생성하는 것이 매우 어렵다는 한계가 있다.

**3. Grouping-based Instance Segmentation (Bottom-up)**
- 포인트별 세만틱 라벨과 중심점 오프셋(Offset)을 예측한 뒤 그룹화하는 방식이다 (예: PointGroup, HAIS). 추론 속도가 빠르지만, 앞서 언급한 세만틱 예측 오류가 그룹화 단계로 전파되는 치명적인 약점이 있다.

본 논문의 SoftGroup은 이 두 가지 접근법의 장점을 결합하여, Soft grouping으로 고품질의 Proposal을 생성하고 Top-down 방식으로 이를 정제한다.

## 🛠️ Methodology

### 전체 시스템 구조
SoftGroup은 크게 두 단계의 파이프라인으로 구성된다: **Bottom-up Grouping $\rightarrow$ Top-down Refinement**.

### 1. Point-wise Prediction Network (Bottom-up 단계)
입력 포인트 클라우드는 Voxel화되어 U-Net 스타일의 백본 네트워크를 통과하며, 두 개의 브랜치를 통해 예측값을 출력한다.
- **Semantic Branch:** 각 포인트 $i$에 대해 모든 클래스에 대한 점수 $S \in \mathbb{R}^{N \times N_{class}}$를 예측한다.
- **Offset Branch:** 각 포인트에서 해당 인스턴스의 기하학적 중심까지의 벡터 $O \in \mathbb{R}^{N \times 3}$를 예측한다.

학습을 위해 세만틱 브랜치는 Cross-Entropy 손실 함수를, 오프셋 브랜치는 $L_1$ 회귀 손실 함수를 사용한다.
$$L_{semantic} = \frac{1}{N} \sum_{i=1}^{N} CE(s_i, s^*_i)$$
$$L_{offset} = \frac{\sum_{i=1}^{N} \mathbb{1}\{p_i\} \lVert o_i - o^*_i \rVert_1}{\sum_{i=1}^{N} \mathbb{1}\{p_i\}}$$

### 2. Soft Grouping
전통적인 방식과 달리, $\text{argmax}$를 사용하지 않고 임계값 $\tau$를 도입한다.
- **과정:** 각 클래스별로 점수가 $\tau$보다 높은 포인트들의 서브셋을 추출한다. 한 포인트가 여러 클래스의 임계값을 넘으면 여러 클래스 서브셋에 동시에 포함될 수 있다.
- **그룹화:** 각 서브셋 내에서 예측된 오프셋을 이용해 포인트를 중심점으로 이동시킨 후, 기하학적 거리 $b$ 이내의 포인트들을 연결하여 인스턴스 Proposal을 생성한다.
- **임계값 $\tau$의 영향:** $\tau$가 낮아지면 Recall(재현율)은 증가하지만 Precision(정밀도)이 낮아진다. 본 논문에서는 정밀도와 재현율의 균형을 맞추기 위해 $\tau = 0.2$로 설정하였다.

### 3. Top-down Refinement
Soft grouping으로 생성된 Proposal들은 배경(Background)이나 잘못된 클래스가 섞여 있을 가능성이 높다. 이를 위해 Tiny U-Net 기반의 정제 단계를 거친다.
- **Classification Branch:** 인스턴스 전체의 특징을 Global Average Pooling으로 집계한 후, 이 인스턴스가 실제 어떤 클래스인지 또는 배경($N_{class}+1$)인지 분류한다.
- **Segmentation Branch:** Proposal 내부에서 포인트별로 이진 마스크 $m_k$를 예측하여 불필요한 배경 포인트를 제거한다.
- **Mask Scoring Branch:** 예측된 마스크와 실제 정답 간의 IoU를 예측하여 신뢰도를 측정한다.

**학습 타겟:** GT 인스턴스와의 $\text{IoU} > 50\%$인 Proposal을 Positive sample로 간주하고 학습시킨다. 최종 손실 함수는 다음과 같이 모든 단계의 합으로 정의된다.
$$L = L_{semantic} + L_{offset} + L_{class} + L_{mask} + L_{mask\_score}$$

## 📊 Results

### 실험 설정
- **데이터셋:** ScanNet v2 (18개 클래스), S3DIS (13개 클래스)
- **지표:** $AP_{50}$, $AP_{25}$, $AP$ (Average Precision) 및 S3DIS의 경우 mCov, mWCov, mPrec, mRec 사용.
- **구현 세부사항:** PyTorch 사용, Adam Optimizer, $\tau=0.2$, Voxel size $0.02\text{m}$, Grouping bandwidth $0.04\text{m}$.

### 주요 결과
1. **정량적 성능:** 
   - **ScanNet v2:** $AP_{50}$ 기준 76.1%를 달성하여 기존 최강 방법론 대비 **+6.2%**라는 압도적인 성능 향상을 보였다.
   - **S3DIS Area 5:** $AP_{50}$ 기준 66.1%를 기록하며 2위 방법론 대비 **+6.8%** 향상되었다.
2. **추론 속도:** ScanNet v2 씬 하나당 **345ms**가 소요되어, 성능 향상에도 불구하고 실시간성에 가까운 매우 빠른 속도를 유지하였다.
3. **객체 탐지 성능:** 마스크에서 Bounding Box를 추출하여 측정한 결과, Box $AP_{50}$에서도 기존 방법들보다 월등히 높은 성능을 보였다.

## 🧠 Insights & Discussion

**강점 및 분석**
- **Soft Grouping의 효과:** 실험 결과, Hard prediction을 사용했을 때보다 $\tau=0.2$를 적용한 Soft grouping이 Recall을 크게 높였음을 확인하였다. 이는 세만틱 예측이 완벽하지 않더라도 인스턴스 후보군을 충분히 확보할 수 있음을 의미한다.
- **Refinement의 필수성:** Soft grouping만으로는 Precision이 낮아지는데, Top-down refinement 단계가 이를 효과적으로 억제하여 최종적인 AP를 끌어올렸다. 특히 Classification branch가 세만틱 예측의 다수결 투표보다 더 신뢰할 수 있는 클래스 정보를 제공함을 입증하였다.

**한계 및 논의**
- 본 논문은 $\tau$ 값을 하이퍼파라미터로 설정하여 최적의 값을 찾았으나, 데이터셋의 특성에 따라 최적의 $\tau$가 달라질 수 있다는 점이 잠재적 한계로 보인다.
- 하지만 전체적인 파이프라인이 단순하고 효율적이어서 다양한 3D 환경에 적용 가능성이 높다.

## 📌 TL;DR

SoftGroup은 3D 인스턴스 세그멘테이션에서 발생하는 **'세만틱 예측 오류 $\rightarrow$ 그룹화 오류'**라는 전파 문제를 해결하기 위해, **Soft semantic scores 기반의 그룹화**와 **Top-down 정제 단계**를 제안한 연구이다. 이를 통해 ScanNet v2와 S3DIS 데이터셋에서 SOTA 성능을 달성하였으며, 빠른 추론 속도까지 확보하였다. 이 연구는 향후 3D 장면 이해(Scene Understanding) 및 로봇 내비게이션 분야에서 고정밀 인스턴스 분할을 구현하는 데 중요한 기초가 될 것으로 보인다.