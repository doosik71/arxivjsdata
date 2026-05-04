# Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation

Yingda Xia, Dong Yang, Zhiding Yu, Fengze Liu, Jinzheng Cai, Lequan Yu, Zhuotun Zhu, Daguang Xu, Alan Yuille, Holger Roth (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 분야에서 정교하게 레이블링 된 데이터(Well-annotated data)를 확보하는 데 드는 막대한 비용과 전문가의 시간 소모 문제이다. 딥러닝 기반의 접근 방식은 높은 성능을 보이지만, 대규모의 학습 데이터가 필수적이며, 의료 데이터의 특성상 숙련된 방사선 전문의의 수동 작업이 필요하여 데이터 획득이 매우 어렵다.

반면, 레이블이 없는(Unlabeled) 데이터는 상대적으로 획득하기 쉽다. 따라서 본 연구는 레이블이 없는 데이터를 효율적으로 활용하여 모델의 성능을 높이는 것을 목표로 한다. 구체적으로는 다음 두 가지 시나리오를 동시에 해결할 수 있는 통합 프레임워크를 제안한다:
1. **준지도 학습(Semi-Supervised Learning, SSL):** 레이블이 있는 소량의 데이터와 대량의 레이블 없는 데이터를 함께 사용하여 성능을 향상시키는 것.
2. **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA):** 소스 도메인의 레이블 데이터와 타겟 도메인의 레이블 없는 데이터를 활용하여, 도메인 간의 분포 차이(Domain Shift)를 극복하고 타겟 도메인에서 높은 성능을 내는 것.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Uncertainty-aware Multi-view Co-training (UMCT)** 프레임워크의 제안이다. 이 프레임워크의 중심 아이디어는 다음과 같다.

1. **데이터 수준의 뷰 차이 생성:** 3D 의료 영상 데이터를 회전(Rotation) 및 치환(Permutation)하여 여러 개의 뷰(View)를 생성하고, 각 뷰마다 독립적인 3D 네트워크를 학습시킨다.
2. **특징 수준의 뷰 차이 강제:** 모델들이 서로 동일한 특징만 학습하여 붕괴(Collapse)되는 것을 방지하기 위해, 대칭적인 $n \times n \times n$ 커널 대신 비대칭적인 $n \times n \times 1$ 커널을 사용하고 2D 사전 학습 가중치로 초기화한다.
3. **불확실성 기반의 의사 레이블링 (ULF):** 모든 뷰의 예측 결과가 항상 정확하지 않으므로, 베이지안 딥러닝(Bayesian Deep Learning)의 에피스테믹 불확실성(Epistemic Uncertainty)을 측정하여 신뢰도가 높은 뷰의 예측값에 더 큰 가중치를 주는 **Uncertainty-weighted Label Fusion (ULF)** 모듈을 도입한다.
4. **통합 프레임워크:** 동일한 구조를 SSL뿐만 아니라 UDA, 그리고 소스 데이터가 없는 극한의 UDA 상황(UDA without source data)에도 적용 가능하다는 것을 입증하였다.

## 📎 Related Works

논문은 다음과 같은 관련 연구들을 검토하고 차별점을 제시한다.

- **준지도 학습 및 의료 영상 분석:** 기존의 Self-training(Teacher-Student 모델)이나 GAN 기반 방식들이 존재하지만, 본 논문은 여러 뷰의 일관성을 강제하는 Co-training 방식을 3D 데이터로 확장하여 접근한다.
- **불확실성 추정(Uncertainty Estimation):** 딥러닝에서 불확실성을 측정하기 위한 베이지안 접근법(특히 Dropout을 통한 근사)이 제안되었으며, 본 논문은 이를 의사 레이블(Pseudo-label)의 가중치를 결정하는 핵심 기제로 사용한다.
- **2D/3D 하이브리드 네트워크:** 2D의 사전 학습 이점과 3D의 공간 정보 활용 능력을 결합하려는 시도가 있었으며, 본 논문은 비대칭 3D 커널을 통해 이를 구현하고 뷰 간의 다양성을 확보한다.
- **비지도 도메인 적응(UDA):** 적대적 학습(Adversarial training)이나 Self-training을 통한 도메인 정렬 방식이 주로 사용되었으나, UMCT는 다중 뷰의 일관성을 통해 보다 안정적인 적응 성능을 보인다.

## 🛠️ Methodology

### 전체 시스템 구조
UMCT 프레임워크는 입력 데이터 $X$에 대해 $N$개의 서로 다른 변환 $T_i$ (회전 및 치환)를 적용하여 $N$개의 뷰를 생성한다. 각 뷰는 독립적인 3D 네트워크 $f_i$를 통해 처리되며, 최종적으로 모든 뷰의 예측 결과가 일치하도록 유도하는 Co-training 과정을 거친다.

### 학습 목표 및 손실 함수
전체 손실 함수는 레이블이 있는 데이터에 대한 지도 학습 손실($L^{sup}$)과 레이블이 없는 데이터에 대한 Co-training 손실($L^{cot}$)의 합으로 정의된다:
$$\sum_{(X,Y) \in S} L^{sup}(X,Y) + \lambda_{cot} \sum_{X \in U} L^{cot}(X, \hat{Y}_i)$$

여기서 지도 학습 손실 $L^{sup}$은 각 뷰의 예측값 $p_i(X)$와 실제 정답 $Y$ 사이의 오차를 측정한다:
$$L^{sup}(X,Y) = \sum_{i=1}^{N} L(p_i(X), Y)$$

Co-training 손실 $L^{cot}$은 각 뷰의 예측값 $p_i(X)$와 ULF 모듈을 통해 생성된 의사 레이블 $\hat{Y}_i$ 사이의 오차를 측정한다:
$$L^{cot}(X, \hat{Y}_i) = \sum_{i=1}^{N} L(p_i(X), \hat{Y}_i)$$

### Uncertainty-weighted Label Fusion (ULF)
의사 레이블 $\hat{Y}_i$는 $i$번째 뷰를 제외한 나머지 모든 뷰의 예측값들을 가중 평균하여 생성한다. 이때 가중치는 각 뷰의 신뢰도 $c(p_j(X))$로 결정된다:
$$\hat{Y}_i = \frac{\sum_{j \neq i} c(p_j(X)) p_j(X)}{\sum_{j \neq i} c(p_j(X))}$$

신뢰도 $c$는 에피스테믹 불확실성 $U_e$의 역수로 정의된다 ($c = 1/U_e$). 에피스테믹 불확실성은 동일한 입력에 대해 Dropout을 적용하여 $K$번 샘플링한 예측값들의 분산으로 추정한다:
$$U_e(y) \approx \frac{1}{K} \sum_{k=1}^{K} \hat{y}_k^2 - \left( \frac{1}{K} \sum_{k=1}^{K} \hat{y}_k \right)^2$$

### 비대칭 3D 커널 설계
뷰 간의 차이를 극대화하기 위해, 일반적인 $n \times n \times n$ 3D 커널 대신 $n \times n \times 1$ 형태의 비대칭 커널을 사용한다. 이는 모델이 단순한 회전/치환 관계만을 학습하는 것을 방지하고, 각 뷰에서 서로 보완적인 특징을 추출하도록 강제한다.

## 📊 Results

### 실험 설정
- **데이터셋:** NIH 췌장 분할 데이터셋(82개 볼륨), 다기관 장기 분할 데이터셋(90개 케이스), MSD(Medical Segmentation Decathlon) 췌장 및 간 데이터셋.
- **지표:** Dice-Sørensen Coefficient (DSC).
- **기준선:** Supervised 3D ResNet-18, DMPCT, DCT, TCSE 등.

### 주요 결과
1. **준지도 학습 (NIH 췌장 데이터):**
   - 레이블 데이터가 10%일 때, 제안 방법(6-view)은 77.87%의 DSC를 기록하여 지도 학습 베이스라인(66.75%) 대비 큰 폭의 성능 향상을 보였다.
   - 특히 20% 레이블 데이터를 사용한 UMCT의 성능(80.35%)이 60% 레이블 데이터를 사용한 순수 지도 학습 성능(78.95%)보다 높게 나타나, 레이블링 비용을 약 70% 절감할 수 있음을 입증했다.

2. **다기관 장기 분할:**
   - 8개의 복부 장기를 대상으로 한 실험에서, 거의 모든 장기에 대해 지도 학습 베이스라인 대비 일관된 성능 향상을 보였다.

3. **비지도 도메인 적응 (UDA):**
   - 소스 데이터(Multi-organ)에서 타겟 데이터(MSD 췌장/간)로 적응시켰을 때, UMCT는 기존의 Adversarial training이나 Self-training 방식보다 높은 DSC를 기록했다.
   - **UMCT-DA (소스 데이터 없음):** 소스 데이터 없이 사전 학습된 모델과 타겟 도메인의 레이블 없는 데이터만으로 학습했을 때도, 표준 UDA 설정과 유사한 수준의 성능을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **비대칭 설계의 중요성:** V-Net과 같은 대칭적 커널 모델보다 비대칭 커널을 사용한 ResNet-18 기반 UMCT가 더 높은 성능을 보였다. 이는 Co-training의 성공 조건인 '뷰 간의 독립성/차이'가 적절히 확보되었음을 의미한다.
- **불확실성 가중치의 효과:** ULF 모듈을 적용했을 때, 단순히 예측값을 평균 내는 것보다 성능이 향상되었다. 이는 불확실성이 높은 (즉, 신뢰도가 낮은) 예측값을 효과적으로 배제하여 의사 레이블의 품질을 높였기 때문이다.

### 한계 및 논의
- **뷰 설정의 고정성:** 현재는 axial, coronal, sagittal 등 미리 정의된 뷰만을 사용한다. 저자들은 향후 더 많은 뷰나 랜덤한 뷰를 도입함으로써 강건성을 높일 수 있을 것이라고 언급한다.
- **도메인 시프트의 범위:** 본 실험에서의 도메인 시프트는 주로 서로 다른 병원의 CT 스캔이나 병리적 상태의 차이에 국한되었다. CT에서 MRI로의 전환과 같은 더 큰 규모의 모달리티 변화(Cross-modality)에 대해서는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할을 위한 **불확실성 인지 다중 뷰 코트레이닝(UMCT)** 프레임워크를 제안한다. 3D 영상을 여러 뷰로 변환하고 비대칭 커널 네트워크를 통해 각 뷰의 다양성을 확보하며, 베이지안 불확실성 추정을 통해 신뢰할 수 있는 의사 레이블만을 사용하여 학습한다. 이 방법은 준지도 학습과 비지도 도메인 적응 모두에서 기존 SOTA 성능을 상회하며, 특히 소스 데이터가 없는 극한의 UDA 상황에서도 효과적임을 증명하여 실제 임상 적용 가능성을 높였다.