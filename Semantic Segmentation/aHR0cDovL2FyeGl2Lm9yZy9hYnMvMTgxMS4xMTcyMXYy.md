# CCNet: Criss-Cross Attention for Semantic Segmentation

Zilong Huang, Xinggang Wang,Member, IEEE, Yunchao Wei, Lichao Huang, Humphrey Shi,Member, IEEE, Wenyu Liu,Senior Member, IEEE, and Thomas S. Huang,Life Fellow, IEEE

## 🧩 Problem to Solve

시맨틱 분할(Semantic Segmentation) 및 객체 탐지와 같은 시각 이해 문제에서 **전체 이미지 문맥 정보(full-image contextual information)**는 매우 중요합니다. 기존의 FCN(Fully Convolutional Network) 기반 방법은 수용장(receptive field)이 고정되어 있어 단거리 문맥 정보만 제공하는 한계가 있었습니다. 비지역(Non-local) 네트워크와 같은 GNN(Graph Neural Network) 기반 방법은 전체 이미지 문맥 정보를 포착하지만, 각 픽셀 쌍의 관계를 측정하기 위해 거대한 어텐션 맵을 생성해야 하므로 시간 및 공간 복잡도가 $O(N^2)$로 매우 높습니다. 특히 시맨틱 분할과 같은 밀집 예측 작업은 고해상도 특징 맵을 요구하기 때문에 이러한 높은 복잡도는 GPU 메모리 사용량과 계산 효율성 측면에서 큰 제약을 야기합니다.

이 논문은 다음과 같은 문제를 해결하고자 합니다: 기존 방법의 **높은 계산 및 메모리 복잡도 없이** 효과적이고 효율적인 방식으로 **전체 이미지 문맥 정보를 획득**하여 시맨틱 분할 성능을 향상시키는 것.

## ✨ Key Contributions

- **새로운 Criss-Cross Attention (CCA) 모듈 제안:** 각 픽셀의 가로 및 세로 경로에 있는 모든 픽셀의 문맥 정보를 효율적으로 수집하여, 기존 비지역 블록(Non-local block) 대비 GPU 메모리 사용량을 11배, FLOPs(부동소수점 연산)를 약 85% 절감하면서도 전체 이미지 의존성(full-image dependencies)을 포착합니다.
- **재귀적 Criss-Cross Attention (RCCA) 모듈 도입:** 단일 CCA 모듈로는 모든 픽셀 간의 연결을 제공할 수 없으므로, CCA 모듈에 재귀 연산을 적용하여 각 픽셀이 최종적으로 전체 이미지의 문맥 정보를 포착할 수 있도록 합니다.
- **Category Consistent Loss (CCL) 제안:** CCA 모듈이 더욱 판별력 있는(discriminative) 특징을 생성하도록 돕기 위해, 동일 카테고리 픽셀의 특징은 가깝게, 다른 카테고리 픽셀의 특징은 멀리 떨어지도록 강제하는 손실 함수를 도입했습니다.
- **최고 수준의 성능 달성:** 제안된 CCNet은 Cityscapes, ADE20K, LIP, CamVid 등 주요 시맨틱 분할 벤치마크와 COCO 인스턴스 분할 벤치마크에서 기존 최고 수준의 성능을 뛰어넘는 결과를 달성했습니다.
- **3D CCA 모듈로 확장:** CCA 모듈을 2D에서 3D 네트워크로 확장하여 장거리 시간-공간 문맥 정보 포착 능력을 입증했습니다.

## 📎 Related Works

- **시맨틱 분할 (Semantic Segmentation):** FCN [8]을 시작으로 DeepLab [10, 28] (atrous spatial pyramid pooling (ASPP) [10, 12]), PSPNet [11] (pyramid pooling module), UNet [27] (인코더-디코더), RefineNet [31], DFN [32] 등 다양한 FCN 기반 방법들이 발전해왔습니다.
- **문맥 정보 통합 (Contextual Information Aggregation):** ASPP [10], PSPNet [11], PSANet [16], Non-local Networks [9], OCNet [56], DANet [57] 등이 있으며, 특히 비지역 네트워크는 자체 어텐션 메커니즘 [17, 18]을 활용하여 전체 이미지 문맥 정보를 포착합니다.
- **그래프 신경망 (Graph Neural Networks, GNN):** CRF [25, 40, 55], MRF [35]와 같은 전통적인 그래프 모델과 더불어, CNN에서 영감을 받아 그래프 구조를 적용한 스펙트럼 기반 [62-65] 및 공간 기반 [9, 66-68] 접근법들이 연구되었습니다. CCNet은 공간 기반 GNN에 속합니다.
- **CCNet과의 차이점:** GCN [59]은 중심 픽셀만 전체 문맥을 인식하는 반면, Non-local Network [9]와 CCNet은 모든 픽셀이 전체 문맥을 인식합니다. Non-local Network는 $O(N^2)$ 복잡도를 가지지만, CCNet은 재귀적 criss-cross 어텐션을 통해 더 효율적인 $O(N\sqrt{N})$ 복잡도로 밀집 문맥 정보를 얻습니다.

## 🛠️ Methodology

CCNet은 심층 합성곱 신경망(DCNN)으로 특징 맵 $X$를 추출한 뒤, 이 특징 맵을 재귀적 Criss-Cross Attention (RCCA) 모듈을 통해 풍부한 문맥 정보로 보강하고, 최종적으로 분할 결과를 예측하는 구조를 가집니다.

1. **초기 특징 추출:** 입력 이미지는 ResNet-101 (ImageNet 사전 학습)과 같은 DCNN을 통과하여 특징 맵 $X \in \mathbb{R}^{C \times W \times H}$를 생성합니다. 상세 유지를 위해 마지막 두 다운샘플링 연산은 제거되고, 확장 합성곱(dilated convolution)이 사용되어 출력 보폭(output stride)을 8로 유지합니다. $X$는 1x1 합성곱을 통해 채널이 축소된 특징 맵 $H$로 변환됩니다.

2. **Criss-Cross Attention (CCA) 모듈:**

   - **입력:** 특징 맵 $H \in \mathbb{R}^{C \times W \times H}$.
   - **쿼리(Q), 키(K), 값(V) 생성:**
     - $H$에 두 개의 1x1 합성곱을 적용하여 쿼리 $Q$와 키 $K$를 생성합니다. $Q, K \in \mathbb{R}^{C' \times W \times H}$ ($C' < C$로 차원 축소).
     - $H$에 또 다른 1x1 합성곱을 적용하여 값 $V \in \mathbb{R}^{C \times W \times H}$를 생성합니다.
   - **어피니티(Affinity) 연산:** 각 위치 $u$에서 쿼리 $Q_u$를 추출하고, $K$에서 $u$와 같은 행 또는 열에 있는 특징 벡터들($\Omega_u$)을 추출합니다. 이들 간의 상관도 $d_{i,u} = Q_u \Omega_{i,u}^{\text{T}}$를 계산하여 어텐션 맵 $A \in \mathbb{R}^{(H+W-1) \times (W \times H)}$를 생성합니다. $D$에 소프트맥스(softmax)를 적용하여 정규화합니다.
   - **어그리게이션(Aggregation) 연산:** 각 위치 $u$에서 값 $V_u$를 추출하고, $V$에서 $u$와 같은 행 또는 열에 있는 특징 벡터들($\Phi_u$)을 추출합니다. 어텐션 맵 $A$를 기반으로 문맥 정보를 집계합니다:
     $$ H'_u = \sum_{i=0}^{H+W-1} A*{i,u} \Phi*{i,u} + H_u $$
        여기서 $H'_u$는 최종 출력 특징 맵 $H' \in \mathbb{R}^{C \times W \times H}$의 위치 $u$에 대한 특징 벡터입니다.

3. **재귀적 Criss-Cross Attention (RCCA) 모듈:**

   - CCA 모듈을 직렬로 $R$번 반복합니다 (기본 설정 $R=2$).
   - 첫 번째 반복에서는 $H$를 입력으로 받아 $H'$를 생성합니다.
   - 두 번째 반복에서는 $H'$를 입력으로 받아 $H''$를 생성합니다.
   - $R=2$일 때, 모든 픽셀이 전체 이미지의 문맥 정보를 포착할 수 있게 됩니다 (정보가 십자 경로를 통해 대각선으로 전파).
   - 반복되는 CCA 모듈은 파라미터를 공유하여 모델의 경량화를 유지합니다.
   - 이러한 재귀적 구조는 비지역 블록의 $O(N^2)$ 대신 $O(N\sqrt{N})$의 계산 복잡도를 가집니다.

4. **Category Consistent Loss (CCL):**

   - 최종 손실 $\mathcal{L}$은 시맨틱 분할 손실 $\mathcal{L}_{\text{seg}}$ (교차 엔트로피 손실)과 CCL의 가중치 합으로 구성됩니다:
     $$ \mathcal{L} = \mathcal{L}_{\text{seg}} + \alpha \mathcal{L}_{\text{var}} + \beta \mathcal{L}_{\text{dis}} + \gamma \mathcal{L}_{\text{reg}} $$
   - $\mathcal{L}_{\text{var}}$: 동일 카테고리 픽셀 특징이 해당 카테고리 평균($\mu_c$)에 가깝도록 유도합니다. 견고한 최적화를 위해 구간별 거리 함수($\phi_{\text{var}}$)를 사용합니다.
     $$ \mathcal{L}_{\text{var}} = \frac{1}{|C|} \sum_{c \in C} \frac{1}{N*c} \sum*{i=1}^{N*c} \phi*{\text{var}}(h*i, \mu_c) $$
        $$ \phi*{\text{var}} = \begin{cases} \| \mu_c - h_i \| - \delta_d + (\delta_d - \delta_v)^2, & \| \mu_c - h_i \| > \delta_d \\ (\| \mu_c - h_i \| - \delta_v)^2, & \delta_v < \| \mu_c - h_i \| \le \delta_d \\ 0, & \| \mu_c - h_i \| \le \delta_v \end{cases} $$
   - $\mathcal{L}_{\text{dis}}$: 다른 카테고리의 평균 특징들이 서로 멀리 떨어지도록 유도합니다.
     $$ \mathcal{L}_{\text{dis}} = \frac{1}{|C|(|C|-1)} \sum_{c*a \in C} \sum*{c*b \in C, c_a \ne c_b} \phi*{\text{dis}}(\mu*{c_a}, \mu*{c*b}) $$
        $$ \phi*{\text{dis}} = \begin{cases} (2\delta*d - \| \mu*{c*a} - \mu*{c*b} \|)^2, & \| \mu*{c*a} - \mu*{c*b} \| \le 2\delta_d \\ 0, & \| \mu*{c*a} - \mu*{c_b} \| > 2\delta_d \end{cases} $$
   - $\mathcal{L}_{\text{reg}}$: 모든 카테고리 평균 특징이 원점 방향으로 끌려오도록 유도합니다.
     $$ \mathcal{L}_{\text{reg}} = \frac{1}{|C|} \sum_{c \in C} \| \mu_c \| $$
   - 실험에서는 $\delta_v=0.5$, $\delta_d=1.5$, $\alpha=\beta=1$, $\gamma=0.001$로 설정되었습니다.

5. **3D Criss-Cross Attention:** 2D CCA를 3D (예: 비디오)로 확장하여 시간 차원의 문맥 정보를 추가적으로 포착합니다.

## 📊 Results

다양한 대규모 데이터셋(Cityscapes, ADE20K, LIP, COCO, CamVid)에서 광범위한 실험을 통해 CCNet의 효과를 검증했습니다.

- **Cityscapes (시맨틱 분할):**
  - ResNet-101 백본으로 Cityscapes 테스트 세트에서 **81.9% mIoU**를 달성하여 기존 SOTA(PSANet 80.1%)를 크게 능가했습니다.
  - RCCA 루프 수에 대한 실험 결과, $R=1$일 때 2.9% mIoU 증가, $R=2$일 때 추가 1.8% 증가 (총 79.8% mIoU)를 보였습니다. $R=3$은 0.4%만 추가 개선되어, 성능과 자원 사용의 균형을 위해 $R=2$가 최적임을 입증했습니다.
  - 비지역(Non-local) 블록과 비교했을 때, RCCA($R=2$)는 GPU 메모리를 11배 적게 사용하고 FLOPs를 약 85% 절감하면서도 더 나은 성능(Non-local 77.3% vs RCCA 78.5% mIoU)을 보였습니다.
- **ADE20K (장면 구문 분석):**
  - ADE20K 검증 세트에서 **45.76% mIoU**를 달성하여 기존 SOTA(EncNet 44.65%)를 1.1% 이상 능가했습니다.
- **LIP (인체 구문 분석):**
  - LIP 검증 세트에서 **55.47% mIoU**를 달성하여 기존 SOTA(CE2P 53.10%)를 2.3% 이상 능가했습니다.
- **COCO (인스턴스 분할):**
  - Mask R-CNN 백본에 RCCA 모듈을 추가하여 성능을 향상시켰습니다 (예: ResNet-101 백본에서 Mask AP가 36.2%에서 37.3%로 상승). 이는 비지역 블록을 추가한 경우(37.1%)보다도 우수합니다.
- **CamVid (비디오 시맨틱 분할):**
  - 3D-RCCA를 사용하여 CamVid 테스트 세트에서 **79.1% mIoU**를 달성, 모든 기존 방법들을 큰 차이로 앞섰습니다.
- **Category Consistent Loss (CCL) 효과:**
  - CCL을 적용하면 ResNet-101 및 ResNet-50 백본 모두에서 약 **0.7%의 mIoU 개선**이 안정적으로 나타났습니다.
  - 제안된 구간별 거리 함수($\phi_{\text{var}}$)는 기존의 2차 함수보다 학습 성공률(9/10 vs 6/10)이 높고 약간 더 나은 성능을 보여, 최적화의 안정성을 개선함을 입증했습니다.

## 🧠 Insights & Discussion

- **효율적인 전체 이미지 문맥 포착:** CCNet은 비지역 네트워크의 $O(N^2)$ 복잡도를 $O(N\sqrt{N})$으로 대폭 줄이면서도 전체 이미지 문맥 정보를 효과적으로 포착합니다. 이는 특히 고해상도 특징 맵이 필요한 밀집 예측 작업에 매우 중요합니다.
- **재귀 연산의 중요성:** 단일 Criss-Cross Attention만으로는 완전한 전체 이미지 문맥 정보를 포착할 수 없으나, 재귀 연산(특히 $R=2$)을 통해 부족한 연결을 보완하고 더 조밀하고 풍부한 문맥 정보를 수집할 수 있음을 입증했습니다. 이는 후속 단계의 어텐션 맵 학습에 기여하여 성능 향상을 이끌어냅니다.
- **판별력 있는 특징 학습:** Category Consistent Loss는 집계된 특징들의 과도한 평활화(over-smoothing) 문제를 해결하고, 동일 카테고리 픽셀 간의 응집력을 높여 더욱 판별력 있는 특징 표현을 가능하게 합니다. 특히, 제안된 구간별 거리 함수는 최적화 과정을 더욱 안정적으로 만듭니다.
- **광범위한 적용 가능성:** CCNet은 시맨틱 분할뿐만 아니라 인스턴스 분할, 인체 구문 분석, 비디오 분할 등 다양한 밀집 예측 작업에서 일관된 성능 향상을 보여, 제안된 모듈의 일반화 능력을 입증했습니다. 이는 Criss-Cross Attention이 다양한 시각 인식 문제의 핵심적인 문맥 모델링 구성 요소로 활용될 수 있음을 시사합니다.

## 📌 TL;DR

- **문제:** 시맨틱 분할에 필수적인 전체 이미지 문맥 정보 포착은 기존 방법(비지역 네트워크)의 높은 계산 복잡도($O(N^2)$)로 인해 비효율적이었습니다.
- **제안 방법:** CCNet은 효율적인 **재귀적 Criss-Cross Attention (RCCA)** 모듈을 제안합니다. 이 모듈은 각 픽셀의 가로 및 세로 경로를 따라 문맥 정보를 수집하며, 재귀 연산($R=2$)을 통해 $O(N\sqrt{N})$의 복잡도로 전체 이미지 의존성을 포착합니다. 또한, 특징의 판별력을 높이기 위해 **Category Consistent Loss (CCL)**를 도입했습니다.
- **주요 결과:** CCNet은 Cityscapes, ADE20K, LIP, CamVid, COCO 등 다양한 벤치마크에서 기존 SOTA를 능가하는 성능을 달성했으며, 비지역 네트워크 대비 GPU 메모리 사용량을 11배, FLOPs를 85% 절감하는 등 높은 효율성을 입증했습니다.
