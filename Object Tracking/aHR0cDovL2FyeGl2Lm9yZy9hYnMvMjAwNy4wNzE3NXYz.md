# Unsupervised Spatio-temporal Latent Feature Clustering for Multiple-object Tracking and Segmentation

Abubakar Siddique, Reza Jalil Mozhdehi, Henry Medeiros (2021)

## 🧩 Problem to Solve

본 논문은 비디오 시퀀스 내의 여러 움직이는 객체들에 대해 시간적으로 일관된 식별자(Temporal Identifier)를 부여하는 Multiple Object Tracking and Segmentation (MOTS) 문제를 해결하고자 한다. 

MOTS는 객체의 탐지, 세그멘테이션, 그리고 추적을 동시에 수행해야 하는 복잡한 작업이다. 기존의 최신 MOTS 방법론들은 대부분 지도 학습(Supervised Learning)에 의존하여 판별적인 임베딩을 생성하고, 정교한 타겟 행동 모델을 기반으로 특징을 연관시키는 방식을 사용한다. 그러나 이러한 방식은 대량의 레이블링된 데이터가 필요하며, 모델의 복잡도가 높다는 단점이 있다.

또한, 비디오 내의 객체들은 외형의 변화, 가려짐(Occlusion), 빠른 움직임 등으로 인해 데이터 분포가 계속 변화한다. 단순히 위치 정보, 형태 정보, 또는 외형 정보 중 하나만을 사용하여 서브스페이스 클러스터링(Subspace Clustering)을 수행할 경우, 만족스러운 추적 성능을 얻기 어렵다. 따라서 본 연구의 목표는 위치와 형태 정보를 결합한 시공간적 잠재 표현(Spatio-temporal Latent Representation)을 학습하여, 비지도 학습 방식으로 객체들을 시간적으로 일관되게 클러스터링하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MOTS 문제를 시공간적 클러스터링 문제로 재정의하는 것이다. 이를 위해 다음과 같은 기여를 제시한다.

1.  **Deep Heterogeneous Autoencoder (DHAE) 제안**: 서로 다른 성격의 입력(Bounding Box의 위치 정보와 Segmentation Mask의 형태 정보)을 동시에 처리하여 판별적인 잠재 특징(Latent Feature)을 생성하는 비지도 학습 구조를 제안한다.
2.  **Task-dependent Uncertainty 기반의 손실 함수**: 서로 다른 특성을 가진 입력 데이터들의 기여도를 동적으로 조절하기 위해, 각 작업의 불확실성(Uncertainty)을 가중치로 사용하는 멀티태스크 학습 전략을 도입하였다.
3.  **제약 기반 클러스터링(Constrained Clustering) 적용**: 시공간적 제약 조건(Must-link, Cannot-link)을 그래프 형태로 구성하고, 이를 Constrained K-means 알고리즘에 적용함으로써 데이터 연관성의 강건함을 높였다.
4.  **광범위한 검증**: 합성 데이터셋(MNIST-MOT, Sprites-MOT)과 실제 데이터셋(KITTI MOTS, MOTSChallenge)을 통해 제안 방법론이 기존의 SOTA 방법들보다 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 서브스페이스 클러스터링(Subspace Clustering)은 데이터를 저차원 부분 공간으로 매핑하여 데이터 간의 관계를 추론하는 비지도 학습 기법이다. 초기에는 행렬 분해(Factorization)나 커널(Kernel) 기반 방식이 사용되었으며, 최근에는 CNN이나 GAN, 그리고 Self-expressive layer를 결합한 딥러닝 기반 방식(예: DSC-Net)이 연구되었다.

그러나 기존 연구들은 다음과 같은 한계가 있다.
- **정적 데이터 중심**: 대부분의 딥러닝 기반 클러스터링은 정적인 데이터에서 판별적인 특징을 찾는 데 집중하며, 비디오와 같은 순차적 데이터(Sequential Data)의 특성을 충분히 반영하지 못한다.
- **공간 정보의 부재**: OSC(Ordered Subspace Clustering)와 같은 순차 데이터 클러스터링 기법이 존재하지만, 이는 주로 비디오 프레임 전체를 클러스터링하는 데 초점을 맞추었을 뿐, 프레임 내의 개별 객체를 구분하기 위한 공간적 측면을 고려하지 않았다.

본 논문은 이러한 한계를 극복하기 위해 객체의 외형뿐만 아니라 위치 정보를 결합한 잠재 공간을 학습하고, 시공간적 제약 조건을 통해 순차 데이터의 특성을 명시적으로 모델링한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인
전체 시스템은 **Multi-Task Feature Extractor (MTFE) $\rightarrow$ Deep Heterogeneous Autoencoder (DHAE) $\rightarrow$ Constraints Graph Construction $\rightarrow$ Constrained K-means** 순으로 구성된다.

### 2. Multi-Task Feature Extractor (MTFE)
MTFE는 각 프레임에서 객체의 세그멘테이션 마스크 $x_m \in \mathbb{R}^{M \times M \times D}$와 바운딩 박스 $x_b \in \mathbb{R}^N$ (중심점 좌표 및 크기, $N=4$), 그리고 탐지 신뢰도 $c$를 추출한다. 이 모듈은 기존의 Mask R-CNN과 같은 사전 학습된 네트워크를 사용할 수 있다.

### 3. Deep Heterogeneous Autoencoder (DHAE)
DHAE는 서로 다른 차원의 입력을 통합하여 공통의 잠재 특징 $Z$를 생성하는 구조이다.
- **Encoder**: 바운딩 박스는 Fully Connected(FC) 레이어로, 마스크는 Convolutional Autoencoder(CAE)로 각각 인코딩되어 $f_b$와 $f_m$을 생성한다.
- **Integration**: 두 특징을 결합한 $f = [f_m, f_b]$는 함수 $h_a$를 통해 잠재 표현 $Z \in \mathbb{R}^F$로 변환된다.
- **Decoder**: $Z$로부터 다시 $f'$를 생성하고, 각각의 디코더를 통해 원래의 바운딩 박스 $y_b$와 마스크 $y_m$을 재구성(Reconstruct)한다.

#### 훈련 목표 및 손실 함수
본 논문은 서로 다른 입력의 불확실성을 고려하기 위해 Maximum Log-likelihood 기반의 멀티태스크 손실 함수를 사용한다.

$$L(W, \sigma_m, \sigma_b) \propto \frac{1}{2\sigma_b^2} \|y_b - f_b^W(x)\|^2 + \frac{1}{2\sigma_m^2} \|y_m - f_m^W(x)\|^2 + \log \sigma_m + \log \sigma_b$$

여기서 $\sigma_b$와 $\sigma_m$은 각각 위치와 형태 재구성 작업의 학습 가능한 불확실성 파라미터이다. 이를 통해 모델은 특정 특징의 노이즈가 심할 경우 해당 손실의 가중치를 자동으로 낮추어 학습의 안정성을 높인다.

### 4. Sequential Data Constraints
시간적 일관성을 보장하기 위해 무방향 그래프 $G_t = (V_t, E_t)$를 구축하여 제약 조건을 부여한다.
- **Cannot-link ($f_{cl}$)**: 다음의 경우 동일한 클러스터에 속할 수 없다.
    - 같은 프레임 내에 존재하는 두 객체.
    - 시간적으로 매우 가깝지만 마스크의 IoU(Intersection over Union)가 0인 경우.
    - 서로 다른 클래스($\gamma_i \neq \gamma_j$)인 경우.
- **Must-link ($f_{ml}$)**: 이전 윈도우에서 동일한 식별자($l_i = l_j$)를 가졌던 객체들은 동일한 클러스터에 속하도록 강제한다.

### 5. Modified Constrained K-means
위의 제약 조건을 만족하면서 잠재 특징 $Z$ 간의 거리를 최소화하는 방향으로 클러스터링을 수행한다. 특히 클러스터의 개수 $|K|$를 미리 정하지 않고, 윈도우 내 최대 객체 수로 시작하여 제약 조건을 만족하지 못하는 새로운 탐지가 발생할 때 동적으로 클러스터를 추가하는 방식을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 합성 데이터셋(MNIST-MOT, Sprites-MOT) 및 실제 데이터셋(KITTI MOTS, MOTSChallenge)을 사용하였다.
- **평가 지표**: MOTA, Frag, IDs, MT, ML 및 sMOTSA, MOTSA, MOTSP 등을 사용하여 추적 성능과 세그멘테이션 일관성을 측정하였다.

### 주요 결과
1.  **합성 데이터셋**: 위치 정보와 형태 정보를 모두 사용하고 제약 그래프를 적용한 `loc+shape+G_t` 설정에서 거의 완벽한(Near-perfect) 성능을 달성하였다. 특히 t-SNE 시각화를 통해 객체가 서로 근접하더라도 잠재 공간에서는 분리됨을 확인하였다.
2.  **실제 데이터셋 (KITTI MOTS)**: 제안 방법론은 정교한 Re-ID나 모션 모델 없이도 기존의 baseline들(EagerMOT, GMPHD, MOTSNet 등)보다 우수한 sMOTSA 및 MOTSA 성능을 보였다. 특히 RGB 정보를 추가했을 때 보행자(Pedestrian) 클래스에서 성능 향상이 뚜렷했다.
3.  **실제 데이터셋 (MOTSChallenge)**: 제약 그래프를 적용했을 때 SOTA 방법론인 GMPHD와 대등한 수준의 성능을 보였으며, 이는 비지도 학습 방식임에도 불구하고 매우 경쟁력 있는 결과이다.

### Ablation Study
- **Uncertainty-aware MTL**: 단순 가중치 합산보다 불확실성 기반 손실 함수를 사용했을 때 sMOTSA가 유의미하게 상승하였다 (MOTSChallenge 기준 약 10.8% 향상).
- **Constraints Graph ($G_t$)**: 제약 그래프의 유무가 성능에 가장 큰 영향을 미쳤으며, 적용 시 sMOTSA가 최대 26.2%(KITTI person)까지 향상되었다.
- **Window Size ($t_{lag}$)**: 윈도우 크기가 너무 크면 모션 모델의 부재로 인해 Must-link 제약 위반이 증가하여 Fragmentation이 늘어나는 경향을 보였으며, $t_{lag}=3$일 때 최적의 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 비지도 학습 기반의 MOTS 프레임워크를 제안하며, 특히 서로 다른 성격의 데이터(위치, 형태)를 통합적으로 학습하는 DHAE와 시공간적 제약을 활용한 클러스터링의 효용성을 입증하였다.

**강점**으로는 별도의 레이블링된 데이터 없이도 위치와 외형 정보를 결합하여 강건한 특징 표현을 학습했다는 점과, 단순한 제약 조건을 통해 복잡한 추적 알고리즘 없이도 높은 시간적 일관성을 확보했다는 점을 들 수 있다.

**한계 및 논의사항**으로는 명시적인 모션 모델(Motion Model)이나 가려짐 추론(Occlusion Reasoning) 메커니즘이 부족하다는 점이 꼽힌다. 이로 인해 윈도우 크기($t_{lag}$) 설정에 민감하게 반응하며, 객체가 빠르게 움직이거나 장시간 가려질 경우 추적 성능이 저하될 가능성이 있다. 또한, 탐지기(Detector)의 신뢰도 임계값(Threshold)에 따라 결과가 변하므로, 최적의 임계값을 찾는 과정이 필요하다.

그럼에도 불구하고, 본 연구는 비지도 학습만으로도 SOTA 수준의 MOTS 성능에 근접할 수 있음을 보여주었으며, 향후 모션 예측 모델과 결합한다면 더욱 강력한 독립적 데이터 연관 메커니즘이 될 가능성이 높다.

## 📌 TL;DR

본 논문은 비지도 학습 방식으로 여러 객체를 추적하고 세그멘테이션하는 프레임워크를 제안한다. **위치와 형태 정보를 통합 학습하는 DHAE**와 **시공간적 제약 조건이 반영된 Constrained K-means**를 통해, 지도 학습 기반의 복잡한 모델 없이도 높은 수준의 시간적 일관성을 가진 객체 추적 성능을 달성하였다. 이 연구는 향후 비지도 기반의 비디오 인스턴스 세그멘테이션 및 객체 추적 연구에 중요한 기초를 제공한다.