# Spatial-Temporal Graph Mamba for Music-Guided Dance Video Synthesis

Hao Tang, Ling Shao, Zhenyu Zhang, Luc Van Gool, Nicu Sebe (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 입력된 음악 데이터를 기반으로 실사 같은 댄스 비디오를 생성하는 **Music-Guided Dance Video Synthesis** 작업이다. 음악과 비디오 데이터는 구조적 특성이 매우 다르기 때문에, 음악의 리듬과 스타일을 시각적인 댄스 동작으로 변환하는 매핑 과정은 매우 복잡하고 도전적인 과제이다.

기존 연구들은 주로 음악에서 스켈레톤(skeleton)을 생성하고 이를 다시 비디오로 변환하는 방식을 취했으나, 다음과 같은 문제점들이 존재하였다. 첫째, 스켈레톤 생성 단계에서 그래프 구조를 손실 함수(loss function) 계산 시에만 활용하여, 관절 간의 복잡한 공간적-시간적 의존성을 모델 학습 과정에 충분히 반영하지 못해 동작이 왜곡되는 현상이 발생하였다. 둘째, Transformer 기반 모델들은 긴 시퀀스를 처리할 때 계산 복잡도가 시퀀스 길이의 제곱에 비례하여 증가하므로 효율성이 떨어진다. 셋째, 스켈레톤-비디오 변환 과정에서 생성 순서의 불일치로 인해 시각적 아티팩트(artifact)가 발생하고 동작이 부자연스러운 문제가 있었다.

따라서 본 논문의 목표는 공간적-시간적 의존성을 효율적으로 캡처하는 새로운 아키텍처를 통해 자연스러운 댄스 스켈레톤을 생성하고, 자기지도 학습 기반의 정규화를 통해 고품질의 댄스 비디오를 합성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 **Spatial-Temporal Graph Mamba (STG-Mamba)** 구조를 도입하여 음악-스켈레톤-비디오로 이어지는 두 단계의 변환 매핑을 최적화하는 것이다.

1. **STGM (Spatial-Temporal Graph Mamba) 블록**: 각 관절을 그래프의 노드로 취급하고, Mamba의 State Space Model (SSM)을 활용하여 공간적(Spatial) 및 시간적(Temporal) 의존성을 선형 복잡도로 효율적으로 모델링한다. 특히 전방향 및 후방향 시간 SSM을 통해 시퀀스의 양방향 맥락을 모두 포착한다.
2. **자기지도 정규화 네트워크 (Self-supervised Regularization Network)**: 스켈레톤-비디오 변환 시, 단순한 베이스라인 생성 외에 전방향(Forward) 및 후방향(Backward) 생성 전략을 도입하고, 이들 간의 일관성을 강제하는 정규화 손실을 통해 비디오의 일관성과 품질을 높였다.
3. **대규모 데이터셋 구축**: 54,944개의 비디오 클립을 포함하며 발레, 팝핀, K-pop 등 다양한 스타일과 성별, 환경이 포함된 대규모 스켈레톤-비디오 변환 데이터셋을 수집하여 모델의 일반화 성능을 높였다.

## 📎 Related Works

기존의 음악 기반 댄스 생성 연구는 주로 CNN, RNN 및 Transformer를 사용하였다.

- **CNN/RNN 기반 방식**: 초기 연구들은 LSTM-autoencoder 등을 사용하였으나, 모션 매니폴드의 비유클리드 기하학적 특성으로 인해 학습의 불안정성과 다양성 부족 문제를 겪었다.
- **GCN (Graph Convolutional Networks) 기반 방식**: 스켈레톤을 그래프로 표현하여 관절 간 관계를 모델링하려 했으나, 일부 연구에서는 이를 단순히 손실 함수 계산 시의 퍼셉추얼 로스(perceptual loss)로만 활용하였다. 이는 모델의 내부 가중치가 관절 간의 실제 물리적 의존성을 충분히 학습하지 못하게 하여 왜곡된 동작을 생성하는 원인이 되었다.
- **Transformer 기반 방식**: DanceFormer와 같은 모델들은 뛰어난 성능을 보였으나, 시퀀스 길이에 따른 이차 복잡도($O(N^2)$)로 인해 계산 비용이 매우 크고 자원 제한적인 환경에서 배포가 어렵다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 선형 복잡도를 가진 **Mamba (Selective SSM)**를 그래프 구조와 결합하여 효율성과 정확성을 동시에 확보하고자 하였다.

## 🛠️ Methodology

전체 시스템은 **음악-스켈레톤 변환(Music-to-Skeleton Translation)**과 **스켈레톤-비디오 변환(Skeleton-to-Video Translation)**의 두 단계 파이프라인으로 구성된다.

### 1. Music-to-Skeleton Translation

음악 $M$과 노이즈 벡터 $z$를 입력받아 관절 위치 벡터 시퀀스 $S$를 생성한다.

- **음악 특징 추출**: 입력 음악을 0.1초 단위로 나누어 오디오 인코더와 양방향 GRU(Bi-GRU)를 통해 은닉 토큰 $H_0$를 추출한다.
- **STGM 블록**: $H_0$는 일련의 STGM 블록을 통과한다. 각 블록은 Layer Normalization(LN), Graph Convolution, SSM, 그리고 잔차 연결(Residual Connection)로 구성된다.
- **STGM의 핵심 구성 요소**:
  - **SG-SSM (Spatial Graph SSM)**: 단일 프레임 내에서 관절 간의 공간적 상호작용을 모델링하여 인체 구조의 세만틱 일관성을 유지한다.
  - **TGF-SSM & TGB-SSM (Temporal Graph Forward/Backward SSM)**: 프레임 간의 시간적 상관관계를 전방향과 후방향에서 각각 모델링하여 시퀀스의 양방향 맥락을 캡처한다.
- **수식 설명**: $l$번째 블록의 출력 $Z_l$은 다음과 같이 계산된다.
$$ Z'_l = \text{GraphConv1d}(\text{MLP}(\text{LN}(Z_{l-1}))) $$
$$ Z_l = \text{MLP}(\text{LN}(\text{SG}_{\text{SSM}}(\sigma(Z'_l))) \times \sigma(\text{MLP}(\text{LN}(Z_{l-1}))) + \text{LN}(\text{TGF}_{\text{SSM}}(\sigma(Z'_l))) \times \sigma(\text{MLP}(\text{LN}(Z_{l-1}))) + \text{LN}(\text{TGB}_{\text{SSM}}(\sigma(Z'_l))) \times \sigma(\text{MLP}(\text{LN}(Z_{l-1})))) $$
여기서 $\sigma$는 활성화 함수이며, 세 가지 SSM의 결과가 결합되어 최종 출력이 결정된다.
- **최적화 목표**: 포즈 퍼셉추얼 로스($L_p$), 특징 매칭 로스($L_f$), $L_1$ 재구성 로스($L_{l1}$)의 가중 합으로 구성된 $L_1$ 손실 함수를 사용하여 학습한다.

### 2. Skeleton-to-Video Translation

생성된 스켈레톤 시퀀스 $S$와 조건 이미지 $I_0$를 이용해 실사 비디오를 생성한다. Vid2Vid를 백본으로 사용하며, 그 위에 **자기지도 정규화 네트워크 $G_r$**을 추가하였다.

- **세 가지 생성 전략**:
  - **Baseline Generation**: $I_i = G_r(\text{concat}(I_0, S_i))$. 조건 이미지와 현재 스켈레톤만으로 프레임을 생성한다.
  - **Forward Generation (FSR)**: $\hat{I}_i = G_r(\text{concat}(I_{i-1}, S_i))$. 이전 프레임 $I_{i-1}$을 참조하여 다음 프레임을 생성함으로써 프레임 간 연속성을 높인다.
  - **Backward Generation (BSR)**: $\tilde{I}_i = G_r(\text{concat}(I_{i+1}, S_i))$. 다음 프레임 $I_{i+1}$을 참조하여 현재 프레임을 생성한다.
- **정규화 손실**: 전방향/후방향 생성 결과가 베이스라인 생성 결과와 일치하도록 강제하는 손실 함수를 도입한다.
$$ L_{fsr} = ||I_i - \hat{I}_i||_1, \quad L_{bsr} = ||I_i - \tilde{I}_i||_1 $$
- **최종 손실 함수**: $L_2 = \lambda_{gan}L_{gan} + \lambda_{l1}(L_{l1} + L_{fsr} + L_{bsr})$
- **추론 절차**: 학습 시에는 FSR과 BSR을 통해 모델을 강하게 제약하지만, 테스트 시에는 효율성을 위해 베이스라인 생성 전략만을 사용하여 결과를 도출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 기존 MDVS 데이터셋(K-pop, 발레, 팝핀) 및 직접 수집한 54,944개의 대규모 스켈레톤-비디오 데이터셋을 사용하였다.
- **평가 지표**:
  - 음악-스켈레톤: PFD(Pose Frechet Distance), VFD(Video Frechet Distance), PVar(Diversity), BC(Beat Consistency) 및 사용자 평가.
  - 스켈레톤-비디오: FID, LPIPS, FVD 및 사용자 선호도.

### 주요 결과

1. **음악-스켈레톤 생성 성능**:
    - 정량적으로 PFD $\downarrow$ (86.2), VFD $\uparrow$ (4.82), BC $\uparrow$ (0.684) 등 모든 지표에서 SOTA 모델인 EDGE 및 DanceFormer를 압도하였다.
    - 정성적으로 기존 방법들이 보이는 '반복적 동작'이나 '급격한 떨림(jerking)' 현상이 현저히 줄어들고 더 부드럽고 현실적인 동작을 생성하였다.
    - **추론 속도**: Mamba의 선형 복잡도 덕분에 Transformer 기반 모델(FACT, DanceFormer, EDGE)보다 훨씬 빠른 추론 시간을 기록하였다.
2. **스켈레톤-비디오 생성 성능**:
    - 사용자 선호도(40.5%)에서 Pix2pixHD나 Vid2Vid보다 높은 평가를 받았으며, FID(35.17) 및 FVD(945.4) 지표에서도 가장 우수한 성능을 보였다.
3. **Ablation Study**: SG-SSM, TGF-SSM, TGB-SSM 각각을 추가했을 때 성능이 단계적으로 향상됨을 확인하였으며, 세 가지를 모두 결합했을 때 최적의 성능이 나타났다. 또한, FSR과 BSR을 동시에 사용했을 때 비디오 품질이 가장 높게 나타났다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **Mamba 아키텍처를 그래프 구조에 성공적으로 통합**하여, Transformer의 강력한 모델링 능력과 RNN/CNN의 효율성을 동시에 잡았다는 점이다. 특히, 댄스 동작과 같이 긴 시퀀스의 데이터에서 전/후방향 SSM을 통해 시간적 의존성을 캡처한 것은 물리적으로 타당한 동작 생성에 결정적인 역할을 하였다.

또한, 단순한 모델 구조 개선에 그치지 않고, 비디오 생성의 고질적인 문제인 '오차 누적'과 '불일치' 문제를 해결하기 위해 **전/후방향 자기지도 정규화**라는 전략적 접근을 취한 점이 인상적이다. 이는 학습 단계에서 강력한 제약 조건을 부여함으로써 추론 시에는 가벼운 모델로도 고품질의 결과를 낼 수 있게 한다.

다만, 본 논문에서 제시한 데이터셋의 수집 과정이 구체적인 필터링 기준보다는 인터넷 수집 위주로 설명되어 있어, 데이터의 정제 수준에 대한 상세한 논의가 부족하다는 점이 아쉬움으로 남는다. 또한, 생성된 비디오의 실사 품질이 백본인 Vid2Vid의 성능에 어느 정도 의존하고 있는지에 대한 분석이 더 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 음악을 실사 댄스 비디오로 변환하기 위해 **STG-Mamba**라는 새로운 프레임워크를 제안한다. 공간적-시간적 그래프 Mamba 블록을 통해 관절 간의 복잡한 의존성을 선형 복잡도로 효율적으로 학습하여 자연스러운 댄스 동작을 생성하며, 전/후방향 자기지도 정규화 네트워크를 통해 시각적 아티팩트가 적은 고품질 비디오를 합성한다. 대규모 데이터셋 구축과 함께 실험을 진행한 결과, 기존 Transformer 기반 SOTA 모델들보다 더 빠르고 정확한 생성 성능을 입증하였다. 이 연구는 향후 실시간 댄스 생성 서비스나 복잡한 인체 동작 합성 연구에 중요한 기반이 될 가능성이 높다.
