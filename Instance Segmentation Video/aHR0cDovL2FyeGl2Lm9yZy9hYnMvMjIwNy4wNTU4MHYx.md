# Online Video Instance Segmentation via Robust Context Fusion

Xiang Li, Jinglu Wang, Xiaohao Xu, Bhiksha Raj, Yan Lu (2022)

## 🧩 Problem to Solve

본 논문은 **Online Video Instance Segmentation (VIS)** 문제를 해결하고자 한다. VIS는 비디오 시퀀스 내에서 객체 인스턴스를 동시에 분류(Classifying), 분할(Segmenting), 그리고 추적(Tracking)하는 것을 목표로 한다.

최근 Transformer 기반의 신경망들이 시공간적 상관관계(Spatio-temporal correlations) 모델링에서 뛰어난 성능을 보이고 있으나, 이들은 주로 비디오 전체나 클립 단위(Clip-level) 입력을 처리하기 때문에 **높은 지연 시간(Latency)과 계산 비용**이라는 치명적인 단점이 있다. 실제 스트리밍 애플리케이션에 적용하기 위해서는 현재 프레임과 소수의 이전 참조 프레임(Reference frames)만을 사용하여 프레임별로 예측을 수행하는 온라인 방식의 접근이 필수적이다.

또한, 기존의 온라인 VIS 방법들은 다음과 같은 한계를 가진다:
1. **특징 수준의 상관관계 무시**: 프레임별 독립 예측 후 후처리 단계에서 매칭 알고리즘을 사용할 경우, 계산 비용이 높고 결과가 깜빡거리는(Flickering) 현상이 발생한다.
2. **참조 프레임의 중복성 및 노이즈**: 단순한 특징 결합(Concatenate) 방식은 참조 프레임과 타겟 프레임의 중요도 차이를 무시하며, 타겟과 무관한 참조 특징이 예측을 방해할 수 있다.
3. **단일 모달리티 의존**: 시각 정보 외에 오디오와 같은 추가적인 모달리티가 제공될 수 있음에도 불구하고, 이를 밀집 예측(Dense prediction) 작업에 활용하려는 시도가 부족했다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Robust Context Fusion (RCF)** 네트워크를 통해 효율적이고 콤팩트한 컨텍스트를 추출하고, 이를 타겟 프레임에 융합하여 정밀하고 일관된 예측을 수행하는 것이다.

주요 기여 사항은 다음과 같다:
1. **Robust Context Fusion (RCF) 모듈 제안**: 참조 특징의 중복성을 줄이기 위해 중요도 기반 압축(Importance-aware compression)을 수행하고, 이를 Transformer Encoder를 통해 타겟 특징과 융합함으로써 온라인 VIS에서 SOTA 성능을 달성하였다.
2. **Matching-free 인스턴스 추적**: 네트워크의 Lipschitz 연속성(Lipschitz continuity)을 활용하여, 별도의 복잡한 매칭 알고리즘 없이 인스턴스 코드의 슬롯(Slot) 인덱스만으로 정체성(Identity)을 유지하는 **순서 보존(Order-preserving) 인스턴스 임베딩** 방법을 제안하였다.
3. **오디오-비주얼 VIS 탐구**: 오디오 신호가 VIS 작업에 미치는 영향을 분석하기 위해 새로운 **AVIS 데이터셋**을 구축하고, 멀티모달 융합 프레임워크를 통해 오디오 신호의 유용성을 검증하였다.

## 📎 Related Works

### Video Instance Segmentation (VIS)
- **Online VIS**: Mask-Track-RCNN, SipMask, SG-Net 등이 있으며, 주로 프레임 기반 예측 후 추적 헤드를 통해 정체성을 연결한다. CrossVIS는 글로벌 인스턴스 임베딩을 통해 견고한 추적을 시도했다.
- **Offline VIS**: VisTR, IFC 등이 있으며, 전체 클립을 동시에 처리하여 시공간적 관계를 모델링하지만 지연 시간이 매우 길다.

### Video Object Segmentation (VOS) 및 Image Instance Segmentation
- VOS는 클래스 구분 없이 마스크를 추적하며, STM 네트워크와 같이 메모리 뱅크를 활용하는 방식이 주를 이룬다.
- 이미지 인스턴스 분할은 Bottom-up(객체 중심, SOLO 등)과 Top-down(Bounding-box 기반, Mask R-CNN 등) 방식으로 나뉜다.

### Audio-Visual Representative Learning
- 기존 연구들은 주로 사운드 로컬라이제이션(Sound localization)이나 오디오-비주얼 분리(Separation)에 집중해 왔으나, 본 논문과 같이 인스턴스 분할과 같은 밀집 예측(Dense prediction) 작업에 오디오를 결합한 연구는 거의 없었다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
시스템은 크게 **특징 추출 $\rightarrow$ Robust Context Fusion (RCF) $\rightarrow$ 디코더 $\rightarrow$ 인스턴스 마스크 생성** 순으로 구성된다. 타겟 프레임 $I_t$와 참조 프레임 $\{I_r\}_{r=t-\delta}^{t-1}$이 입력으로 들어오면, 공유된 Backbone을 통해 특징을 추출하고 이를 RCF 모듈에서 융합한다.

### Robust Context Fusion (RCF)
RCF의 목적은 참조 데이터의 노이즈를 줄이고 타겟 프레임의 중요도를 높이는 것이다.

1. **타겟 토큰 ($O_{tgt}$)**: 타겟 프레임은 공간적/의미적 단서가 가장 중요하므로, 채널 차원만 압축하고 공간 차원은 유지한다.
   $$O_{tgt} = P(\phi_{C7 \to C'}(f_t))$$
   여기서 $\phi_{C7 \to C'}$는 $1 \times 1$ convolution 레이어이며, $P$는 flatten 및 positional encoding 추가 연산이다.

2. **참조 토큰 ($O_{ref}$)**: 참조 프레임은 중복성이 높으므로 공간 및 채널 차원을 모두 압축한다.
   $$O_{ref} = P(\phi_{\delta \cdot C7 \to C'}(\phi_{H \times W7 \to K \times K}(W \cdot f^{ref}_t)))$$
   - $W$: 학습 가능한 픽셀 단위 가중치 맵으로, 배경 노이즈를 제거하고 전경에 집중하게 한다.
   - $\phi_{H \times W7 \to K \times K}$: 공간 차원을 $H \times W$에서 $K \times K$로 압축하는 pooling 또는 depth-wise conv 레이어이다.

3. **토큰 융합**: $O_{tgt}$와 $O_{ref}$를 결합하여 Transformer Encoder에 입력하며, 출력된 융합 토큰 $O' = O'_{tgt} \oplus O'_{ref}$를 얻는다.

### 디코더 및 인스턴스 생성
- **인스턴스 코드 ($e_t$)**: 학습 가능한 인스턴스 쿼리(Learnable instance query)를 사용하여 Transformer Decoder를 통해 각 인스턴스의 정체성을 나타내는 코드 $e_t \in \mathbb{R}^{C_e \times N}$를 생성한다.
- **분할 맵 ($S_t$)**: 융합된 타겟 토큰 $O'_{tgt}$를 Mask Decoder에 입력하여 세그멘테이션 맵 $S_t$를 생성한다.
- **최종 마스크 ($M_t$)**: 인스턴스 코드 $e_t$로부터 동적 필터 $\theta_t$를 생성하고, 이를 $S_t$에 적용하는 **동적 컨볼루션(Dynamic Convolution)**을 통해 최종 마스크를 얻는다.
  $$M_t = \theta_t^T S_t$$

### 손실 함수 (Loss Function)
헝가리안 알고리즘(Hungarian algorithm)을 통해 예측치와 Ground-truth 간의 최적 할당 $\hat{\sigma}$를 찾고, 다음과 같은 손실 함수를 최소화한다.
$$L = \sum_{i=0}^{N} -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} \text{Dice}(\hat{m}_{\hat{\sigma}(i)}, m_i)$$
분류 손실(Cross-entropy)과 마스크 손실(Dice loss)의 합으로 구성된다.

### Order-preserving 인스턴스 추적
본 논문은 네트워크의 **Lipschitz 연속성**을 근거로, 입력 영상의 변화가 작으면 출력값의 변화도 작다는 점을 이용한다. 즉, $\lVert I_t - I_{t-1} \rVert \to 0$일 때 $\lVert \Theta(I_t) - \Theta(I_{t-1}) \rVert \to 0$이 성립한다. 따라서 인스턴스 코드 $e_t$의 슬롯 순서가 유지된다고 가정하며, 별도의 매칭 없이 **슬롯 인덱스 자체를 인스턴스 ID로 사용**한다.

### 오디오-비주얼 확장
- **오디오 특징 추출**: Raw audio $\rightarrow$ STFT $\rightarrow$ Log-Mel Spectrogram $\rightarrow$ VGG $\rightarrow$ bi-LSTM 순으로 처리하여 오디오 토큰 $O_{aud}$를 생성한다.
- **융합**: $O_{aud}$를 시각 토큰과 동일하게 RCF 모듈의 Transformer Encoder에 입력하여 cross-modal attention을 통해 융합한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Youtube-VIS 2019, 2021 및 신규 구축한 AVIS 데이터셋.
- **백본**: ResNet-101, Swin-B, Swin-L.
- **지표**: mAP (mean Average Precision), AR (Average Recall).

### 주요 결과
1. **온라인 VIS 성능**: ResNet-101 기준, Youtube-VIS 2019에서 **40.8 mAP**를 기록하여 기존 온라인 SOTA인 PCAN(37.6 mAP)을 크게 상회하였다. 특히 Swin-L 백본 사용 시 47.6 mAP까지 성능이 향상되었다.
2. **오프라인 방법과의 비교**: 비디오 전체를 입력으로 받는 VisTR보다 높은 성능을 보이면서도, 지연 시간(Latency)은 획기적으로 줄였다 (오프라인 IFC: 6.4s $\rightarrow$ 제안 방법: 0.171s).
3. **AVIS 데이터셋 실험**: 오디오 정보를 추가했을 때 mAP가 소폭 상승(최대 +1.9)하는 경향을 보였으나, t-test 결과 통계적으로 유의미한 수준(p-value $\le 0.05$)은 아니었다. 이는 실제 야생(In-the-wild) 시나리오에서 오디오-비주얼 간의 상관관계가 약하기 때문으로 분석된다.

### Ablation Study
- **압축의 효과**: 참조 특징을 압축하지 않고 그대로 융합했을 때 성능이 급격히 저하되었다. 이는 Softmax 연산 시 타겟 프레임의 중요도가 희석되기 때문이며, 본 논문의 **중요도 기반 압축**이 온라인 설정에서 필수적임을 입증하였다.
- **참조 프레임 수**: 1~3개 정도의 참조 프레임이 가장 효과적이었으며, 너무 많아지면 오히려 불필요한 정보가 유입되어 성능이 하락하였다.
- **순서 보존 감독**: 명시적인 순서 보존 손실 함수를 추가해도 성능 향상이 없었다. 이는 순서 보존이 학습된 네트워크의 자연스러운 속성임을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 통찰
- **온라인 VIS의 특수성 이해**: 오프라인 VIS와 달리 온라인 VIS에서는 타겟 프레임의 절대적 중요성이 높다. 이를 위해 참조 토큰의 크기를 줄여 타겟 토큰이 Attention 메커니즘을 주도하게 만든 설계가 매우 효과적이었다.
- **추적 패러다임의 전환**: 복잡한 매칭 알고리즘 대신 네트워크의 수치적 안정성(Lipschitz continuity)에 기반한 인덱스 추적 방식을 제안함으로써 추론 효율성을 극대화하였다.

### 한계 및 비판적 해석
- **오디오 활용의 한계**: 오디오 모달리티를 도입했음에도 성능 향상이 미미했다는 점은 아쉽다. 저자들은 이를 "야생 시나리오의 약한 상관관계"와 "짧은 참조 윈도우" 탓으로 돌렸으나, 이는 단순히 융합 방식의 문제일 수도 있다. 더 긴 시간 범위의 오디오 컨텍스트를 캡처할 수 있는 구조가 필요해 보인다.
- **순서 보존의 가정**: Lipschitz 연속성에 기반한 추적은 인접 프레임 간 변화가 작을 때만 유효하다. 급격한 화면 전환이나 매우 작은 객체가 빠르게 움직이는 경우, 인덱스 기반 추적이 실패할 가능성이 크며, 이를 보완하기 위해 IoU 기반의 예외 처리를 추가한 점은 이 방법의 잠재적 불안정성을 반증한다.

## 📌 TL;DR

본 논문은 온라인 비디오 인스턴스 분할(VIS)을 위해 **중요도 기반의 참조 특징 압축(RCF)**과 **순서 보존형 인스턴스 임베딩**을 제안하였다. 이를 통해 계산 비용을 획기적으로 줄이면서도 SOTA 성능을 달성했으며, 특히 복잡한 매칭 과정 없이 인덱스만으로 객체를 추적하는 효율적인 방식을 제시하였다. 또한 최초로 오디오-비주얼 VIS 데이터셋(AVIS)을 구축하여 멀티모달 확장 가능성을 탐색하였다. 이 연구는 실시간 비디오 분석 시스템의 추론 속도와 정확도를 동시에 잡을 수 있는 실용적인 프레임워크를 제공한다.