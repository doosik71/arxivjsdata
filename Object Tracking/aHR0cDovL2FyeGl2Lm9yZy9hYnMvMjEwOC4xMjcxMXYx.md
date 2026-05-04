# Learning to Track Objects from Unlabeled Videos

Jilai Zheng, Chao Ma, Houwen Peng, Xiaokang Yang (2021)

## 🧩 Problem to Solve

본 논문은 레이블이 없는 비디오(unlabeled videos)로부터 직접 학습 가능한 Unsupervised Single Object Tracker (USOT)를 구축하는 것을 목표로 한다. 기존의 지도 학습(supervised learning) 기반 추적기는 방대한 양의 어노테이션 데이터가 필요하며, 이는 비용과 시간이 많이 소요되는 문제점이 있다. 이를 해결하기 위해 등장한 기존의 비지도 학습 추적기들은 다음과 같은 세 가지 핵심적인 성능 병목 현상을 가지고 있다.

첫째, **Moving Object Discovery**의 부재이다. 정답 바운딩 박스(ground truth bounding box)가 없기 때문에 기존 방식들은 비디오에서 무작위로 영역을 크롭(random cropping)하여 가짜 템플릿으로 사용하였다. 이는 객체와 배경을 구분하는 능력을 저하시키며, 특히 스케일 변화를 추정하기 위한 바운딩 박스 회귀(regression) 학습을 불가능하게 만든다.

둘째, **Rich Temporal Variation Exploitation**의 한계이다. 기존 비지도 학습 방식은 매우 짧은 시간 범위(예: 10프레임 미만) 내에서만 전후방 추적을 수행한다. 이처럼 짧은 구간에서는 객체의 외형 변화가 거의 없기 때문에, 긴 시간 동안 발생하는 다양한 외형 및 모션 변화를 학습할 수 없다.

셋째, **Online Update**의 어려움이다. 지도 학습 기반 추적기는 온라인 모듈을 통해 템플릿을 업데이트하며 성능을 높이지만, 비지도 학습에서는 객체의 대략적인 위치조차 알 수 없기 때문에 온라인 업데이트 브랜치를 학습시키는 것이 매우 어렵다.

결과적으로 본 논문의 목표는 이러한 세 가지 난제를 해결하여 비지도 학습 추적기와 지도 학습 추적기 간의 성능 격차를 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오에서 움직이는 객체를 스스로 발견하고, 이를 단계적으로 확장하여 학습시키는 3단계 학습 파이프라인을 설계한 것이다.

1. **움직이는 객체의 정교한 발견**: 단순 무작위 크롭 대신, 비지도 광학 흐름(unsupervised optical flow)과 동적 계획법(Dynamic Programming, DP)을 결합하여 움직이는 객체의 부드러운 바운딩 박스 시퀀스를 생성한다.
2. **단계적 학습 전략**: 먼저 단일 프레임 쌍을 이용해 Naive Siamese Tracker를 학습시켜 기초적인 초기값을 확보한 후, 이를 더 긴 시간 범위로 확장하여 학습시킨다.
3. **Cycle Memory Learning**: 전방 추적 결과들을 메모리 큐(memory queue)에 저장하고 다시 역방향으로 추적하여 일관성을 맞추는 사이클 메모리 학습 방식을 제안한다. 이를 통해 긴 시간 동안의 외형 변화를 학습함과 동시에 온라인 업데이트 기능을 구현한다.

## 📎 Related Works

**Supervised Tracking**
최근의 딥러닝 기반 추적기는 Siamese 네트워크 구조를 주로 사용한다. SiamFC는 템플릿과 탐색 영역의 특징을 추출해 cross-correlation을 계산하며, 이후 SiamRPN, DiMP, Ocean 등은 RPN 도입, 온라인 업데이트 모듈 추가, 앵커 프리 회귀 등을 통해 성능을 고도화하였다. 그러나 이들은 모두 대량의 레이블된 데이터가 필수적이다.

**Unsupervised Tracking**
비지도 학습 기반 추적기는 주로 자기 일관성(self-consistency)이나 사이클 일관성(cycle-consistency)을 pretext task로 활용한다. UDT는 전후방 추적의 일관성을 이용하며, $S^2SiamFC$는 단일 프레임 쌍에서 전경/배경 분류기를 학습시킨다. 하지만 본 논문에서 지적했듯이, 이들은 무작위 샘플링에 의존하거나 학습 시간 범위가 너무 짧아 지도 학습 기반 모델과의 성능 격차가 크게 존재한다는 한계가 있다.

## 🛠️ Methodology

본 논문이 제안하는 비지도 학습 체계는 총 3단계로 구성된다.

### 3.1 Moving Object Discovery

무작위 크롭의 문제를 해결하기 위해, 비지도 광학 흐름과 DP를 사용하여 움직이는 객체의 궤적을 생성한다.

1. **Candidate Box Generation**: 비지도 알고리즘인 ARFlow를 사용하여 프레임 $I_t$와 $I_{t+T_f}$ 사이의 광학 흐름 맵 $F_t$를 계산한다. 이후 거리 맵 $D_t$를 생성하고 다음과 같은 식을 통해 이진 마스크 $M_t$를 얻는다.
    $$M^i_t = \begin{cases} 1 & \text{if } D^i_t \geq \alpha \cdot \max_j(D^j_t) + (1-\alpha) \cdot \text{mean}_j(D^j_t) \\ 0 & \text{o.w.} \end{cases}$$
    여기서 $D^i_t = \| F^i_t - \text{mean}_j(F^j_t) \|^2$ 이다. 마스크에서 연결된 영역을 기반으로 후보 박스를 생성하며, 이미지 중앙 편향을 고려한 품질 점수 $S_c$가 가장 높은 박스를 최종 후보 박스 $B_t$로 선택한다.

2. **Box Sequence Generation**: 후보 박스들의 노이즈를 제거하기 위해 DP를 적용한다. 궤적의 부드러움을 보장하기 위해 DIoU 메트릭을 수정한 보상 함수 $R_{dp}$를 사용한다.
    $$R_{dp}(B_t, B_{t'}) = \text{IoU}(B_t, B_{t'}) - \gamma \cdot R_{DIoU}(B_t, B_{t'})$$
    $\gamma > 1$로 설정하여 거리 패널티를 강화함으로써 최적의 부드러운 경로를 찾는다. 선택되지 않은 프레임은 선형 보간법(linear interpolation)으로 채운다.

### 3.2 Naive Siamese Tracker

생성된 박스 시퀀스 $B'$를 사용하여 처음부터 Siamese 추적기를 학습시킨다.

1. **데이터 필터링**: 비디오 수준 점수 $Q_v$ (DP에 의해 선택된 프레임 비율)와 프레임 수준 점수 $Q_f$ (인접 프레임 중 DP 선택 비율)를 사용하여 저품질 박스를 제거한다.
2. **아키텍처 및 손실 함수**: ResNet-50을 백본으로 사용하며, PrPool로 템플릿 특징을 풀링한 후 multi-scale correlation을 계산한다. 출력은 전경/배경 분류를 위한 $R_{cls}$와 바운딩 박스 회귀를 위한 $R_{reg}$이다. 손실 함수는 다음과 같다.
    $$L_{naive} = L_{reg} + \lambda_1 L_{cls}$$
    여기서 $L_{reg}$는 IoU loss, $L_{cls}$는 Binary Cross-Entropy (BCE) loss이다.

### 3.3 Cycle Memory Training

Naive 추적기의 한계(시간적 변화 학습 부족, 온라인 업데이트 불가)를 극복하기 위한 단계이다.

1. **학습 절차**: 템플릿 $z_t$로부터 $N_{mem}$개의 인접 프레임으로 전방 추적을 수행하고, 그 결과들을 메모리 큐에 저장한다. 이후 다시 원래의 탐색 영역 $x_t$로 역방향 추적을 수행하여 일관성 손실 $L_{mem}$을 계산한다.
2. **프레임 범위 설정**: 객체가 사라지지 않으면서도 충분한 변화를 학습할 수 있도록, DP 궤적의 연속성과 프레임 품질 점수 $Q_f$를 기준으로 동적으로 시간 범위 $[T_l, T_u]$를 설정한다.
3. **특징 통합 (Confidence-Value Strategy)**: 메모리 큐의 각 상관 맵 $\{C^u_{corr}\}$를 통합하기 위해, 신뢰도 맵 $C^u_{conf}$와 값 맵 $C^u_{val}$을 생성하고 다음과 같이 통합 상관 맵 $C$를 구한다.
    $$C = \sum_{1 \le u \le N_{mem}} \text{softmax}(C^u_{conf}) \odot C^u_{val}$$
    최종 손실 함수는 다음과 같다.
    $$L = L_{reg} + \lambda_1 L_{cls} + \lambda_2 L_{mem}$$

## 📊 Results

### 실험 설정

- **데이터셋**: GOT-10k, ImageNet VID, LaSOT, YouTube-VOS의 레이블 없는 학습 데이터를 사용하였다.
- **비교 대상**: 비지도 추적기(LUDT, LUDT+, $S^2SiamFC$ 등) 및 지도 학습 추적기(SiamFC, ATOM, DiMP 등).
- **평가 지표**: Accuracy (A), Robustness (R), Expected Average Overlap (EAO), Success rate, Precision.

### 주요 결과

- **VOT 벤치마크**: VOT2016, 2017/18, 2020 모두에서 기존 최신 비지도 추적기인 LUDT+를 큰 차이로 능가하였다. 특히 VOT2017/18에서 USOT*는 EAO 기준 LUDT+보다 11.4포인트 높은 성능을 보였다.
- **TrackingNet**: USOT*는 LUDT+ 대비 Success 5.2, Precision 7.1 포인트 상승을 기록하며 지도 학습 모델에 근접한 성능을 보였다.
- **LaSOT (장기 추적)**: 평균 길이가 2000프레임이 넘는 LaSOT에서 USOT*가 LUDT+를 크게 앞질렀으며, 이는 제안된 Cycle Memory 학습이 장기적인 외형 변화를 효과적으로 학습했음을 증명한다.

### 절제 연구 (Ablation Study)

- **학습 단계의 필수성**: 무작위 박스를 사용하거나 Naive Siamese 학습 단계 없이 바로 Cycle Memory 학습을 진행했을 때 성능이 급격히 하락함을 확인하여, 제안한 3단계 파이프라인의 정당성을 입증하였다.
- **시간 범위**: 기존 방식($<10$ 프레임)과 달리 본 방법은 평균 41.1~64.6 프레임의 긴 간격에서 학습이 가능함을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 비지도 학습의 고질적인 문제인 '부정확한 가짜 레이블' 문제를 **Optical Flow $\rightarrow$ DP $\rightarrow$ 단계적 학습**이라는 체계적인 파이프라인으로 해결했다는 점이다. 특히, 단일 프레임 학습(Naive)에서 시작해 점진적으로 시간 범위를 넓혀가는 전략은 네트워크가 갑작스러운 변화에 무너지지 않고 안정적으로 학습될 수 있도록 돕는다.

또한, 실험 결과에서 USOT(비지도 백본)가 USOT*(ImageNet 사전학습 백본)와 유사하거나 특정 벤치마크(OTB2015)에서는 더 나은 성능을 보인 점은, 객체 추적 작업에 있어 일반적인 이미지 분류 데이터보다 비디오 자체에서 추출한 비지도 표현 학습이 더 효율적일 수 있음을 시사한다.

다만, 비지도 광학 흐름(ARFlow)에 의존하여 초기 박스를 생성하므로, 광학 흐름 자체가 실패하는 매우 복잡한 배경이나 극심한 조명 변화 상황에서의 강건성에 대해서는 추가적인 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 레이블 없는 비디오에서 **1) 광학 흐름과 DP 기반의 객체 발견 $\rightarrow$ 2) 단일 프레임 Naive 학습 $\rightarrow$ 3) 사이클 메모리 기반 장기 학습 및 온라인 업데이트**로 이어지는 3단계 비지도 학습 프레임워크(USOT)를 제안한다. 이를 통해 기존 비지도 추적기의 성능 병목을 해결하여, 최신 비지도 모델을 압도하고 지도 학습 모델에 근접하는 성능을 달성하였다. 이 연구는 대규모 레이블링 작업 없이도 고성능 추적기를 학습시킬 수 있는 가능성을 제시하여 향후 비지도 비디오 분석 연구에 중요한 기여를 할 것으로 보인다.
