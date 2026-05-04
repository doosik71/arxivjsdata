# SiamCorners: Siamese Corner Networks for Visual Tracking

Kai Yang, Zhenyu He, Wenjie Pei, Zikun Zhou, Xin Li, Di Yuan and Haijun Zhang (2021)

## 🧩 Problem to Solve

본 논문은 비주얼 트래킹(Visual Tracking) 분야에서 널리 사용되는 Anchor-based Siamese 네트워크, 특히 Region Proposal Network(RPN) 기반 방식들이 가진 한계점을 해결하고자 한다. 기존의 Anchor-based 트래커들은 다음과 같은 두 가지 주요 문제점을 안고 있다.

첫째, Anchor box의 설계와 관련된 복잡성과 비효율성이다. 수천 개의 Anchor box를 생성하여 Ground Truth(GT)와의 Intersection-over-Union(IoU)를 계산함으로써 긍정 및 부정 샘플을 정의하는데, 이 과정에서 긍정 샘플과 부정 샘플 간의 심각한 불균형이 발생한다. 이는 학습 속도를 저하시키고 분류 모델의 성능을 저하시켜, 결과적으로 회귀 네트워크의 정확도까지 떨어뜨리는 원인이 된다.

둘째, 하이퍼파라미터 튜닝의 어려움이다. Anchor box의 개수, 크기, 종횡비(Aspect Ratio)를 수동으로 설정해야 하며, 이는 주로 경험적인 휴리스틱에 의존한다. 이러한 고정된 설정은 타겟의 크기가 급격하게 변하거나 매우 작은 타겟을 추적할 때 유연하게 대응하지 못하는 한계를 가진다.

따라서 본 논문의 목표는 사전 정의된 Anchor box 없이 타겟을 추적하는 Anchor-free 트래커인 SiamCorners를 제안하여, 모델의 유연성을 높이고 설계의 복잡성을 제거하는 것이다.

## ✨ Key Contributions

SiamCorners의 핵심 아이디어는 바운딩 박스를 직접 예측하는 대신, 이를 두 개의 지점인 좌상단(Top-left) 코너와 우하단(Bottom-right) 코너의 쌍으로 정의하여 예측하는 것이다. 이를 위해 다음과 같은 핵심적인 설계 전략을 도입하였다.

1. **Anchor-free Proposal Network**: 사전 정의된 Anchor box를 완전히 제거함으로써 수동 설정의 번거로움을 없애고, 다양한 크기와 모양의 타겟에 대해 더 일반화된 성능을 제공한다.
2. **Modified Corner Pooling Layer**: 객체 탐지(Object Detection)의 CornerNet 개념을 트래킹에 맞게 수정하여, 타겟 특이적(Target-specific) 정보를 통합할 수 있는 구조를 설계하였다.
3. **Layer-wise Feature Aggregation**: ResNet-50의 서로 다른 깊이($conv3, conv4, conv5$)에서 추출된 특징들을 개별적으로 처리하고 통합함으로써, 저수준(Low-level)의 세밀한 정보와 고수준(High-level)의 의미론적 정보를 모두 활용한다.
4. **Novel Penalty Term**: 인접 프레임 간의 중심점, 너비, 높이의 변화량을 고려하는 새로운 페널티 함수를 도입하여, 급격한 변화나 이상치(Outlier)를 제거하고 최적의 바운딩 박스를 선택하도록 한다.

## 📎 Related Works

기존의 Siamese 네트워크 기반 트래커인 SiamRPN 및 SiamRPN++는 Anchor box를 통해 속도와 정확도의 균형을 맞추었으나, 앞서 언급한 Anchor 설계의 복잡성과 샘플 불균형 문제라는 한계가 있었다. 

최근 등장한 Anchor-free 트래커인 FCAF나 Ocean 등은 FCOS 탐지기를 기반으로 타겟의 중심점을 예측하고 사방의 거리를 회귀하는 방식을 사용한다. 그러나 이러한 방식들은 여전히 거친 위치를 찾기 위한 분류 네트워크 단계가 필요하여 연산 효율성 면에서 개선의 여지가 있다.

본 논문이 참고한 CornerNet은 바운딩 박스를 코너 쌍으로 예측하는 새로운 패러다임을 제시하였다. 하지만 CornerNet은 클래스 특정적(Class-specific)인 탐지기로 설계되어, 추적 대상이 매번 달라지는 비주얼 트래킹 작업에 직접 적용할 수 없다. SiamCorners는 이를 해결하기 위해 Siamese 구조 내에서 타겟 정보를 통합하는 수정된 Corner Pooling 레이어를 제안함으로써 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
SiamCorners는 템플릿 브랜치와 서치 브랜치로 구성된 Siamese 구조를 따른다. 템플릿으로는 초기 프레임에서 타겟의 상, 하, 좌, 우 경계 정보를 담은 4개의 경계 이미지($z_t, z_l, z_b, z_r$)를 크롭하여 사용한다. 이 이미지들은 ResNet-50 백본을 통해 특징 맵으로 변환된 후, 서치 이미지의 특징 맵과 Depth-wise correlation 연산을 거쳐 상관관계 특징 맵을 생성한다. 이후 Corner Pooling 레이어를 통해 코너 히트맵(Heatmap)과 오프셋(Offset)을 예측하고, 이를 디코딩하여 최종 바운딩 박스를 도출한다.

### 손실 함수 (Loss Function)
네트워크는 코너의 위치를 정확히 예측하기 위해 히트맵 손실($L_{tr}$)과 오프셋 손실($L_{off}$)의 합으로 구성된 전체 손실 함수를 최적화한다.

$$L = L_{tr} + \lambda L_{off}$$

히트맵 손실 $L_{tr}$은 Focal Loss의 변형을 사용하여 긍정 샘플과 부정 샘플의 불균형을 해소하며, 긍정 위치 주변의 페널티를 줄이기 위해 2D 가우시안 분포를 적용한다.

$$L_{tr} = -\frac{1}{K} \sum_{xy} \begin{cases} (1-\hat{X}_{xy})^\alpha \log(\hat{X}_{xy}) & \text{if } X_{xy} = 1 \\ (1-X_{xy})^\beta (\hat{X}_{xy})^\alpha \log(1-\hat{X}_{xy}) & \text{otherwise} \end{cases}$$

오프셋 손실 $L_{off}$는 네트워크의 스트라이드(Stride)로 인한 이산화 오차를 보정하기 위해 Smooth L1 Loss를 사용한다.

$$L_{off} = \frac{1}{K} \sum_{k=1}^K \text{SmoothL1Loss}(o_k, \hat{o}_k)$$

### Modified Corner Pooling
수정된 Corner Pooling 레이어는 상관관계 특징 맵에서 각 픽셀이 코너인지 여부를 판단한다. 예를 들어 우하단 코너를 예측하기 위해, 하단 경계 특징 맵 $f_b$에서는 가로 방향으로, 우측 경계 특징 맵 $f_r$에서는 세로 방향으로 Max-pooling을 수행하여 각 픽셀까지의 최대 활성화 값을 구하고 이를 합산한다.

$$b_{ij}^a = \max(f_{b,ij}^a, b_{i(j-1)}^a), \quad r_{ij}^a = \max(f_{r,ij}^a, r_{(i-1)j}^a)$$

이렇게 구해진 결과물에 Projection shortcut과 컨볼루션 레이어를 적용하여 최종 히트맵과 오프셋을 출력한다.

### 디코딩 및 포스트 프로세싱
예측된 히트맵에서 NMS(Non-maximum suppression)를 통해 상위 $N$개의 후보 코너 쌍을 추출한다. 이후 다층 특징 융합(Multi-level features fusion)을 통해 각 레이어($conv3, 4, 5$)에서 생성된 후보들을 통합한다. 최종 박스 선택을 위해 다음과 같은 페널티 전략을 사용한다.

1. **크기 및 비율 페널티**: 이전 프레임과 비교하여 너비, 높이 및 종횡비의 급격한 변화가 있는 후보의 점수를 낮춘다.
2. **변동성 인덱스 페널티**: 중심점, 너비, 높이의 절대적 변화량의 합을 계산하여 변화가 적은 박스에 가중치를 부여한다.
3. **선형 보간(Linear Interpolation)**: 최종 선택된 박스의 크기가 프레임 간에 부드럽게 변하도록 보정한다.

## 📊 Results

### 실험 설정
- **백본**: ImageNet으로 사전 학습된 ResNet-50
- **데이터셋**: OTB100, UAV123, LaSOT, NFS30, TrackingNet
- **하드웨어**: NVIDIA RTX 2080 GPU (속도 42 FPS 달성)
- **지표**: Success score (AUC), Precision score

### 주요 결과
SiamCorners는 대부분의 벤치마크에서 SOTA(State-of-the-art) 수준의 성능을 보였다.
- **OTB100**: ATOM보다 우수하며, DiMP보다는 약간 낮지만 기존 Anchor-based 방식인 DaSiamRPN보다 Success score 기준 1.3% 향상된 결과를 보였다.
- **UAV123**: Success score 61.4%, Precision score 81.9%를 기록하며 SiamRPN++를 능가하였다. 특히 빠른 움직임(Fast motion) 시나리오에서 DaSiamRPN보다 AUC score가 7.8% 높게 나타나, Anchor-free 방식의 강점을 입증하였다.
- **NFS30**: AUC 53.7%를 달성하며 SiamRPN++ 대비 4.5% 향상되었으며, 속도 또한 7 FPS 더 빨랐다.
- **LaSOT**: Success plot과 Normalized Precision plot 모두에서 최상위 성능을 기록하였다.

### 소거 연구 (Ablation Study)
- **백본 레이어**: 단일 레이어보다는 $conv3, 4, 5$를 모두 통합했을 때 UAV123에서 0.614 AUC로 가장 높은 성능을 보였다.
- **오프셋 네트워크**: 오프셋 네트워크 제거 시 LaSOT에서 3.9%의 성능 하락이 발생하여, 스트라이드 오차 보정의 중요성이 확인되었다.
- **코너 풀링**: Original Corner Pooling보다 Modified Corner Pooling을 사용했을 때 성능이 향상되었다.
- **템플릿 구성**: 하나의 통합 템플릿보다 4개의 경계 템플릿(FRT)을 사용했을 때 UAV123 기준 6.3%의 성능 향상이 있었다.

## 🧠 Insights & Discussion

본 논문은 Anchor-box라는 고정된 틀에서 벗어나 코너 쌍을 예측함으로써 트래킹 모델의 유연성을 극대화하였다. 특히 기존 Anchor-based 방식들이 겪었던 하이퍼파라미터 튜닝의 어려움과 샘플 불균형 문제를 근본적으로 해결하였다. 

실험 결과에서 나타나듯, 타겟의 크기 변화나 종횡비 변화, 특히 빠른 움직임이 발생하는 상황에서 높은 강건성을 보인 점은 매우 고무적이다. 이는 고정된 크기의 Anchor에 의존하지 않고 픽셀 수준에서 코너를 예측하기 때문에 가능한 결과이다.

다만, 연산 시간 분석(Table IV) 결과 ResNet-50의 마지막 블록인 $conv5$에서 가장 많은 시간이 소요됨을 확인하였다. 또한, 코너 풀링 연산 역시 세 번째로 많은 시간을 차지하고 있어, 향후 더 효율적인 풀링 연산 구조에 대한 연구가 필요할 것으로 보인다.

## 📌 TL;DR

SiamCorners는 비주얼 트래킹에서 복잡한 Anchor box 설계를 제거하고, 타겟을 좌상단과 우하단 코너의 쌍으로 예측하는 **Anchor-free Siamese 네트워크**이다. 수정된 Corner Pooling 레이어, 다층 특징 융합, 그리고 프레임 간 변동성을 제어하는 페널티 함수를 통해 **높은 정확도와 실시간 속도(42 FPS)**를 동시에 달성하였다. 특히 빠른 움직임과 급격한 크기 변화가 있는 환경에서 기존 Anchor-based 방식보다 뛰어난 성능을 보이며, 향후 Anchor-free 트래킹 연구에 중요한 기반을 제공할 것으로 기대된다.