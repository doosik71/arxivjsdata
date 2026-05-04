# SiamTHN: Siamese Target Highlight Network for Visual Tracking

Jiahao Bao, Kaiqiang Chen, Xian Sun, Liangjin Zhao, Wenhui Diao, Menglong Yan (2021)

## 🧩 Problem to Solve

본 논문은 Siamese 네트워크 기반의 비주얼 객체 추적(Visual Object Tracking) 모델들이 가진 두 가지 주요 문제점을 해결하고자 한다.

첫 번째 문제는 **유사도 응답 맵(Similarity Response Map)의 배경 영향성**이다. 기존의 Siamese 추적기들은 백본 네트워크에서 생성된 특징 맵(Feature Map)의 각 채널을 동일하게 처리한다. 이로 인해 생성된 유사도 응답 맵이 타겟 영역에 집중하지 못하고 배경의 방해 요소에 민감하게 반응하게 되며, 결과적으로 후속 단계인 추적 헤드(Tracking Head)의 특징 디코딩 효율을 저하시킨다.

두 번째 문제는 **분류(Classification) 브랜치와 회귀(Regression) 브랜치 간의 정렬 불일치(Misalignment)**이다. 기존 모델들은 두 브랜치가 구조적으로 연결되어 있지 않고 학습 과정에서도 각각 독립적으로 최적화된다. 그러나 실제 추론 단계에서는 분류 브랜치의 결과(최대 신뢰도 점수)를 바탕으로 회귀 브랜치의 바운딩 박스를 선택하기 때문에, 분류 점수는 높지만 실제 위치 정확도는 낮은 예측 결과가 발생하는 불일치 문제가 나타난다.

따라서 본 논문의 목표는 타겟 영역에 더 집중할 수 있는 특징 선택 메커니즘을 도입하고, 두 브랜치를 공동으로 최적화할 수 있는 학습 방법을 제안하여 추적의 정확도와 신뢰성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특징 맵의 차원 축소 과정에서 타겟 관련 특징을 동적으로 강화하는 모듈을 도입하고, 손실 함수를 통해 두 브랜치의 상관관계를 학습시키는 것이다.

1. **Target Highlight Module (THM) 제안**: 채널 차원 축소 과정에서 타겟과 관련된 채널은 강화하고 중요하지 않은 채널은 억제하는 동적 채널 특징 밸런싱을 수행한다. 이를 통해 유사도 응답 맵이 타겟 영역에 더 명확하게 집중되도록 한다.
2. **Corrective Loss 제안**: 추가적인 네트워크 구조 변경 없이, 분류 손실(Classification Loss)을 계수로 사용하여 회귀 손실(Regression Loss)을 보정하는 새로운 손실 함수를 제안한다. 이를 통해 분류와 회귀 브랜치가 서로 조율되어 최적화되도록 유도함으로써 예측 결과의 일관성을 높인다.
3. **SiamTHN 모델 구축**: 위 두 가지 기여를 통합하여 SiamBAN을 기반으로 한 Siamese Target Highlight Network(SiamTHN)를 개발하였으며, 실시간성(38 fps)과 높은 정확도를 동시에 달성하였다.

## 📎 Related Works

### Siamese Network 기반 추적기

SiamFC를 시작으로 SiamRPN, SiamRPN++ 등 다양한 모델이 등장하였다. SiamRPN은 RPN을 도입하여 스케일 변화 문제를 해결하려 했으나 앵커(Anchor) 설정에 의존하는 한계가 있었다. 이를 해결하기 위해 SiamBAN, SiamCAR와 같은 Anchor-free 추적기들이 제안되었다. 하지만 이러한 모델들 역시 유사도 응답 맵이 배경에 민감하고, 헤드의 두 브랜치가 독립적으로 학습된다는 공통적인 한계를 가지고 있다.

### 어텐션 메커니즘 (Attentional Mechanisms)

SENet, ECANet 등 다양한 채널 어텐션 기법들이 백본 네트워크의 특징 추출 능력을 높이기 위해 사용되어 왔다. 기존의 추적기들(예: RASNet) 또한 어텐션을 사용했지만, 이는 주로 백본 네트워크 내부의 특징 표현력을 높이는 데 집중되었다. 반면, 본 논문의 THM은 백본이 아닌 **유사도 매칭 모듈(Similarity Matching Module)** 내의 차원 축소 과정에 배치되어 타겟 하이라이팅이라는 구체적인 목적을 수행한다는 점에서 차별화된다.

### 바운딩 박스 지역화 전략 (Bounding Box Localization Strategy)

분류와 회귀 브랜치 간의 불일치 문제를 해결하기 위해 IoU-Net, PISA 등에서 예측된 IoU를 활용하거나 추가적인 지역화 브랜치를 도입하는 방식이 제안되었다. 그러나 이러한 방식은 모델의 복잡도를 증가시킨다. 본 논문은 추가적인 구조적 변경 없이 오직 손실 함수(Corrective Loss)만으로 이 문제를 해결하려 한다는 점이 기존 접근 방식과의 차이점이다.

## 🛠️ Methodology

### 전체 시스템 구조

SiamTHN은 기본적으로 SiamBAN 구조를 따르며, 크게 **Siamese Network**, **Similarity Matching Module**, **Tracking Head**의 세 부분으로 구성된다.

- **Siamese Network**: 수정된 ResNet-50을 백본으로 사용하여 템플릿과 검색 이미지에서 특징 맵을 추출한다.
- **Similarity Matching Module**: 추출된 특징 맵을 입력으로 하여 유사도 응답 맵을 생성한다. 이때 본 논문에서 제안한 **THM**이 적용되어 DW-Xcorr(Depth-wise Cross Correlation) 이전에 특징을 최적화한다.
- **Tracking Head**: 유사도 응답 맵을 입력받아 분류 점수 맵과 바운딩 박스 회귀 맵을 생성한다.

### Target Highlight Module (THM)

THM은 DW-Xcorr에 입력되기 전, 채널 차원 축소 과정에서 타겟 관련 정보를 보존하고 강화하는 역할을 한다.

1. **Squeeze**: 평균 풀링(Average Pooling)을 통해 특징 맵의 공간 차원을 압축한다.
2. **Excitation**: 두 개의 컨볼루션 레이어를 통해 채널 간의 상호작용을 학습하고 시그모이드($\delta$) 함수를 통해 채널별 가중치를 생성한다.
3. **Highlight**: 생성된 가중치를 차원 축소된 특징 맵에 곱하여 타겟 관련 특징을 강조한다.

수식으로 표현하면, 입력 특징 텐서 $f \in \mathbb{R}^{H \times W \times C_{in}}$에 대해 최종 출력 $W(f)$는 다음과 같다.
$$W(f) = \pi(f) \cdot \text{conv}(f)$$
여기서 가중치 $\pi(f)$는 다음과 같이 계산된다.
$$\pi(f) = \delta(\text{conv}_2(\text{conv}_1(\text{avg}(f))))$$
$\text{avg}$는 평균 풀링을, $\text{conv}_1, \text{conv}_2$는 가중치 생성을 위한 컨볼루션 레이어를 의미한다.

### Corrective Loss

분류 브랜치와 회귀 브랜치의 정렬 불일치를 해결하기 위해 제안된 손실 함수이다.

먼저, 회귀 손실 $L_{reg}$는 안정성을 위해 Smooth L1 손실과 IoU 손실을 결합하여 정의한다.
$$L_{reg} = L(\text{Smooth L1}) + L_{IoU}$$
여기서 $L_{IoU} = 1 - \text{IoU}$이다.

핵심은 양성 샘플(Positive Sample) $x_i$에 대한 손실 함수 $L_{pos}$에 분류 손실 $\text{CE}$를 가중치로 적용하는 것이다.
$$L_{pos} = \text{CE}(p_i, y_i) + (1 + e^{-\text{CE}(p_i, y_i)}) L_{reg}$$
이 식에서 $(1 + e^{-\text{CE}})$ 항은 분류 성능이 좋을수록(즉, $\text{CE}$가 낮을수록) 회귀 손실에 더 큰 가중치를 부여하게 된다. 이를 통해 모델은 분류 점수가 높은 샘플에 대해 더 정확한 바운딩 박스를 생성하도록 강제되며, 결과적으로 두 브랜치 간의 예측 일관성이 높아진다.

전체 손실 함수 $L_{all}$은 양성 샘플과 음성 샘플(단순 $\text{CE}$ 적용)의 평균으로 계산된다.
$$L_{all} = \frac{1}{N} \left( \sum_{i \in \text{pos}} L_{pos} + \sum_{j \in \text{neg}} \text{CE}(p_j, y_j) \right)$$

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-2015, VOT2016, DTB70, UAV123, UAV20L 등 5개의 벤치마크 데이터셋에서 평가하였다. 학습에는 GOT-10k, COCO, ImageNet VID/DET를 사용하였다.
- **지표**: Precision Plot, Success Plot(AUC), EAO, Accuracy, Robustness 등을 사용하였다.
- **환경**: Nvidia RTX 3090 GPU 사용, 수정된 ResNet-50 백본 기반.

### 주요 결과

- **정량적 성과**:
  - **VOT2016**: EAO 0.510, Robustness 0.126을 기록하며 비교 모델 중 가장 우수한 성능을 보였다.
  - **OTB-2015**: 특히 배경 clutter(Background Clutters)와 평면 외 회전(Out-of-Plane Rotation) 상황에서 강점을 보였으며, 성공률과 정밀도가 baseline 대비 각각 0.013, 0.023 향상되었다.
  - **UAV 데이터셋**: DTB70, UAV123, UAV20L 모두에서 SOTA급 성능을 달성하였다. 특히 UAV20L의 다양한 속성 분석 결과, 시야 밖(Out-of-View), 스케일 변화, 유사 객체 상황에서 매우 강건함을 입증하였다.
- **효율성**: 38 fps의 속도로 작동하여 실시간 추적이 가능함을 보였다.
- **계산 복잡도**: SiamBAN 대비 연산량은 0.31 GFlops, 파라미터 수는 0.84 M 증가하는 수준으로, 매우 가벼운 모듈임을 확인하였다.

### 소거 연구 (Ablation Study)

- **THM의 효과**: SiamBAN에 THM을 추가했을 때 성공률이 0.543에서 0.582로 상승하였다. 기존의 일반적인 어텐션(SE block)을 추가한 것보다 성능 향상 폭이 훨씬 컸다.
- **Corrective Loss의 효과**: Corrective Loss를 적용했을 때 성공률이 0.543에서 0.568로 향상되었으며, THM과 함께 적용했을 때 최종적으로 0.594의 최고 성능을 달성하였다.

## 🧠 Insights & Discussion

본 논문은 Siamese 추적기의 고질적인 문제인 '배경 민감도'와 '브랜치 간 불일치'를 매우 효율적인 방법으로 해결하였다.

가장 주목할 점은 **THM의 배치 위치**이다. 기존의 많은 연구가 백본의 특징 추출 단계에서 어텐션을 사용한 것과 달리, 본 연구는 유사도 계산 직전의 차원 축소 단계에 THM을 배치함으로써 "유사도 응답 맵을 타겟에 집중시킨다"는 목적을 직접적으로 달성하였다. 이는 단순히 특징을 강화하는 것보다 더 실질적인 성능 향상으로 이어졌음을 실험적으로 보여준다.

또한, **Corrective Loss**는 네트워크의 구조적 복잡도를 높이지 않고 오직 학습 전략(손실 함수)만으로 분류-회귀 브랜치의 정렬 문제를 해결했다는 점에서 매우 실용적이다. 이는 추가적인 파라미터 증가 없이도 추론 단계에서의 예측 신뢰도를 높일 수 있는 영리한 접근 방식이다.

다만, 본 논문은 ResNet-50 백본을 기반으로 하고 있으며, 최근의 Transformer 기반 추적기들과의 직접적인 정량적 비교보다는 기존 Siamese-RPN 계열의 발전 방향에 집중하고 있다. 향후 연구에서는 Transformer 구조에 THM이나 Corrective Loss와 같은 아이디어를 접목했을 때의 효과를 검증할 필요가 있다.

## 📌 TL;DR

SiamTHN은 Siamese 추적기의 유사도 응답 맵이 배경에 민감한 문제와 분류-회귀 브랜치 간의 예측 불일치 문제를 해결한 모델이다. 이를 위해 타겟 관련 채널을 동적으로 강화하는 **Target Highlight Module(THM)**과 두 브랜치를 공동 최적화하는 **Corrective Loss**를 제안하였다. 그 결과, 추가적인 연산 부담 거의 없이 38 fps의 속도로 다수의 벤치마크 데이터셋에서 SOTA 성능을 달성하였으며, 특히 배경 노이즈가 심하거나 유사 객체가 존재하는 환경에서 탁월한 강건함을 보여준다.
