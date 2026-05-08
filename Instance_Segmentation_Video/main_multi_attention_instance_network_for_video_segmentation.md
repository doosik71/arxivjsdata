# Multi-Attention Instance Network for Video Segmentation

Juan León Alcázar, María A. Bravo, Ali K. Thabet, Guillaume Jeanneret, Thomas Brox, Pablo Arbeláez, and Bernard Ghanem (2019)

## 🧩 Problem to Solve

본 논문은 비디오 내의 여러 객체를 독립적으로 식별하고 분할하는 **Multi-instance video segmentation** 문제를 해결하고자 한다. 기존의 비디오 객체 분할(Video Object Segmentation, VOS) 방법들은 주로 배경에서 전경 객체 하나를 분리하는 이진 분할(binary task)에 집중해 왔으며, 여러 인스턴스를 동시에 처리하는 작업은 여전히 어려운 과제로 남아 있다.

멀티 인스턴스 분할의 주요 난제는 다음과 같다:

1. **일관성 유지**: 시퀀스 전체에 걸쳐 모든 인스턴스에 대해 공간적, 시간적으로 일관된 마스크를 생성하고 정확한 라벨을 할당해야 한다.
2. **의미론적 정의 부족**: 분할 대상이 첫 프레임에서 임의로 선택되므로, 대상 객체에 대한 명확한 클래스 정보(semantics)가 부족하다.
3. **가변성**: 비디오 진행에 따라 객체의 외형, 크기, 가시성이 계속해서 변화한다.
4. **복잡한 역동성**: 각 비디오 시퀀스마다 고유하고 복잡한 장면 역동성이 존재하며, 이를 도메인 특정 지식(domain-specific knowledge) 없이 모델링하기 어렵다.

기존의 One-shot VOS 방법들은 오프라인 학습 후 각 비디오마다 모델을 미세 조정하는 **Online Training** 단계에 크게 의존한다. 하지만 이는 계산 비용이 매우 높고 대규모 데이터셋에 적용하기 부적합하다는 한계가 있다. 따라서 본 논문의 목표는 온라인 학습 없이, 단일 forward pass만으로 여러 인스턴스를 효율적으로 분할할 수 있는 범용적인 오프라인 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 도메인 특정 지식이나 온라인 미세 조정 없이, **범용적인 시공간 어텐션 큐(generic spatio-temporal attention cues)를 통합**하여 멀티 인스턴스 분할을 수행하는 것이다. 주요 기여 사항은 다음과 같다:

1. **단일 Forward Pass 기반 멀티 인스턴스 분할**: 도메인 특정 지식이나 인스턴스별 미세 조정 없이, 한 번의 추론으로 임의의 개수(최대 $N$개)의 인스턴스를 동시에 분할하는 아키텍처를 제안한다.
2. **Weighted Instance Dice (WID) Loss**: 클래스 불균형이 심한 데이터셋에서 작은 객체에 대한 페널티를 강화하고 인스턴스 간 겹침을 방지하는 새로운 손실 함수를 도입하였다.
3. **시공간 큐의 통합**: 정적 특징(RGB), 단기 모션 큐(Unit Optical Flow), 단기 시간적 큐(STA), 장기 시간적 큐(LTA)를 단일 엔드 투 엔드 네트워크에 통합하였다.
4. **효율적인 디코더 설계**: Dilated Separable Convolution을 이용한 멀티 스케일 디코더를 통해 계산 비용을 낮추면서도 광범위한 수용 영역(receptive field)을 확보하였다.

## 📎 Related Works

기존의 비디오 객체 분할 연구는 단순한 이미지 분할의 확장판에서 시작하여, 첫 프레임의 정답(ground-truth)을 이용하는 **One-shot learning** 태스크로 진화하였다.

- **One-Shot VOS**: 많은 최신 방법론들이 이진 오프라인 모델을 학습시킨 후, 온라인 단계에서 특정 인스턴스에 맞게 모델을 최적화한다. 그러나 이러한 방식은 멀티 인스턴스 시나리오에서 각 인스턴스마다 별도의 온라인 학습 세션을 거쳐야 하므로 매우 비효율적이다.
- **손실 함수**: VOS 데이터셋은 배경 비중이 매우 높아 클래스 불균형 문제가 심각하다. Focal Loss나 Dice Loss 등이 제안되었으나, 본 논문은 멀티 인스턴스 상황에서 인스턴스 크기에 따른 가중치를 부여하는 WID Loss를 통해 이를 개선하고자 한다.
- **시간적 큐(Temporal Cues)**: 이전 연구들은 Dense Optical Flow 기반의 궤적(trajectory)이나 단순한 모션 정보를 사용했다. 본 논문은 이를 확장하여 트래킹 알고리즘 기반의 장기 어텐션(LTA)과 이전 프레임 마스크를 워핑(warping)한 단기 어텐션(STA)을 모두 활용한다.

## 🛠️ Methodology

### 전체 시스템 구조

MAIN은 단일 엔코더-디코더 구조로, 입력으로 RGB 이미지, Unit Optical Flow(uOF), 장기 어텐션(LTA), 단기 어텐션(STA)을 함께 받는다. 출력은 $N \times H \times W$ 차원의 텐서로, 여기서 $N$은 데이터셋 내 최대 인스턴스 개수이다.

### 1. Weighted Instance Dice (WID) Loss

멀티 인스턴스 시나리오의 특성상 작은 객체의 오차가 전체 성능에 큰 영향을 미치며, 서로 다른 인스턴스가 겹쳐 예측되는 문제가 발생한다. 이를 해결하기 위해 다음과 같은 WID 손실 함수를 정의한다:

$$WID(P,G) = \sum_{i}^{n} \alpha(g_i)(1 - D(p_i, g_i)) + \sum_{i}^{n} \sum_{j \neq i}^{n} D(p_i, p_j)$$

- $P, G$: 각각 예측된 인스턴스 세트와 정답(Ground-truth) 세트이다.
- $D(p, g)$: 표준 Dice coefficient이다.
- $\alpha(g_i)$: 인스턴스 크기에 반비례하는 가중치로, $\alpha(g_i) = 1 - \frac{|g_i|}{W H}$로 정의되어 작은 객체일수록 더 큰 페널티를 부여한다.
- $\sum \sum D(p_i, p_j)$: 서로 다른 인스턴스 예측값 간의 겹침을 페널티로 부여하여 인스턴스 할당 오류를 줄인다.

### 2. Attention Priors (어텐션 큐)

네트워크는 다음과 같은 네 가지 정보를 입력으로 사용하여 시공간적 맥락을 파악한다.

- **Unit Optical Flow (uOF)**: FlowNet 2.0으로 계산한 광학 흐름(Optical Flow) 벡터 $\mathbf{o} = (x, y)$를 단위 방향 벡터 $\hat{\mathbf{o}} = \frac{\mathbf{o}}{|\mathbf{o}|}$와 정규화된 크기(magnitude)로 변환한 3채널 큐이다. 이는 큰 변위나 추정 오류가 있을 때 일반적인 벡터 필드보다 더 안정적이다.
- **Long-term Attention (LTA)**: DA-Siam-RPN 트래커를 사용하여 객체의 대략적인 위치를 바운딩 박스(bounding box) 형태로 추적한다. 이는 객체의 전역적인 위치 정보를 제공한다.
- **Short-term Attention (STA)**: 이전 프레임($t-1$)의 분할 결과 마스크를 현재 프레임($t$)으로 워핑(warping)하여 생성한다. $\text{warp}(S_t, o_t) = S_{t+1}$ 관계를 이용하여 타겟의 즉각적인 궤적과 대략적인 외형을 제공한다.

### 3. Multi-Scale Separable Decoder

디코더는 Feature Pyramid Network (FPN) 구조를 따르며, 다음과 같은 연산을 통해 효율성을 높였다.

- **Dilated Convolutions**: 해상도 손실 없이 수용 영역을 기하급수적으로 확장한다.
- **Separable Convolutions**: 채널 간 상관관계와 공간적 상관관계를 분리하여 계산 비용을 줄이고 추론 속도를 높인다.
- **구조**: ResNet-50 백본의 각 블록 끝에서 사이드 출력(side outputs)을 추출하고, 이를 2배 업샘플링한 뒤 $[1 \times 1, 3 \times 3 (\text{dilation } 1), 3 \times 3 (\text{dilation } 2), 3 \times 3 (\text{dilation } 3)]$ 순의 separable convolution 스택을 통과시켜 점진적으로 융합한다.

### 4. 학습 전략

- **Instance Shuffle**: 데이터셋 내 인스턴스 개수의 분포가 불균형하여(대부분 1~3개), 인스턴스 채널을 무작위로 셔플하여 학습시킴으로써 모든 출력 채널이 균등하게 학습되도록 유도한다.
- **Curriculum Learning**: 트래커의 오류에 적응시키기 위해, 처음에는 정답 마스크로 생성한 완벽한 LTA/STA를 사용하다가 점차 트래커가 예측한 실제 큐로 교체하며 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: Youtube-VOS (대규모, seen/unseen 구분), DAVIS-17 (고품질, 소규모).
- **지표**: Jaccard Index ($\mathcal{J}$, 영역 유사도) 및 F-measure ($\mathcal{F}$, 윤곽선 정확도 및 시간적 안정성).
- **비교 대상**: OSVOS, OSMN, OnAVOS 등 최신 오프라인 및 온라인 VOS 방법론.

### 주요 결과

1. **정량적 성능**:
   - **Youtube-VOS**: 오프라인 설정에서 SOTA를 달성하였다. 특히 학습 시 보지 못한 객체 카테고리인 **Unseen** 지표에서 $\mathcal{J}$는 6.8%, $\mathcal{F}$는 12.7% 향상되어 매우 강력한 일반화 성능을 보였다.
   - **DAVIS-17**: 오프라인 방법론 중 최고 성능을 기록하였으며, 온라인 방법론과 비교해도 경쟁력 있는 수치를 보였다.
2. **효율성**: 추론 속도가 **30.3 FPS**로 실시간 처리가 가능하며, 이는 기존 SOTA 방법론들보다 한 차원 더 빠른 속도이다.
3. **Ablation Study**:
   - LTA가 전체 성능 향상에 가장 결정적인 역할을 하며, STA는 경계선을 정교화(refinement)하는 데 기여함을 확인하였다.
   - WID Loss가 기존의 Binary Cross Entropy나 단순 Dice Loss보다 멀티 인스턴스 분할에서 훨씬 우수한 성능을 보였다.
   - 단일 인스턴스를 여러 번 예측하는 것보다 멀티 인스턴스를 한 번에 예측하는 방식이 더 효과적이었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 **온라인 학습 없이도 온라인 학습 기반 모델에 근접하는 성능**을 낸 것이다. 이는 구체적인 객체의 의미론적 정보(semantics)에 의존하는 대신, uOF, LTA, STA와 같은 범용적인 시공간 큐를 효과적으로 융합했기 때문이다. 특히 Unseen 카테고리에서의 높은 성능은 모델이 특정 클래스의 외형을 암기한 것이 아니라, 비디오의 일반적인 움직임과 어텐션 메커니즘을 통해 객체를 추적하고 분할하고 있음을 시사한다.

### 한계 및 비판적 해석

논문에서 제시된 정성적 결과(Figure 8, 12 등)를 보면, **시각적으로 매우 유사한 인스턴스가 겹치거나(overlap) 복잡하게 상호작용할 때** 인스턴스 라벨이 서로 뒤바뀌는(label switching) 문제가 발생한다. 이는 모델이 클래스 정보 없이 오직 어텐션 큐에만 의존하기 때문에 발생하는 근본적인 한계로 보인다. 또한, 매우 빠르게 움직이는 객체의 경우 uOF나 트래커(LTA)의 오차가 커져 분할 성능이 저하되는 경향이 있다.

## 📌 TL;DR

본 논문은 온라인 미세 조정 없이 단일 forward pass로 여러 객체를 분할하는 **Multi-Attention Instance Network (MAIN)**를 제안한다. **WID Loss**라는 새로운 손실 함수와 **LTA(장기), STA(단기), uOF(모션)** 큐를 통합한 구조를 통해 Youtube-VOS와 DAVIS-17 벤치마크에서 오프라인 SOTA 성능을 달성하였다. 특히 실시간 속도(30.3 FPS)와 높은 일반화 능력을 갖추어, 대규모 비디오 데이터셋에 효율적으로 적용 가능한 실용적인 프레임워크를 제시하였다.
