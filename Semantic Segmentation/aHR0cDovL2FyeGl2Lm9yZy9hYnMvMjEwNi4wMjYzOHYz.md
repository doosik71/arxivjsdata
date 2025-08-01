# Associating Objects with Transformers for Video Object Segmentation

Zongxin Yang, Yunchao Wei, Yi Yang

## 🧩 Problem to Solve

기존의 비디오 객체 분할(VOS) 방법들은 단일 객체 디코딩에 초점을 맞춰 개발되었습니다. 이로 인해 다중 객체 시나리오에서는 각 객체를 개별적으로 매칭하고 분할한 후 결과를 결합해야 했습니다. 이러한 '사후 결합(post-ensemble)' 방식은 GPU 메모리와 연산 자원을 객체 수에 비례하여 소모하므로, 컴퓨팅 자원이 제한적인 환경에서 다중 객체 VOS의 훈련 및 적용에 큰 제약을 가했습니다. 본 논문은 이러한 비효율성을 해결하고, 도전적인 다중 객체 환경에서 더 효율적이고 성능 좋은 임베딩 학습을 실현하는 방법을 연구합니다.

## ✨ Key Contributions

* **식별 메커니즘(Identification Mechanism) 제안**: 다중 객체를 단일 프레임워크 내에서 효율적으로 연관시키고 동시에 디코딩하는 식별 메커니즘을 제안했습니다. 이 메커니즘 덕분에 다중 객체 훈련 및 추론이 단일 객체 처리만큼 효율적으로 이루어질 수 있음을 처음으로 입증했습니다.
* **장단기 트랜스포머(Long Short-Term Transformer, LSTT) 설계**: 식별 메커니즘을 기반으로 하는 새롭고 효율적인 VOS 프레임워크인 LSTT를 설계했습니다. LSTT는 계층적인 다중 객체 매칭 및 전파를 구성하며, 트랜스포머를 VOS에 적용하여 객체 매칭 및 전파를 위한 계층적 프레임워크를 구축한 첫 사례입니다.
* **최첨단 성능 및 효율성 달성**: YouTube-VOS, DAVIS 2017 등 다중 객체 벤치마크와 DAVIS 2016 등 단일 객체 벤치마크에서 기존 최첨단 방법들을 능가하는 성능을 달성함과 동시에 훨씬 뛰어난 효율성(3배 이상 빠른 다중 객체 실행 시간)을 보여주었습니다. 특히, 3차 대규모 VOS 챌린지에서 1위를 차지했습니다.

## 📎 Related Works

* **준지도 비디오 객체 분할(Semi-supervised VOS)**: 초기 DNN 기반 VOS 방법들은 테스트 시에 네트워크를 미세 조정하여 특정 객체에 집중하도록 했지만, 이는 효율성을 저해했습니다 (예: OSVOS, MoNet, OnAVOS).
* **런타임 개선 방법**: 최근 연구들은 온라인 미세 조정을 피하고 런타임을 개선하는 데 중점을 둡니다.
  * **메모리 네트워크 기반**: STM, EGMN, KMN은 메모리 네트워크를 활용하여 과거 프레임의 특징을 저장하고 비국소적 어텐션 메커니즘으로 현재 프레임을 분할합니다.
  * **픽셀 레벨 매칭 기반**: PML, VideoMatch, FEELVOS, CFBI 등은 픽셀 또는 패치 수준의 매칭을 통해 현재 프레임을 첫 프레임 또는 이전 프레임과 매칭합니다.
  * **트랜스포머 적용**: SST는 트랜스포머 블록을 사용하여 픽셀 수준의 유사성 맵과 시공간 특징을 추출하지만, 과거 프레임의 마스크 정보가 블록 내에서 전파 및 집계되지 않아 타겟 인지(target-aware) 방식과는 차이가 있습니다.
* **시각 트랜스포머(Visual Transformers)**: 자연어 처리(NLP) 분야에서 성공을 거둔 트랜스포머는 최근 이미지 분류, 객체 감지/분할 등 다양한 컴퓨터 비전 작업에 도입되어 CNN 기반 네트워크 대비 유망한 성능을 보이고 있습니다. 본 논문은 이러한 트랜스포머를 VOS의 계층적 어텐션 기반 전파에 효과적으로 적용합니다.

## 🛠️ Methodology

본 논문은 `Associating Objects with Transformers (AOT)`라는 새로운 VOS 프레임워크를 제안합니다. AOT는 주로 두 가지 핵심 메커니즘으로 구성됩니다.

1. **다중 객체 연관을 위한 식별 메커니즘 (Identification Mechanism)**:
    * **목표**: 네트워크가 서로 다른 수의 타겟에 유연하게 적응하도록 하여 다중 객체 마스크 정보를 엔드-투-엔드 방식으로 전파하고 디코딩할 수 있도록 합니다.
    * **식별 임베딩**:
        * $M$개의 식별 벡터($C$ 차원)를 저장하는 식별 뱅크 $D \in \mathbb{R}^{M \times C}$를 초기화합니다.
        * 비디오 장면에 $N$개의 타겟($N < M$)이 있을 경우, 각 타겟에 고유한 식별 벡터를 무작위로 할당합니다.
        * 타겟의 원-핫 마스크 $Y \in \{0,1\}^{THW \times N}$를 식별 임베딩 $E \in \mathbb{R}^{THW \times C}$로 변환하는 공식은 다음과 같습니다:
            $$E = \text{ID}(Y,D) = YPD$$
            여기서 $P \in \{0,1\}^{N \times M}$는 $N$개의 식별 임베딩을 무작위로 선택하기 위한 순열 행렬입니다.
        * 이 식별 임베딩 $E$를 어텐션 값 $V$와 결합하여 메모리 프레임에서 현재 프레임으로 모든 타겟의 식별 정보가 전파되도록 합니다:
            $$V' = \text{AttID}(Q,K,V,Y|D) = \text{Att}(Q,K,V+E)$$
    * **식별 디코딩**:
        * 집계된 특징 $V'$로부터 모든 $M$개의 식별자에 대한 확률 로짓 $L_D \in \mathbb{R}^{HW \times M}$를 컨볼루션 디코딩 네트워크 $F_D$를 사용하여 예측합니다.
        * 할당된 식별자를 선택하고 확률을 계산하여 모든 $N$개 타겟의 최종 확률 예측 $Y' \in [0,1]^{HW \times N}$를 얻습니다:
            $$Y' = \text{softmax}(PF_D(V')) = \text{softmax}(PL_D)$$
    * **훈련**: 식별 뱅크 $D$는 훈련 가능하며, 각 비디오 샘플 및 최적화 반복마다 식별 선택 행렬 $P$를 무작위로 재초기화하여 모든 식별 벡터가 동등하게 경쟁하도록 합니다.
    * **패치별 식별 뱅크 (Patch-wise Identity Bank)**: 높은 해상도 입력 마스크의 픽셀에 직접 식별자를 할당하기 어려운 문제를 해결하기 위해, 각 식별자가 패치 내 16x16 위치에 해당하는 서브-식별 벡터를 가지는 확장된 뱅크를 사용합니다.

2. **계층적 매칭 및 전파를 위한 장단기 트랜스포머 (Long Short-Term Transformer, LSTT)**:
    * **목표**: 단일 어텐션 계층으로는 다중 객체 연관을 충분히 모델링하기 어렵다는 점을 해결하고, 일련의 어텐션 계층을 사용하여 계층적 매칭 및 전파를 구성합니다.
    * **LSTT 블록 구조**:
        * **자기 어텐션(Self-Attention)**: 현재 프레임 내의 타겟들 간의 연관 또는 상관관계를 학습합니다.
        * **장기 어텐션(Long-Term Attention, AttLT)**: 참조 프레임 및 저장된 과거 예측 프레임을 포함하는 장기 메모리 프레임으로부터 타겟 정보를 집계합니다. 시간 간격이 가변적이고 길 수 있으므로 비국소적(non-local) 어텐션을 사용합니다.
            $$\text{AttLT}(X_{t}^{l},X_{m}^{l},Y_m) = \text{AttID}(X_{t}^{l}W_{K}^{l},X_{m}^{l}W_{K}^{l},X_{m}^{l}W_{V}^{l},Y_m|D)$$
            여기서 $X_{t}^{l}$는 현재 프레임 특징, $X_{m}^{l}$과 $Y_m$은 메모리 프레임 특징과 마스크입니다.
        * **단기 어텐션(Short-Term Attention, AttST)**: 현재 프레임의 각 위치에 대해 시공간적으로 인접한 이웃($n$개의 주변 프레임, $\lambda \times \lambda$ 공간 이웃)에서 정보를 집계하여 시간적 부드러움을 학습합니다.
            $$\text{AttST}(X_{t}^{l},X_{n}^{l},Y_n|p) = \text{AttLT}(X_{t}^{l,p},X_{n}^{l,N(p)},Y_{n}^{l,N(p)})$$
            여기서 $X_{t}^{l,p}$는 현재 프레임의 위치 $p$에서의 특징, $N(p)$는 위치 $p$를 중심으로 하는 $\lambda \times \lambda$ 공간 이웃을 나타냅니다.
        * **피드 포워드 모듈**: 일반적인 2계층 MLP로 구성됩니다.
    * **계층적 구조**: 여러 LSTT 블록($L$개의 계층)을 쌓아 깊이를 조절하며 성능과 속도를 유연하게 조절할 수 있습니다.
    * **훈련**: 모바일넷-V2(MobileNet-V2)를 백본 인코더로 사용하며, 합성 비디오 시퀀스로 사전 훈련 후 VOS 벤치마크로 본 훈련을 진행합니다.

## 📊 Results

* **YouTube-VOS (다중 객체)**:
  * R50-AOT-L은 2018/2019 유효성 검사 분할에서 84.1%의 J&F 점수를 달성하여 기존 최첨단 방법(CFBI+, 82.8%)을 능가하며, 14.9 FPS의 효율적인 속도를 유지했습니다.
  * AOT-S (82.6% J&F)는 CFBI+ (82.8%)와 비슷한 성능을 보이면서도 7배 이상 빠른 속도(27.1 FPS vs 4.0 FPS)를 기록했습니다.
  * 가장 작은 모델인 AOT-T는 실시간 속도(41.0 FPS)를 유지하며 80.2% J&F를 달성했습니다.
* **DAVIS 2017 (다중 객체)**:
  * R50-AOT-L은 유효성 검사(84.9%) 및 테스트(79.6%) 분할 모두에서 모든 경쟁자를 능가하며, 18.0 FPS의 효율적인 속도를 유지했습니다. 다중 객체 처리 속도가 단일 객체 처리 속도와 동일하다는 점이 중요합니다.
  * AOT-T (79.9% J&F, 51.4 FPS)는 기존 실시간 방법(SAT, 72.3%)보다 훨씬 뛰어난 성능을 보였습니다.
* **DAVIS 2016 (단일 객체)**:
  * R50-AOT-L은 91.1%의 J&F 점수로 새로운 최첨단 성능을 달성했습니다.
  * AOT-B (89.9% J&F)는 CFBI+ (89.9%)와 비슷한 성능을 보였고 5배 이상 빠른 속도(29.6 FPS vs 5.9 FPS)를 기록했습니다.
* **Qualitative Results**: AOT는 유사한 객체가 많거나 복잡한 다중 객체 시나리오에서 기존 CFBI에 비해 더 정확하게 객체를 추적하고 분할하는 것을 시각적으로 보여주었습니다. 다만, 아주 작은 객체(예: 스키 폴, 시계) 분할에는 한계가 있었습니다.
* **어블레이션 연구**: 식별자 수($M=10$이 최적), 지역 윈도우 크기($\lambda=15$가 좋음), 지역 프레임 수(이전 1프레임이 가장 좋음), LSTT 블록 수(블록 수가 많을수록 성능 향상), 위치 임베딩(상대 위치 임베딩이 중요) 등이 성능에 미치는 영향을 분석하여 각 구성 요소의 중요성을 입증했습니다.

## 🧠 Insights & Discussion

* **다중 객체 처리의 효율성 혁신**: AOT는 다중 객체 비디오 객체 분할에서 여러 객체를 마치 단일 객체처럼 효율적으로 처리할 수 있음을 처음으로 보여주며, 기존의 비효율적인 '개별 처리 후 결합' 패러다임을 바꿨습니다. 이는 컴퓨팅 자원이 제한된 환경에서도 다중 객체 VOS의 광범위한 적용 가능성을 열었습니다.
* **계층적 어텐션의 중요성**: LSTT의 계층적 구조는 다중 객체 정보가 점진적으로 집계되고 연관되어 더 정확한 어텐션 기반 매칭을 가능하게 함을 시각적 분석과 어블레이션 연구를 통해 입증했습니다. 특히, 장기 어텐션과 단기 어텐션의 조합이 전체 성능 향상에 필수적임을 확인했습니다.
* **유연한 성능-속도 균형**: LSTT 블록 수를 조절함으로써 실시간 속도(AOT-T)와 최첨단 성능(SwinB-AOT-L) 사이에서 유연하게 균형을 맞출 수 있음을 보여주었습니다.
* **한계점**: 작은 객체에 대한 특별한 설계가 없어 일부 매우 작은 객체에 대해서는 분할에 실패하는 경우가 있었습니다. 또한, 더 강력하고 효율적인 인코더/디코더 설계는 여전히 개선될 여지가 있는 개방형 문제입니다.
* **향후 연구 방향**: 제안된 식별 메커니즘은 대화형 VOS(interactive VOS), 비디오 인스턴스 분할(video instance segmentation), 다중 객체 추적(multi-object tracking) 등 다중 객체 매칭이 필요한 관련 작업에도 유망하게 적용될 수 있을 것으로 기대됩니다. LSTT의 계층적 구조는 이러한 작업에서 비디오 표현을 처리하는 새로운 솔루션으로 활용될 수 있습니다.

## 📌 TL;DR

기존 VOS 모델의 다중 객체 처리 비효율성 문제를 해결하기 위해, 본 논문은 다중 객체를 단일 프레임워크에서 효율적으로 연관 및 디코딩하는 **식별 메커니즘**과 계층적 객체 매칭 및 전파를 위한 **장단기 트랜스포머(LSTT)**를 포함하는 **AOT(Associating Objects with Transformers)**를 제안합니다. AOT는 다중 객체 VOS가 단일 객체 처리만큼 효율적일 수 있음을 입증하며, 주요 벤치마크에서 기존 최첨단 모델보다 뛰어난 성능과 월등히 빠른 속도를 달성했습니다.
