# Curriculum Learning: A Survey
Petru Soviany, Radu Tudor Ionescu, Paolo Rota, Nicu Sebe

## 🧩 Problem to Solve
기존의 기계 학습 모델 훈련은 데이터를 무작위로 섞어 사용하며, 이는 인간이 쉬운 개념부터 어려운 개념까지 점진적으로 학습하는 방식과 다릅니다. 이러한 무작위 학습 방식은 수렴 속도와 최종 성능 측면에서 한계가 있습니다. 커리큘럼 학습(Curriculum Learning, CL)은 이러한 문제점을 해결하기 위해 쉬운 샘플부터 어려운 샘플로 순서를 정해 훈련하는 방식을 제안하지만, '쉬움'과 '어려움'을 정의하는 기준(난이도 측정)과 어려운 데이터를 도입하는 속도(페이싱 함수)를 결정하는 것이 주요 과제입니다. 또한, CL이 다양한 분야에서 성공적인 결과를 보였음에도 불구하고 주류 학습 전략으로 널리 채택되지 못하고 있습니다.

## ✨ Key Contributions
- 기존의 커리큘럼 학습 방법론들을 **단일한 일반화된 틀 아래 정형화**하고, 이를 통해 커리큘럼 학습의 일반적인 공식을 정의했습니다.
- 커리큘럼 학습이 데이터($E$), 모델($M$), 태스크($T$), 그리고 성능 측정($P$) 등 기계 학습 접근 방식의 네 가지 주요 구성 요소에 어떻게 적용될 수 있는지 분석했으며, 이들이 모두 **손실 함수 스무딩(loss function smoothing)**이라는 공통된 해석으로 연결됨을 보였습니다.
- 데이터 유형, 태스크, 커리큘럼 전략, 순위 기준, 커리큘럼 스케줄 등 다양한 분류 기준을 고려하여 **다중 관점의 커리큘럼 학습 방법론 분류 체계(taxonomy)를 수동으로 구축**했습니다.
- 수동으로 구축한 분류 체계를 응집형 클러스터링(agglomerative clustering) 알고리즘을 사용하여 **자동으로 구축한 계층적 트리(dendrogram)와 상호 검증**하여 분류 체계의 타당성을 높였습니다.
- 커리큘럼 학습의 이점을 강조하고, 이 전략이 주류 기계 학습 연구에서 더 널리 채택되도록 **커리큘럼 학습의 광범위한 적용을 옹호**했습니다.

## 📎 Related Works
- **Narvekar et al. (2020)**: 강화 학습(Reinforcement Learning, RL)에 적용된 커리큘럼 학습에 초점을 맞춘 설문 조사를 진행했습니다. 본 논문은 RL에 국한되지 않는 더 일반적인 관점을 제시합니다.
- **Wang et al. (2021)**: 커리큘럼 학습의 동기, 정의, 이론 및 여러 잠재적 응용 분야를 다루는 유사한 설문 조사입니다. 본 논문은 더 포괄적인 검토와 함께 자동 클러스터링을 통해 새로운 관점을 제공합니다.
- **Hard Example Mining (HEM)**: Shrivastava et al. (2016) 등이 제시한 어려운 예제에 집중하는 전략으로, 커리큘럼 학습과는 상반되는 접근 방식입니다.
- **Anti-curriculum**: Braun et al. (2017) 등이 제시한 어려운 예제부터 학습하는 역순 커리큘럼입니다.
- **Active Learning (AL)**: Chang et al. (2017) 등은 예제의 난이도가 아닌 불확실성(uncertainty)에 기반한 샘플 선택을 제안하며, 이는 SPL(Self-Paced Learning) 및 HEM과 보완적일 수 있습니다.

## 🛠️ Methodology
본 논문은 커리큘럼 학습(CL)의 일반적인 프레임워크를 정의하고, 기존 방법론들을 체계적으로 분류합니다.

### 커리큘럼 학습의 일반화된 공식
- **최적화 문제로서의 CL**: CL은 비볼록 최적화(non-convex optimization)에서 잘 알려진 '연속화 방법(continuation method)'으로 볼 수 있습니다. 이는 초기에는 최적화하기 쉬운 단순(매끄러운) 목적 함수에서 시작하여, 점진적으로 덜 매끄러운 버전으로 전환하며 원래의 비볼록 목적 함수에 도달하는 방식입니다.
- **적용 수준**: Mitchell (1997)의 기계 학습 정의($E$: 경험, $M$: 모델, $T$: 태스크, $P$: 성능 측정)에 따라 CL을 적용할 수 있습니다. 대부분 데이터($E$) 수준에서 적용되지만, 모델 용량($M$)이나 태스크 복잡성($T$)을 점진적으로 늘리는 방식도 동등한 손실 함수 스무딩 효과를 가져옵니다.
- **핵심 구성 요소 (알고리즘 1)**:
    - **커리큘럼 기준($C$)**: 샘플/태스크의 난이도를 결정하는 방법 (예: 이미지의 모양 복잡성, 텍스트의 문장 길이, SNR, 교사 네트워크, 모델의 학습 진행 상황).
    - **커리큘럼 수준($l$)**: 커리큘럼을 적용할 대상 (데이터, 모델, 태스크, 성능 측정).
    - **스케줄러($S$)**: 커리큘럼을 언제 업데이트할지 결정하는 함수 (예: 선형/로그 페이싱, 현재 성능 기반).
    - **선택 방법**: 현재 난이도 수준에 맞는 샘플 서브셋($E^*$)을 선택하는 방식 (예: 배치 처리, 가중치 부여, 샘플링).

### 커리큘럼 학습 방법론 분류 (수동 분류)
본 논문은 7가지 주요 커리큘럼 학습 범주를 제안합니다:
- **Vanilla CL**: 사전에 정의된 규칙 기반의 난이도 기준으로 샘플을 선택. (예: Bengio et al., 2009)
- **Self-Paced Learning (SPL)**: 모델의 현재 성능에 따라 샘플의 난이도를 반복적으로 측정하여 훈련 순서를 동적으로 변경. (예: Kumar et al., 2010)
- **Balanced CL (BCL)**: 전통적인 CL 기준 외에 샘플의 다양성까지 고려하여 여러 정렬 기준을 도입. (예: Soviany, 2020)
- **Self-Paced CL (SPCL)**: 사전 정의된 기준과 모델의 학습 기반 측정 지표를 함께 사용하여 훈련 순서를 정의. (예: Jiang et al., 2015)
- **Progressive CL (PCL)**: 개별 샘플의 난이도가 아닌, 모델 용량이나 태스크 설정을 점진적으로 변화시킴. (예: Karras et al., 2018)
- **Teacher-Student CL**: 보조 모델(교사)이 주 모델(학생)의 최적 학습 파라미터를 결정하여 커리큘럼을 구현. (예: Kim and Choi, 2018)
- **Implicit CL (ICL)**: 특정 훈련 방법론의 부수적인 효과로 쉬움-어려움 스케줄이 자연스럽게 발생. (예: Sinha et al., 2020)

### 계층적 클러스터링 (자동 분류)
- **방법**: 모든 논문의 초록에서 추출한 어휘를 기반으로 TF-IDF 벡터를 생성하고, 워드 연결(Ward's linkage) 기반의 응집형 클러스터링 알고리즘을 적용했습니다.
- **목표**: 수동으로 정의한 분류 체계가 객관적인 데이터 기반 클러스터링과 일치하는지 검증합니다.

## 📊 Results
- **성능 향상**: 커리큘럼 학습은 다양한 기계 학습 태스크에서 수렴 속도 향상과 더 나은 정확도를 제공함이 입증되었습니다. 특히 컴퓨터 비전, 자연어 처리, 음성 처리, 의료 영상, 강화 학습 등 광범위한 분야에서 성공적으로 적용되었습니다.
- **분류 체계의 유효성 검증**:
    - 자동화된 계층적 클러스터링 결과(덴드로그램)는 수동으로 정의한 분류 체계와 상당한 일관성을 보였습니다.
    - 가장 큰 동질적 클러스터는 SPL 방법론이었으며, 두 번째로 큰 클러스터는 강화 학습 방법론이었습니다. 이는 SPL이 CL과 독립적으로 발전해왔으며, 강화 학습 분야의 CL이 다른 도메인과 뚜렷한 차이점을 보인다는 점(주로 태스크 수준의 커리큘럼)과 일치합니다.
    - 도메인 적응(Domain Adaptation) 및 음성 처리 방법론들도 별도의 동질적 클러스터를 형성했습니다.
    - 이미지 분류 및 텍스트 처리 방법론은 하나의 큰 이질적 클러스터에 속했지만, 세부적인 하위 클러스터는 명확하게 구분되지 않았습니다.

## 🧠 Insights & Discussion
### 일반적인 논의 및 향후 방향
- **데이터 다양성 문제**: 커리큘럼 학습이 항상 성능 향상을 가져오는 것은 아니며, 난이도 측정 기준이 특정 클래스의 쉬운 예제에만 편향될 경우 데이터 다양성을 저해하여 최적화 과정이 비효율적이 될 수 있습니다. 이는 커리큘럼 학습 적용 시 추가적인 요소를 제어해야 함을 시사합니다.
- **모델 및 성능 수준 커리큘럼 부족**: 대부분의 CL 연구는 데이터 또는 태스크 수준에 집중되어 있으며, 모델 용량($M$) 또는 성능 측정($P$) 수준의 커리큘럼은 충분히 탐구되지 않았습니다. 이러한 방식은 명시적인 난이도 측정 없이도 적용 가능하므로, 향후 더 효율적인 접근법을 모색할 필요가 있습니다.
- **비지도 및 자기지도 학습에서의 CL 부족**: 비지도 학습(Unsupervised Learning) 및 자기지도 학습(Self-Supervised Learning) 분야에서 커리큘럼 학습 연구가 부족합니다. 레이블 없이 쉬운 샘플부터 학습하는 것은 비지도 모델 최적화에 좋은 시작점이 될 수 있으며, 특정 상황에서는 데이터 다양성을 줄이는 것이 도움이 될 수도 있습니다.
- **CL과 SGD의 관계 부족한 이해**: 커리큘럼 학습이 신경망에 주로 적용되는 이유는 비볼록 목적 함수를 가지기 때문이며, 이는 SGD(Stochastic Gradient Descent)의 무작위성과 관련이 있습니다. CL은 SGD가 지역 최솟값(local minimum)에 수렴하도록 제한할 수 있어, 최적화 제어가 어려워질 수 있습니다. 향후 연구에서는 자동화된 훈련 프로세스 조절이나 진화 알고리즘과 같은 대체 최적화 방법과의 결합을 탐색할 필요가 있습니다.

### 도메인별 향후 방향
- **컴퓨터 비전**: Vision Transformer 모델에 CL을 적용하여 사전 훈련 단계(자기지도 태스크의 난이도 순서화)와 미세 조정 단계(데이터 수준 난이도 예측, 모델 수준 토큰 스무딩) 모두에서 수렴 속도와 정확도를 향상시킬 수 있습니다.
- **의료 영상**: 의료 영상용 Transformer 모델에 CL을 적용할 수 있습니다. 데이터 수준 CL의 경우, 건강한 조직 또는 병변이 우세한(구분하기 쉬운) 이미지부터 학습을 시작하는 것이 효과적일 수 있습니다.
- **자연어 처리**: 인간의 언어 학습 방식(제한된 어휘에서 점진적 확장)을 모방한 어휘 크기 기반의 CL을 설계하여 언어 Transformer의 성능을 향상시킬 수 있습니다. 초기 어휘 선정 및 확장 시기를 결정하는 것이 중요합니다.
- **신호 처리**: 신호 노이즈 수준이나 소스 개수와 같은 도메인별 특성을 고려한 맞춤형 CL 전략을 개발하여 노이즈 제거 및 소스 분리 등의 문제를 해결할 수 있습니다.

## 📌 TL;DR
본 설문조사는 기계 학습 모델 훈련 시 데이터를 **쉬운 것부터 어려운 것 순서로 제시**하는 **커리큘럼 학습(CL)**을 다룹니다. 기존 훈련 방식의 한계(수렴 속도, 성능)를 개선하고자 하는 CL은 데이터를 조작하는 것이 핵심이지만, **난이도 정의 및 페이싱**이라는 과제를 안고 있습니다.

논문은 CL을 **손실 함수 스무딩을 위한 연속화 방법**으로 일반화하고, 이를 데이터($E$), 모델($M$), 태스크($T$), 성능($P$) 등 다양한 수준에 적용할 수 있음을 보입니다. 또한, CL 방법론을 **바닐라 CL, SPL, BCL, SPCL, PCL, Teacher-Student CL, Implicit CL**의 7가지 유형으로 분류하는 다중 관점의 분류 체계를 제시하며, 이를 자동화된 **계층적 클러스터링 결과로 검증**합니다.

주요 결과로 CL이 **컴퓨터 비전, 자연어 처리, 음성 처리, 의료 영상, 강화 학습** 등 광범위한 도메인에서 모델 성능 향상에 기여함을 확인했습니다. 또한, **데이터 다양성 저해 가능성, 모델/성능 수준 CL의 미탐색, 비지도/자기지도 학습에서의 잠재력, 그리고 SGD와의 복잡한 상호작용** 등 CL의 한계점과 향후 연구 방향을 제시하며, 특히 **Transformer 모델**과의 결합 가능성을 강조합니다.