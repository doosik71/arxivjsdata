# 3D Indoor Instance Segmentation in an Open-World

Mohamed El Amine Boudjoghra et al. (2023)

## 🧩 Problem to Solve

본 논문은 3D 실내 인스턴스 분할(3D Indoor Instance Segmentation) 분야에서 기존의 '폐쇄 세계(Closed-world)' 가정의 한계를 극복하고자 한다. 기존 방법론들은 학습 단계에서 정의된 시맨틱 클래스들만을 추론 단계에서 분할할 수 있다고 가정하며, 학습되지 않은 클래스는 단순히 배경(background)으로 처리하거나 무시하는 경향이 있다.

그러나 실제 환경에서는 학습 데이터에 포함되지 않은 수많은 미지의 객체들이 존재한다. 이러한 환경에서 모델이 다음과 같은 능력을 갖추는 것이 본 연구의 목표이다.
1. 학습된 '기존 클래스(Known classes)'를 정확히 식별하고 분할하는 것.
2. 학습되지 않은 객체를 '미지 클래스(Unknown class)'로 인식하여 구분해내는 것.
3. 새로운 클래스에 대한 레이블이 제공되었을 때, 기존 지식을 잊지 않고 점진적으로 학습(Incremental Learning)하는 것.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 미지 객체를 식별하기 위한 전용 메커니즘을 도입하고, 이를 위해 의사 레이블링(Pseudo-labeling)과 임베딩 공간의 분리, 그리고 추론 단계의 확률 보정을 결합한 것이다.

주요 기여 사항은 다음과 같다.
- **Open-World 3D Instance Segmentation 프레임워크**: 미지 객체를 정확히 식별하고 점진적으로 학습할 수 있는 최초의 3D 실내 인스턴스 분할 방법론을 제안한다.
- **Confidence Thresholding (CT) 기반 자동 레이블링**: 학습 시 미지 클래스에 대한 고품질의 의사 레이블을 생성하여 모델의 성능 저하를 막고 클래스 간 분리도를 높인다.
- **Reachability-based Probability Correction (PC)**: 추론 단계에서 객체의 특성과 기존 클래스 프로토타입과의 거리를 계산하여 미지 클래스의 예측 확률을 보정함으로써 인식률을 높인다.
- **현실적인 Open-World 데이터셋 분할(Splits)**: 객체 빈도(Frequency), 지역 특성(Region), 무작위성(Randomness)을 고려한 세 가지 벤치마크 설정을 제안하여 엄격한 평가 체계를 구축하였다.

## 📎 Related Works

### 기존 3D 인스턴스 분할 연구
기존 연구들은 크게 Bottom-up 방식의 그룹화/클러스터링 기반 기법, Top-down 방식의 제안(Proposal) 기반 기법, 그리고 최근의 Transformer 기반 기법(예: Mask3D)으로 나뉜다. 하지만 이 모든 방법은 고정된 시맨틱 레이블 세트 내에서만 작동하는 폐쇄 세계 가정을 전제로 한다.

### Open-World 객체 인식 연구
2D 영역에서는 OW-DETR와 같은 오픈 월드 객체 탐지 연구가 진행되었으며, 3D 영역에서도 일부 시맨틱 분할 연구가 시도되었다. 그러나 기존 3D 오픈 월드 접근 방식은 다음과 같은 한계가 있다.
- 2D Vision Language Model(VLM)에 과도하게 의존하여 3D 고유의 기하학적 특성을 무시한다.
- 새로운 레이블이 추가되었을 때 모델의 성능을 지속적으로 향상시킬 수 있는 점진적 학습 경로가 부족하다.
- 학습을 위해 이미지와 3D 장면의 쌍(pair)이 반드시 필요하다는 제약이 있다.

## 🛠️ Methodology

### 전체 파이프라인
본 모델은 **Mask3D**를 베이스라인으로 사용한다. CNN 기반 백본을 통해 특징 맵을 추출하고, Transformer 디코더를 통해 쿼리($Q$)를 정제하여 클래스 레이블과 인스턴스 마스크를 예측한다. 여기에 오픈 월드 대응을 위한 **Contrastive Clustering**과 **Probability Correction** 모듈이 추가된다.

### 1. 미지 클래스 의사 레이블 생성 (Auto-labeler)
미지 클래스(레이블 0)에 대한 정답지가 없으므로, 학습 과정에서 의사 레이블을 생성한다. 저자들은 기존의 top-k 선택 방식 대신 **Confidence Thresholding (CT)** 방식을 제안한다. 

객체의 Objectness confidence score $s_j$는 다음과 같이 계산된다.
$$s_j = s_{cls,j} \cdot M_j \cdot \frac{1(M_j > 0.5)}{|1(M_j > 0.5)|}$$
여기서 $s_{cls,j}$는 분류 헤드의 최대 출력 확률, $M_j$는 쿼리와 특징 간의 유사도를 나타내는 히트맵이다. 설정한 임계값 이상의 신뢰도를 가지면서 기존 클래스의 마스크와 IoU가 낮은 영역을 미지 클래스의 의사 레이블로 할당한다.

### 2. Contrastive Clustering 및 쿼리 할당
쿼리 임베딩 공간에서 기존 클래스와 미지 클래스를 명확히 분리하기 위해 Contrastive Clustering을 수행한다. 각 클래스별 프로토타입 $Q^p$를 정의하고, **Hinge embedding loss**를 사용하여 동일 클래스 쿼리는 응집시키고 서로 다른 클래스는 밀어낸다.
$$L_{cont}(q_c) = \sum_{i=0}^{|K_t|} \ell(q_c, q_i)$$
$$\ell(q_c, q_i) = \begin{cases} ||q_c - q_i||^2 & i=c \\ \max(0, \Delta - ||q_c - q_i||^2) & i \neq c \end{cases}$$
여기서 $\Delta$는 마진(margin)이며, 헝가리안 매칭(Hungarian matching)을 통해 할당된 쿼리들은 큐(Query Store)에 저장되어 지수 이동 평균(EMA) 방식으로 프로토타입을 업데이트한다.

### 3. Reachability-based Probability Correction (PC)
추론 시, 미지 객체가 기존 클래스와 유사한 특징을 가질 때 발생하는 오분류를 해결하기 위해 확률 보정(PC)을 수행한다. 최종 미지 클래스 확률 $P(0;q_j)$는 분류 헤드의 확률과 보정 확률의 결합으로 결정된다.
$$P(0;q_j) = P_{cls}(0;q_j) \cup P_{corr}(0;q_j)$$

보정 확률 $P_{corr}$은 두 가지 요소의 곱으로 구성된다.
1. **객체성 확률 $P_{corr}(o;q_j)$**: 쿼리가 배경이거나 미지 객체일 확률 (기존 클래스 확률의 합을 1에서 뺀 값).
2. **도달 가능성 확률 $P_{corr}(0;o,q_j)$**: 쿼리가 기존 클래스 프로토타입들로부터 얼마나 멀리 떨어져 있는지에 따른 확률. 이는 시그모이드 함수 $\sigma$를 통해 계산되며, 거리 $\gamma(q_j)$가 멀수록 높은 값을 가진다.
$$\gamma(q_j) = \min_{q_i} ||q_j - q_i||^2$$

### 4. 점진적 학습 (Incremental Learning)
새로운 클래스를 추가 학습할 때 발생하는 치명적 망각(Catastrophic Forgetting)을 방지하기 위해 **Exemplar Replay**를 도입한다. 이전 태스크의 클래스들에 대한 예시 샘플(exemplars)을 유지하며 현재 태스크의 새로운 클래스와 함께 미세 조정(Fine-tuning)을 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet200
- **평가 지표**:
    - **Wilderness Impact (WI)**: 미지 객체가 기존 클래스로 잘못 예측되어 정밀도에 미치는 영향 (낮을수록 좋음).
    - **Absolute Open Set Error (A-OSE)**: 기존 클래스로 오분류된 미지 객체의 절대 수 (낮을수록 좋음).
    - **Unknown-Recall (U-Recall)**: 미지 객체를 얼마나 잘 찾아냈는가 (높을수록 좋음).
    - **mAP**: 기존 클래스에 대한 평균 정밀도.
- **데이터 분할(Splits)**:
    - **Split A (Instance frequency)**: 빈도수가 높은 클래스를 먼저 학습.
    - **Split B (Region-based)**: 로봇이 실내를 탐색할 때 마주칠 법한 지역적 순서로 학습.
    - **Split C (Random)**: 무작위 샘플링을 통한 불균형 상황 가정.

### 주요 결과
- **성능 우위**: 제안된 `3D-OWIS` 모델은 모든 시나리오에서 `3D-OW-DETR` 및 베이스라인인 `Mask3D`보다 뛰어난 성능을 보였다. 특히 기존 클래스의 mAP를 유지하면서 미지 클래스의 Recall을 효과적으로 높였다.
- **구성 요소의 영향 (Ablation)**:
    - **CT(Confidence Thresholding)**가 없으면 저품질의 의사 레이블로 인해 기존 클래스의 성능(mAP)이 크게 하락하고 WI와 A-OSE가 증가한다.
    - **PC(Probability Correction)**를 추가하면 기존 클래스의 성능을 거의 해치지 않으면서 미지 클래스의 Recall을 대폭 향상시킨다.
- **점진적 학습**: Exemplar Replay를 통해 새로운 클래스를 학습한 후에도 이전 클래스에 대한 지식을 성공적으로 유지함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 3D 오픈 월드 환경에서 '미지 객체 식별'과 '점진적 학습'이라는 두 마리 토끼를 잡기 위해 정교한 파이프라인을 구축하였다. 특히 t-SNE 시각화 분석을 통해, 제안 방법론이 미지 클래스의 쿼리들을 기존 클래스 프로토타입으로부터 효과적으로 분리하여 클러스터링하고 있음을 입증하였다. 또한, 단순한 top-k 의사 레이블링보다 신뢰도 임계값(CT)을 사용하는 것이 기존 지식 보존에 훨씬 유리하다는 점을 밝혀낸 것이 중요하다.

### 한계 및 비판적 해석
- **CT의 트레이드오프**: CT는 기존 클래스의 성능을 보호하지만, 학습에 사용되는 의사 레이블의 수를 줄이기 때문에 결과적으로 미지 클래스를 분할하는 자체 능력은 다소 감소시키는 경향이 있다. 이를 PC 모듈로 보완하고 있으나, 근본적인 데이터 부족 문제는 여전하다.
- **PC의 의존성**: 확률 보정(PC)의 효과는 기존 클래스들의 클러스터 특성에 의존한다. 논문에서도 언급되었듯이, 데이터 불균형이 심한 클래스의 경우 보정 성능이 저하될 가능성이 있다.
- **현실적 제약**: Exemplar Replay를 사용하지만, 실제 환경에서 모든 과거 데이터를 보관하는 것이 메모리 측면에서 효율적인지에 대한 추가 논의가 필요하다.

## 📌 TL;DR

본 논문은 3D 실내 인스턴스 분할에서 학습되지 않은 객체를 '미지(Unknown)'로 식별하고 이를 점진적으로 학습하는 **3D-OWIS** 프레임워크를 제안한다. **Confidence Thresholding**을 통한 고품질 의사 레이블 생성, **Contrastive Clustering**을 통한 임베딩 공간 분리, 그리고 추론 시 **Reachability-based Probability Correction**을 통해 기존 클래스의 성능 저하 없이 미지 객체 인식률을 극대화하였다. 이 연구는 향후 자율 주행 로봇이나 AR/VR 시스템이 낯선 환경에서 새로운 객체를 스스로 발견하고 학습하는 능력을 갖추는 데 중요한 기초가 될 것으로 보인다.